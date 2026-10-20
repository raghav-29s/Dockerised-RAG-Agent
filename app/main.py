from datetime import datetime, timezone
import io
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi import UploadFile, File
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from app import llm
from app.evaluation import run_evaluation
from app.ingest import extract_pages
from app.rag import answer_question, collection, count_tokens, model, retrieve
from app.utils import chunk_document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

app = FastAPI(
    title="Opkey RAG Agent",
    description="Question answering over ERP documentation with retrieval-augmented generation."
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        examples=["Which transaction is used for logging and tracing ABAP components?"]
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    document: str | None = Field(None, description="Only search chunks from this filename")


class Source(BaseModel):
    rank: int
    source: str
    page: int
    section: str
    chunk_index: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_mode: str = Field(
        description='"llm", "extractive" (no API key configured) or "none" (nothing retrieved)'
    )
    llm_model: str | None
    sources: list[Source]


def _chunk_ids(filename):
    return collection.get(where={"source": filename}, include=[])["ids"]


@app.get("/health")
def health():

    count = collection.count()

    return {
        "status": "ok",
        "docs_indexed": count
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    if request.document and not _chunk_ids(request.document):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document}' is not indexed."
        )

    chunks = retrieve(
        request.question,
        request.top_k,
        request.document
    )

    try:
        answer, mode = answer_question(request.question, chunks)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return {
        "question": request.question,
        "answer": answer,
        "answer_mode": mode,
        "llm_model": llm.LLM_MODEL if mode == "llm" else None,
        "sources": chunks
    }


@app.get("/documents")
def get_documents():

    documents = {}

    for meta in collection.get(include=["metadatas"])["metadatas"]:

        doc = documents.setdefault(meta["source"], {
            "id": meta["source"],
            "chunks": 0,
            "pages": meta.get("document_pages"),
            "ingested_at": meta.get("ingested_at")
        })

        doc["chunks"] += 1

    return {
        "documents": sorted(documents.values(), key=lambda d: d["id"])
    }


# A plain def (not async) so FastAPI runs the CPU-heavy parsing and embedding
# in a worker thread instead of blocking the event loop
@app.post("/ingest")
def upload_document(
    file: UploadFile = File(...)
):

    # Keep only the base name so a crafted filename cannot write outside data/
    filename = os.path.basename(file.filename or "")
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension or filename}'. Upload a .pdf or .txt file."
        )

    content = file.file.read()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Extract text page by page (running footers and TOC lines removed)
    if extension == ".pdf":
        try:
            pages = extract_pages(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read '{filename}' as a PDF: {exc}"
            )
    else:
        try:
            pages = [(1, content.decode("utf-8"))]
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' is not valid UTF-8 text."
            )

    # Section-aware chunking with sentence overlap
    chunks = chunk_document(
        pages,
        count_tokens
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No extractable text found in '{filename}'. Scanned PDFs need OCR, which is not supported."
        )

    # Only files that parsed successfully are kept
    with open(os.path.join("data", filename), "wb") as f:
        f.write(content)

    texts = [chunk["text"] for chunk in chunks]

    # Embeddings
    embeddings = model.encode(
        texts
    ).tolist()

    # Re-ingesting a file replaces its old chunks instead of leaving stale ones behind
    old_ids = _chunk_ids(filename)
    if old_ids:
        collection.delete(ids=old_ids)

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Store in Chroma
    ids = []

    metadatas = []

    for idx, chunk in enumerate(chunks):

        ids.append(
            f"{filename}_{idx}"
        )

        metadatas.append({
            "source": filename,
            "page": chunk["page"],
            "page_end": chunk["page_end"],
            "section": chunk["section"],
            "chunk_index": idx,
            "chunk_size": len(chunk["text"]),
            "document_pages": len(pages),
            "ingested_at": ingested_at
        })

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "message": "Uploaded and indexed",
        "chunks": len(chunks),
        "pages": len(pages),
        "filename": filename,
        "replaced_chunks": len(old_ids)
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):

    ids = _chunk_ids(filename)

    if not ids:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' is not indexed."
        )

    # Only the chunks are removed; the uploaded file stays in data/
    collection.delete(ids=ids)

    return {
        "deleted": True,
        "id": filename,
        "chunks_deleted": len(ids)
    }


@app.get("/evaluate")
def evaluate(
    top_k: int = Query(5, ge=1, le=20, description="Chunks retrieved per question"),
    retrieval_only: bool = Query(
        False,
        description="Only compute hit rate and MRR (seconds instead of minutes, no LLM calls)"
    )
):

    if collection.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents indexed yet. Ingest data/erp_manual.pdf via POST /ingest first."
        )

    return run_evaluation(top_k, retrieval_only)
