import chromadb
from sentence_transformers import SentenceTransformer

from app import llm

NOT_FOUND_ANSWER = "I could not find this in the ingested documents."

ANSWER_SYSTEM_PROMPT = """You answer questions about ERP documentation using only the numbered context passages provided.
- Use only facts stated in the context. Do not use outside knowledge.
- Cite the passages you used inline, e.g. [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I could not find this in the ingested documents."
- Be concise: answer in a few sentences or a short list."""

# Load embedding model once
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="erp_docs"
)


def count_tokens(text):
    return len(model.tokenizer(text, add_special_tokens=False)["input_ids"])


def retrieve(question, top_k, document=None):
    """Return the top_k most similar chunks as dicts, best match first."""
    count = collection.count()
    if count == 0:
        return []

    query_args = {
        "query_embeddings": [model.encode(question).tolist()],
        "n_results": min(top_k, count),
    }
    if document:
        query_args["where"] = {"source": document}

    results = collection.query(**query_args)
    if not results["documents"] or not results["documents"][0]:
        return []

    return [
        {
            "rank": rank,
            "source": meta["source"],
            "page": meta["page"],
            "section": meta["section"],
            "chunk_index": meta["chunk_index"],
            "distance": round(distance, 4),
            "text": doc,
        }
        for rank, (doc, meta, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
            start=1,
        )
    ]


def format_context(chunks):
    return "\n\n".join(
        f"[{c['rank']}] ({c['source']}, page {c['page']})\n{c['text']}"
        for c in chunks
    )


def answer_question(question, chunks):
    """Return (answer, mode).

    mode is "llm", "extractive" (no API key configured, the top chunk is returned)
    or "none" (nothing was retrieved).
    """
    if not chunks:
        return NOT_FOUND_ANSWER, "none"

    if not llm.enabled():
        return chunks[0]["text"][:500], "extractive"

    answer = llm.chat([
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ])
    return answer, "llm"
