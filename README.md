# Dockerised RAG Agent

## Overview

This project is a Dockerized Retrieval-Augmented Generation (RAG) agent that answers natural-language questions about ERP documentation through a REST API. It is designed for large and complex documents like ERP manuals, where finding an answer by hand means navigating dozens of pages.

Documents are uploaded via `POST /ingest`, their text is extracted page by page and split into section-aware chunks (each tagged with its page number and section title), embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in ChromaDB. For each question, the most relevant chunks are retrieved and passed to an LLM (`qwen2.5:3b`, served locally by Ollama), which writes an answer grounded in those chunks with `[n]` citations. The retrieved source chunks are returned alongside the answer so every claim can be traced back to the document.

Everything runs with Docker Compose and **no paid API key is required**: the LLM runs in its own container. ChromaDB data and the downloaded model are stored in Docker volumes, so they persist across restarts.

The knowledge base is the **SAP ERP 6.0 EHP8 Application Operations Guide** (`data/erp_manual.pdf`, 50 pages).

### High-Level Architecture

```text
  INGESTION   POST /ingest (PDF upload)                       ── rag-api container (FastAPI :8000) ──
  ─────────
  pypdf text extraction, page by page
     │    remove running footers, table-of-contents leader dots, soft hyphens
     ▼
  section-aware chunking
     │    split at numbered headings (e.g. 3.2.3.1), then pack whole sentences / table rows
     │    ≤ 180 tokens per chunk, 40-token overlap, section title at the top of every chunk
     ▼
  all-MiniLM-L6-v2 embeddings
     │
     ▼
  ChromaDB (chroma_data volume)      metadata per chunk: source, page, page_end, section, chunk_index
     ▲
     │ top-K similarity search
     │
  QUERY       POST /query {"question", "top_k"}
  ─────
  embed question ──► top-K chunks ──► prompt: rules + question + numbered chunks [1]..[k]
                                                   │
                                                   ▼
                              ollama container: qwen2.5:3b (OpenAI-compatible API, ollama_models volume)
                                                   │
                                                   ▼
                 { "answer": "... [1]", "sources": [ text, file, page, section, distance ] }

  EVALUATION  GET /evaluate
  ──────────
  eval/test_set.json ──► same retrieve + answer path ──► hit rate / MRR (exact) + faithfulness / relevance (LLM judge)
```

# Prerequisites & Setup

## Prerequisites

Before running the project, make sure the following software is installed on your machine:

* Docker Desktop (Docker Engine + Docker Compose v2)
* Git
* Python 3.11+ (only required if you want to run the application locally without Docker)
* About 15 GB of free disk space (app image 2.4 GB, Ollama image 9.2 GB, `qwen2.5:3b` model 1.9 GB)
* About 4 GB of free RAM for the local model. No GPU is needed.

### Verify Installation

```bash
docker --version
docker compose version
git --version
python --version
```

---

## LLM Configuration (no API key needed)

The default `.env.example` uses the free `qwen2.5:3b` model served by the bundled Ollama container. You do not need to create any account or API key.

```env
OPENAI_BASE_URL=http://ollama:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_MODEL=qwen2.5:3b
PORT=8000
```

The app talks to the LLM through the OpenAI-compatible API, so switching to OpenAI `gpt-4o-mini` only needs a config change. Get a key at https://platform.openai.com/api-keys and set in `.env`:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

> Note: API keys are never hardcoded. `.env` is git-ignored; only `.env.example` with placeholder values is committed.

---

# Running the Project with Docker

The application is fully containerized and can be started with a single command.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd opkey-rag-agent
```

### 2. Create Environment File

```bash
cp .env.example .env
```

The defaults work as-is. Edit the file only if you want to use a different LLM (see above), or to publish the API on another port: for example `PORT=9000` serves it at `http://localhost:9000` (the examples below assume the default 8000).

### 3. Build and Start the Application

```bash
docker compose up --build
```

This starts three services:

| Service | Purpose |
| --- | --- |
| `ollama` | Local LLM server |
| `ollama-pull` | One-shot job that downloads `qwen2.5:3b` on first start, then exits |
| `rag-api` | The FastAPI app; starts once the model is available |

The first start downloads the Docker images and the model, which took about 11 minutes on our machine. Later starts take seconds.

---

### 4. Verify the Application

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "docs_indexed": 0
}
```

### 5. Ingest the Sample Document

```bash
curl -X POST http://localhost:8000/ingest \
-F "file=@./data/erp_manual.pdf"
```

`docs_indexed` in `/health` now shows the number of indexed chunks (137).

---

## Access API Documentation

Once the container is running, open:

```text
http://localhost:8000/docs
```

FastAPI's interactive Swagger UI will be available for testing all endpoints.

---

## Stopping the Application

```bash
docker compose down
```

To remove containers, networks, and volumes:

```bash
docker compose down -v
```

> Note : Using `-v` will remove the persisted ChromaDB volume (all indexed document embeddings) and the downloaded model.

## API Endpoint Reference

### 1 Health Check

**Method:** `GET`
**Path:** `/health`

**Description:**
Returns the current health status of the application along with the number of document chunks currently indexed in the vector database.

#### Response Example

```json
{
  "status": "ok",
  "docs_indexed": 137
}
```

#### cURL Example

```bash
curl http://localhost:8000/health
```

---

### 2 Ingest Document

**Method:** `POST`
**Path:** `/ingest`

**Description:**
Uploads a PDF or plain-text document, extracts its text page by page, splits it into section-aware chunks, generates embeddings using the all-MiniLM-L6-v2 model, and stores them in ChromaDB with their page number and section title. Uploading a file with the same name again replaces its previous chunks.

#### Request Format

**Content-Type:** `multipart/form-data`

| Field | Type     | Required |
| ----- | -------- | -------- |
| file  | `.pdf` or `.txt` file | Yes      |

The file is saved to `data/` under its base name only (any directory part of the uploaded filename is dropped), and only after it has been parsed successfully.

#### Response Example

```json
{
  "message": "Uploaded and indexed",
  "chunks": 137,
  "pages": 50,
  "filename": "erp_manual.pdf",
  "replaced_chunks": 0
}
```

`replaced_chunks` is the number of chunks removed because a file with the same name had been ingested before.

#### Errors

| Status | When |
| --- | --- |
| 415 | The file is not a `.pdf` or `.txt` |
| 400 | The file is empty, cannot be read as a PDF, or a `.txt` file is not UTF-8 |
| 422 | No `file` field in the form, or no extractable text (for example a scanned PDF) |

#### cURL Example

```bash
curl -X POST http://localhost:8000/ingest \
-F "file=@./data/erp_manual.pdf"

# Plain-text files work the same way
curl -X POST http://localhost:8000/ingest \
-F "file=@./notes.txt"
```

---

### 3 Query the Agent

**Method:** `POST`
**Path:** `/query`

**Description:**
Embeds the question, retrieves the `top_k` most similar chunks from ChromaDB, and asks the LLM to answer using only those chunks. Returns the generated answer with `[n]` citations plus the retrieved source chunks. If the documents do not contain the answer, the agent replies `"I could not find this in the ingested documents."` instead of guessing.

#### Request Format

**Content-Type:** `application/json`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| question | string | Yes | Non-empty question |
| top_k | integer | No | Chunks to retrieve, 1 to 20 (default 5) |
| document | string | No | Only search chunks from this filename |

#### Request Example

```json
{
  "question": "Which database table grows fast if EHS reports are stored without an external DMS?",
  "top_k": 5
}
```

#### Response Example

```json
{
  "question": "Which database table grows fast if EHS reports are stored without an external DMS?",
  "answer": "Data base table DRAO grows fast if EHS reports are stored without an external DMS. [1]",
  "answer_mode": "llm",
  "llm_model": "qwen2.5:3b",
  "sources": [
    {
      "rank": 1,
      "source": "erp_manual.pdf",
      "page": 24,
      "section": "3.2.3.1 Data Growth and Data Archiving Monitors in EHS",
      "chunk_index": 67,
      "distance": 0.725,
      "text": "3.2.3.1 Data Growth and Data Archiving Monitors in EHS\nMost critical regarding database growth are EHS reports. ... If you do not use an external DMS system, data base table DRAO grows fast and consumes considerable space. ..."
    },
    {
      "rank": 2,
      "source": "erp_manual.pdf",
      "page": 25,
      "section": "3.2.4 Archiving for SAP Product Lifecycle Management (PLM)",
      "chunk_index": 71,
      "distance": 0.94,
      "text": "3.2.4 Archiving for SAP Product Lifecycle Management (PLM)\nData Growth and Data Archiving Monitors\nThe following tables are the fastest growing tables in SAP Product Lifecycle Management ..."
    }
  ]
}
```

The response contains all `top_k` sources (two are shown here, with `text` shortened).

`answer_mode` is `"llm"` normally, `"extractive"` if no LLM is configured (the top chunk is returned as-is), or `"none"` if nothing was retrieved. `distance` is the ChromaDB L2 distance (lower is more similar).

#### Errors

| Status | When |
| --- | --- |
| 404 | `document` is given but no document with that name is indexed |
| 422 | Missing or empty `question`, or `top_k` outside 1 to 20 |
| 502 | The LLM server could not be reached |

#### cURL Example

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{"question":"Which transaction is used for logging and tracing ABAP components?","top_k":5}'
```

---

### 4 List Indexed Documents

**Method:** `GET`
**Path:** `/documents`

**Description:**
Lists the documents in the vector store with their number of chunks, number of pages and ingest time. A document's `id` is its filename, which is what `DELETE /documents/{filename}` expects.

#### Response Example

```json
{
  "documents": [
    {
      "id": "erp_manual.pdf",
      "chunks": 137,
      "pages": 50,
      "ingested_at": "2026-09-12T14:45:25+00:00"
    }
  ]
}
```

#### cURL Example

```bash
curl http://localhost:8000/documents
```

---

### 5 Delete Document

**Method:** `DELETE`
**Path:** `/documents/{filename}`

**Description:**
Removes all indexed chunks of the document with this `id` (its filename) from the vector database. The uploaded file itself stays in `data/`, so it can be ingested again later.

#### Example Request

```text
DELETE /documents/erp_manual.pdf
```

#### Response Example

```json
{
  "deleted": true,
  "id": "erp_manual.pdf",
  "chunks_deleted": 137
}
```

#### Errors

| Status | When |
| --- | --- |
| 404 | No document with this `id` is indexed |

#### cURL Example

```bash
curl -X DELETE http://localhost:8000/documents/erp_manual.pdf
```

---

### 6 Evaluation

**Method:** `GET`
**Path:** `/evaluate`

**Description:**
Runs the evaluation suite in `eval/test_set.json` through the same retrieval and answer path as `/query`, then returns aggregate metrics and a per-question breakdown. See [Evaluation Results](#evaluation-results) for how each metric is computed.

> Note: each of the 22 questions needs two LLM calls (answer + judge). With the default local `qwen2.5:3b` on a CPU-only laptop (Intel i7-1185G7) the full run took **about 17 minutes**, so keep the request open or use the command-line version below. With a hosted model such as `gpt-4o-mini` it takes well under a minute. For a quick check, `GET /evaluate?retrieval_only=true` computes only the retrieval metrics and returns in seconds.

| Query parameter | Default | Description |
| --- | --- | --- |
| top_k | 5 | Chunks retrieved per question (1 to 20) |
| retrieval_only | false | `true` computes only hit rate and MRR and skips the LLM, so it returns in seconds |

Returns `409` if no documents are indexed yet.

#### Response Example

```json
{
  "num_questions": 22,
  "num_answerable": 20,
  "top_k": 5,
  "hit_rate": 0.95,
  "mrr": 0.875,
  "faithfulness": 0.83,
  "answer_relevance": 0.745,
  "answer_correctness": 0.66,
  "abstention_accuracy": 1.0,
  "llm_model": "qwen2.5:3b",
  "results": [
    {
      "id": 1,
      "question": "Which transaction is used for logging and tracing ABAP components?",
      "answerable": true,
      "hit": true,
      "hit_rank": 1,
      "retrieved": ["erp_manual.pdf p.17 #42", "erp_manual.pdf p.17 #44", "erp_manual.pdf p.22 #63", "erp_manual.pdf p.18 #46", "erp_manual.pdf p.17 #43"],
      "answer": "SLG1",
      "faithfulness": 1.0,
      "answer_relevance": 0.5,
      "correctness": 0.5,
      "judge_reason": "The generated answer only mentions SLG1, which is one of the transactions mentioned in the context. However, it does not directly address the question about which transaction is used for logging and tracing ABAP components."
    }
  ]
}
```

`results` contains one entry per question (truncated here).

#### cURL Example

```bash
# Full run: retrieval metrics + LLM answers + LLM-as-judge scores
curl http://localhost:8000/evaluate

# Quick run: hit rate and MRR only
curl "http://localhost:8000/evaluate?retrieval_only=true"
```

The same suite can be run from the command line, which prints the results as Markdown tables:

```bash
docker compose exec rag-api python -m app.evaluation
docker compose exec rag-api python -m app.evaluation --retrieval-only
```


## Design Decisions

### Document Selection

For this project, an ERP (Enterprise Resource Planning) operations manual was used as the primary document source. ERP documentation was selected because it represents large, domain-specific, and real-world enterprise datasets.
Such manuals are typically lengthy and include technical details, configurations, workflows, and procedural guidelines, making manual navigation difficult and time-consuming. This makes them well-suited for a RAG-based system that enables fast, natural language-based information retrieval.
Additionally, ERP documents contain structured sections and consistent terminology, which makes them ideal for evaluating semantic search and vector-based retrieval performance.

### LLM Selection: qwen2.5:3b via Ollama

Answers are generated by `qwen2.5:3b`, served by an Ollama container that is part of the Docker Compose stack.

* **Zero cost and zero setup:** a reviewer does not need any account or API key; `docker compose up --build` starts the whole stack.
* **Runs on CPU:** the 3B model needs about 2 GB of RAM and no GPU.
* **Instruction following:** for its size it reliably follows the two rules that matter most for RAG: answer only from the provided context, and say "I could not find this in the ingested documents" when the context does not contain the answer.
* **Swappable:** the app uses the OpenAI-compatible API, so moving to `gpt-4o-mini` or a larger local model is a `.env` change.
* **Deterministic:** answers use `temperature=0`.

Trade-offs: on a CPU-only laptop (Intel i7-1185G7) an answer takes about 25 seconds, and the small model occasionally omits the `[n]` citation markers. The `sources` list is always returned, so answers stay traceable either way.

Ollama's default context window is smaller than a prompt with 5 chunks and gets silently truncated, so `docker-compose.yml` sets `OLLAMA_CONTEXT_LENGTH=8192`.

### Embedding Model Cost Considerations

Embeddings are computed locally with Sentence Transformers, and the LLM runs locally in Ollama, so the system incurs no API costs.

---

### Chunking Strategy

Chunks are built in three steps (`app/ingest.py` and `app/utils.py`):

1. **Clean extraction, page by page.** pypdf extracts each page separately, so every chunk knows its page number. Lines that repeat at the bottom of 3 or more pages are treated as the running footer (`C U S T O M E R`, `© 2015 SAP SE …`, the chapter name, the page number) and removed, as are table-of-contents lines with leader dots. Soft hyphens that split words across lines (`Opti­ mizer`) are rejoined.
2. **Split into sections.** The guide uses numbered headings such as `3.2.3.1 Data Growth and Data Archiving Monitors in EHS`. A line only counts as a heading if its number follows the previous heading in order (for example 3.1.9 → 3.2), which filters out table rows that happen to start with a number. This finds 72 sections in the guide. Chunks never cross a section boundary, so a chunk does not mix two topics.
3. **Pack whole sentences.** Within a section, sentences, bullet points and table rows are packed into chunks of at most **180 tokens**, and each chunk repeats up to **40 tokens** of whole sentences from the end of the previous chunk in the same section. Every chunk starts with its section title, so both the embedding and the LLM see which section the text comes from.

Each chunk is stored with the metadata `source`, `page`, `page_end`, `section`, `chunk_index` and `chunk_size`.

**Why 180 tokens.** The embedding model, all-MiniLM-L6-v2, reads at most 256 tokens and silently ignores the rest. The previous fixed 1,000-character chunks measured up to 686 tokens, and 20 of 109 were longer than 256, so their endings were never embedded. Sizing chunks in tokens keeps every chunk fully embedded. Three budgets below that limit were compared on the test set's retrieval metrics (no LLM involved):

| Chunking | Chunks | Over 256 tokens | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Fixed 1,000 characters, 200 overlap (previous) | 109 | 20 | 0.65 | 0.80 | 0.72 |
| Section-aware, 120 tokens, 30 overlap | 204 | 0 | 0.70 | 0.90 | 0.77 |
| **Section-aware, 180 tokens, 40 overlap (chosen)** | 137 | 0 | 0.80 | 0.95 | 0.88 |
| Section-aware, 240 tokens, 50 overlap | 108 | 0 | 0.80 | 0.95 | 0.86 |

180 and 240 tokens retrieve equally well here (the 0.02 MRR difference is a single question ranked one place lower). 180 was chosen because 5 chunks of 180 tokens make a shorter LLM prompt, and prompt processing is most of the answer time on CPU. At 120 tokens, sections are cut too finely and two questions miss.

---

### Embedding Model: all-MiniLM-L6-v2

The project uses the **all-MiniLM-L6-v2** embedding model from the Sentence Transformers framework.

#### Model Used

`sentence-transformers/all-MiniLM-L6-v2`

This model is used for generating semantic embeddings of document chunks and user queries.It was selected because it captures semantic meaning beyond exact keyword matching, enabling relevant retrieval even when query and document wording differ.
It is lightweight and efficient, offering fast embedding generation, low memory usage, and quick indexing, making it suitable for local and Docker-based deployments.
Additionally, it is well-supported in the Sentence Transformers ecosystem and integrates smoothly with ChromaDB, making it a reliable choice for RAG systems.

---

### Vector Database Selection

ChromaDB was selected as the vector database for storing and retrieving document embeddings.

Key reasons for this choice include:

* Simple integration with Python applications
* Persistent local storage support
* Efficient vector similarity search
* Lightweight deployment requirements
* Docker-friendly architecture
* No external database server required

The use of ChromaDB also simplifies deployment because all vector data can be stored locally within a Docker volume, ensuring persistence across container restarts.

---

### Handling Tables, Headers, and Complex Formatting

PDF text extraction using PyPDF works well for standard content but has limitations with complex layouts such as tables, multi-column text, headers, footers, and embedded images. These elements may not always preserve their original structure during extraction.

Running headers and footers are detected (lines repeated at the bottom of several pages) and removed before chunking, so they do not add the same noise to every chunk. Tables are extracted as plain text with one table row per line. The chunker keeps each row intact and only splits inside a row if that single row is longer than the chunk budget, but column alignment is lost, so values from different columns run together (for example `EHS Yes Transport organizer in transaction SE01`). Multi-column layouts may also be merged into a linear reading order instead of preserving visual alignment.

Currently, the system focuses on text-based retrieval, so diagrams, images, and charts are not processed or indexed.

#### Citation and Traceability

Every answer cites the chunks it used as `[n]`, and the response includes those chunks in `sources` with their file name, page number, section title, chunk index and similarity distance. This lets users check each answer against the original document.


## Evaluation Results

### Evaluation Methodology

The test set in [`eval/test_set.json`](eval/test_set.json) has **22 questions**: 20 answerable questions spread over all 9 chapters of the guide, each with a reference answer checked against the PDF, plus 2 out-of-scope questions (purchase order approval, payroll) that the document does not cover.

| Metric | How it is computed |
| --- | --- |
| **Hit Rate@5** | A question is a hit when a chunk containing its evidence string (e.g. `SLG1`, `DRAO`) is in the top 5 results. Deterministic, no LLM. Evidence strings are used instead of chunk IDs so the test set stays valid if the chunking changes. |
| **MRR** | Mean reciprocal rank of the first matching chunk (1.0 = always ranked first). |
| **Faithfulness** | LLM judge: fraction of the answer's claims supported by the retrieved chunks. |
| **Answer Relevance** | LLM judge: does the answer directly address the question (answerable questions only). |
| **Answer Correctness** | LLM judge: agreement with the reference answer (answerable questions only). |
| **Abstention Accuracy** | For out-of-scope questions: did the agent say it could not find the answer instead of inventing one. |

The judge is an LLM-as-judge scorer (custom prompt in `app/evaluation.py`) that uses the same `qwen2.5:3b` model. A small model judging its own answers is a known weakness, so the judge scores should be read as indicative; the retrieval metrics are exact.

### Evaluation Summary

Measured with `GET /evaluate` (22 questions, 20 answerable, `top_k = 5`, answer and judge model `qwen2.5:3b`):

| Metric | Score |
| --- | --- |
| Hit Rate@5 | 0.95 |
| MRR | 0.88 |
| Faithfulness | 0.83 |
| Answer Relevance | 0.74 |
| Answer Correctness (LLM judge) | 0.66 |
| Answer Correctness (manual review) | 0.88 |
| Abstention Accuracy (out-of-scope) | 1.00 |

**Manual review:** because the judge is a small model, every answer was also checked by hand against its reference answer (1 = states the fact the question asks for, 0.5 = partially, 0 = wrong or missing). Result: 17 correct, 1 partial, 2 wrong, so 0.88. The generated answers are in [`eval/results_qwen2.5-3b.json`](eval/results_qwen2.5-3b.json) for anyone who wants to re-check them.

**Effect of the chunking change** (same test set, same model):

| Metric | Fixed 1,000-character chunks | Section-aware chunks |
| --- | --- | --- |
| Hit Rate@5 | 0.80 | **0.95** |
| MRR | 0.72 | **0.88** |
| Faithfulness | 0.81 | 0.83 |
| Answer Relevance | 0.75 | 0.74 |
| Answer Correctness (LLM judge) | 0.66 | 0.66 |

Per question:

| # | Question | Hit rank | Faithfulness | Relevance | Correctness (judge) | Correct (manual) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Which transaction is used for logging and tracing ABAP components? | 1 | 1.00 | 0.50 | 0.50 | ✓ |
| 2 | Which transaction code can be used to access the SAP CCMS monitor sets? | 2 | 0.50 | 0.75 | 0.50 | ✓ |
| 3 | Which SAP Note describes how to enable the auto-alert mechanism of CCMS? | 1 | 1.00 | 0.50 | 0.00 | ✓ |
| 4 | How do you monitor report shipping in EHS? | 1 | 0.50 | 0.80 | 0.60 | ✓ |
| 5 | Where can you monitor EHS report generation and report shipping? | 1 | 0.50 | 0.80 | 0.80 | ✓ |
| 6 | Which transaction is used to create users in EHP8 for SAP ERP 6.0? | 1 | 1.00 | 0.50 | 0.50 | ✓ |
| 7 | How can you display all active Business Functions in the system? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 8 | What was SAP ERP 6.0 formerly called? | **miss** | 1.00 | 0.50 | 0.00 | ✗ |
| 9 | Where should you create a CSS ticket if errors occur in PLM Web UI applications? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 10 | Which SAP Note has information about troubleshooting EH&S Expert? | 1 | 0.00 | 0.00 | 0.00 | ✗ |
| 11 | How can you physically delete EHS specification data that is marked for deletion? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 12 | Which database table grows fast if EHS reports are stored without an external DMS? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 13 | Which FS-CM archiving object is the fastest growing? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 14 | Which transport management tool is used for EHS? | 2 | 0.50 | 0.50 | 0.50 | ✓ |
| 15 | Which support role is shipped for remote support of ABAP-based components? | 1 | 1.00 | 0.50 | 1.00 | ✓ |
| 16 | In a distributed EHS installation, how is data distributed from the master system to the logistics systems? | 1 | 1.00 | 1.00 | 0.50 | ✓ |
| 17 | What does an application log consist of? | 1 | 0.50 | 0.75 | 0.50 | ✓ |
| 18 | Which tools are required to install an SAP enhancement package? | 2 | 1.00 | 1.00 | 1.00 | ✓ |
| 19 | Who are the target groups of the SAP ERP Application Operations Guide? | 1 | 0.75 | 0.80 | 0.80 | ½ |
| 20 | What is the IS-OIL log object and subobject for nomination? | 1 | 1.00 | 1.00 | 1.00 | ✓ |
| 21 | What is the purchase order approval workflow? | out of scope | 1.00 | n/a | n/a | ✓ abstained |
| 22 | How do I run payroll in SAP HCM? | out of scope | 1.00 | n/a | n/a | ✓ abstained |

Both out-of-scope questions were answered with "I could not find this in the ingested documents." (abstention accuracy 1.00); relevance and correctness are not scored for them.

**Reading the numbers:** section-aware chunking fixed 3 of the 4 retrieval misses of the previous chunking (questions 11, 14 and 19) and moved questions 3 and 10 up to rank 1; only question 18 dropped, from rank 1 to 2. The judge's correctness score stayed at 0.66 even though more answers became correct, because the judge scores several correct one-line answers at 0.5 or 0 (for example `SLG1` for question 1 and the quoted SAP Note `617547` for question 3). The manual review (0.88) is the more reliable measure of answer quality; see failure case 4.

### Documented Failure Cases

The full evaluation output, including every generated answer and the judge's reasoning, is saved in [`eval/results_qwen2.5-3b.json`](eval/results_qwen2.5-3b.json). The ranks below were measured by ranking all 137 chunks for each question.

**Fixed by section-aware chunking.** The previous fixed-size chunking had four retrieval misses, and three of them are now retrieved. The target-group list (question 19) had been buried in a chunk that mixed three topics. The `RC1PHDEL` sentence (question 11) and the `SE01` table (question 14) had ranked 8th and 10th. Question 5, where the model had picked the wrong transaction code (`SLG1` instead of `CG5Z`) out of a dense bullet list, is now answered correctly.

#### Failure Case 1: A paraphrased question does not retrieve the answer (question 8)

**Query:** "What was SAP ERP 6.0 formerly called?"

**Observed:** "I could not find this in the ingested documents." The agent abstained instead of guessing, which is the right behavior when retrieval fails.

**Root cause:** the answer is in section 1.1 ("This guide exclusively refers to the name SAP ERP 6.0, rather than the formerly used name mySAP ERP 2005"), but those chunks rank only **18th** and **21st** of 137. The top 5 are *6.4 Release and Upgrade Management*, *4 Management of SAP enhancement package 8*, the front matter, *1.3 Important SAP Notes* and *2 Technical System Landscape*. The most distinctive words in the question, "SAP ERP 6.0", appear all over those release- and version-heavy sections, so the small embedding model matches that topic rather than the meaning "former name".

**Fix:** hybrid retrieval with BM25, so the rare word "formerly" can pull the right chunk up, or query rewriting that expands the question (for example "former name", "renamed").

#### Failure Case 2: Right chunk retrieved, but the answer is only a citation number (question 10)

**Query:** "Which SAP Note has information about troubleshooting EH&S Expert?"

**Observed:** the complete answer was `1`. The judge scored it 0 on all three metrics.

**Root cause:** retrieval was correct (rank 1). The chunk from section 3.1.10.1 says "For more information on EH&S Expert logs and traces and how to enable them, see 1364100." The note number appears as a bare number without the words "SAP Note", and the prompt asks the model to cite passages as `[1]`. The 3B model answered with the passage number instead of the note number.

**Fix:** a larger answer model, and a prompt that asks for the answer first and the citation after it, or a check that rejects an answer consisting only of a citation.

#### Failure Case 3: Flattened tables and lists shape the answer (questions 14 and 19)

**Queries:** "Which transport management tool is used for EHS?" and "Who are the target groups of the SAP ERP Application Operations Guide?"

**Observed:** question 14 was answered with the raw table row "EHS Yes Transport organizer in transaction SE01". Question 19 listed 4 of the 5 target groups and left out "Support Specialist".

**Root cause:** the chunk for question 14 is *Table 35: Component Change Management Tools*, which pypdf extracts as rows without their column headers, so the model copied the row instead of answering in a sentence. The fact (`SE01`) is right, but the answer is hard to read. For question 19 the five groups are a bullet list, and the small model dropped the last item.

**Fix:** layout-aware table extraction that writes each row with its column names (for example "Component: EHS | Transport Management Tool: Transport organizer in transaction SE01"), and a larger answer model for complete lists.

#### Failure Case 4: The judge underrates correct answers

**Observed:** the manual review marks 17 of 20 answers correct (0.88), but the judge's correctness average is 0.66. Question 1 was answered `SLG1`, which is correct, and got 0.5 because it "only mentions SLG1". Question 3 quoted "see 617547", which is correct, and got 0. Question 6 answered `SU01`, which is correct, and got 0.5 for not mentioning PFCG, which the question did not ask about.

**Root cause:** the judge is the same 3B model that writes the answers. It penalizes short answers and answers that do not repeat every detail of the reference answer, and its reasons sometimes contradict the answer it is grading. The retrieval metrics (hit rate, MRR) are computed without an LLM and are not affected.

**Fix:** a stronger, independent judge model (for example `gpt-4o-mini`) or RAGAS, and continued manual spot checks of the judge's scores, as done here.

#### Improvements with more time

1. Hybrid BM25 plus vector retrieval, or query rewriting (failure case 1).
2. A larger answer model, such as `qwen2.5:7b` or `gpt-4o-mini`, with an "answer first, then cite" prompt (failure cases 2 and 3).
3. Table extraction that keeps the column headers on every row (failure case 3).
4. An independent judge model (failure case 4).

---

### Limitations

* The LLM judge is the same small model that writes the answers, so faithfulness and relevance scores are indicative rather than exact.
* On a CPU-only machine each answer takes about 20 to 25 seconds, and the full `/evaluate` takes about 17 minutes (`/evaluate?retrieval_only=true` takes seconds).
* Section detection relies on numbered headings (1, 1.1, 1.1.1). A document without numbered headings is chunked by sentences as a single section.
* No reranking stage after retrieval.
* No OCR support for scanned PDFs.
* No image or diagram understanding.
* Limited handling of complex table structures.

---

### Future Improvements

Given additional development time, the following enhancements would likely improve retrieval quality and overall system performance:

1. Hybrid retrieval combining BM25 keyword search with vector search, which helps with exact codes like transaction names and SAP Note numbers.
2. A cross-encoder reranking model to improve retrieval precision.
3. A stronger, independent judge model (for example `gpt-4o-mini`) or the RAGAS library for evaluation.
4. Streaming answers token by token via Server-Sent Events.
5. Better table extraction using layout-aware parsing (pdfplumber or PyMuPDF).
6. OCR support for scanned and image-based PDFs.
7. Conversation memory for multi-turn interactions.

## Video link 

https://drive.google.com/file/d/1uve9GYYJ3QBdItojJtK04ywL_8j-jFnH/view?usp=drivesdk
