# Enterprise RAG Assistant

A locally-hosted, retrieval-augmented generation system for querying internal
company documents, tickets, and policies. Combines hybrid (dense + keyword)
retrieval, cross-encoder reranking, a fine-tuned local LLM, and full
observability into a single FastAPI service, fronted by a streaming Next.js
chat UI.

> Built as a portfolio-grade demonstration of a production-shaped RAG stack:
> real ingestion pipelines, real evaluation, real guardrails, and real
> containerization - not a notebook demo.

## Architecture

```
                     ┌────────────────────┐
                     │   Next.js (React)  │  streaming chat UI
                     │   :3000            │  (useChat + TextStreamChatTransport)
                     └─────────┬──────────┘
                               │ POST /query  (CORS)
                     ┌─────────▼──────────┐
                     │   FastAPI  :8000   │
                     │  guardrails → retrieval → generation → stream
                     └───┬─────────┬──────┘
              ┌──────────┘         └───────────┐
   ┌──────────▼──────────┐          ┌──────────▼──────────┐
   │  Hybrid Retriever    │          │  Qwen2.5-0.5B-Instruct│
   │  (LangChain)          │          │  + QLoRA adapter      │
   │  ┌─────────────────┐ │          │  (PEFT/TRL, local GPU) │
   │  │ Pinecone (dense) │ │          └────────────────────────┘
   │  │ BAAI/bge-small   │ │
   │  │ 384-dim, cosine  │ │
   │  └────────┬────────┘ │
   │  ┌────────▼────────┐ │
   │  │ BM25 (keyword)   │ │          ┌────────────────────────┐
   │  │ local pickle     │ │          │   Arize Phoenix :6006   │
   │  └────────┬────────┘ │          │  OTel traces: retrieval,│
   │           │ weighted RRF fusion  │  rerank, LLM generation │
   │  ┌────────▼────────┐ │          └────────────────────────┘
   │  │ BGE cross-encoder│ │
   │  │ reranker (20→3)  │ │          ┌────────────────────────┐
   │  └─────────────────┘ │          │   MSSQL (Docker)        │
   └───────────────────────┘         │  document metadata,     │
                                      │  idempotent ingestion   │
                                      │  state                  │
                                      └────────────────────────┘
```

**Backend** (`backend/`) - FastAPI + Uvicorn. `POST /query` runs guardrails,
then hybrid retrieval + reranking, then streams a generated, cited answer.
`GET /health` reports live index/dependency status (document count, vector
count, hybrid-search availability, reranker model) rather than hardcoded
values. `POST /feedback` logs thumbs-up/down ratings for future RLHF use.

**Frontend** (`frontend-next/`) - Next.js 16 (App Router) + the Vercel AI SDK
(`ai`, `@ai-sdk/react`). Streams tokens live via `useChat` +
`TextStreamChatTransport`, renders answers as GitHub-flavored Markdown with
syntax-highlighted code blocks, and shows a collapsible "View Retrieved
Context" drawer with the exact chunks and rerank scores behind each answer.

**Retrieval** (`backend/retriever.py`, `ingestion/`) - Pinecone holds
384-dimensional `BAAI/bge-small-en-v1.5` embeddings (cosine similarity) of
document chunks. A local BM25 index (`rank_bm25`, pickled by
`ingestion/build_bm25.py`) covers exact-match/keyword queries dense
embeddings tend to miss. Both are fused via LangChain's `EnsembleRetriever`
(weighted Reciprocal Rank Fusion) and reranked down to the top 3 by a
`BAAI/bge-reranker-base` cross-encoder.

**Generation** (`backend/model.py`, `finetuning/`) - `Qwen2.5-0.5B-Instruct`
fine-tuned with QLoRA (4-bit NF4, PEFT + TRL `SFTTrainer`) on enterprise-style
Q&A pairs, served locally via `transformers`, streamed token-by-token with
`TextIteratorStreamer`.

**Metadata** (`infrastructure/`, `backend/database.py`) - MSSQL (Dockerized)
stores document metadata and idempotent per-document ingestion state (content
hashing, so re-running ingestion only touches changed documents).

**Observability** (`backend/api.py`) - Arize Phoenix + OpenTelemetry.
LangChain's own retrieval/rerank steps are auto-instrumented; the raw
`transformers` generation step (which LangChain can't see, since it isn't a
LangChain `Runnable`) gets an explicit manual span with prompt/completion
token counts and latency.

## Key Features

- **Real-time token streaming** - `backend/model.py` streams generation via a
  background-threaded `TextIteratorStreamer`; `backend/api.py` exposes it as
  a `StreamingResponse`, consumed live by the Next.js frontend's `useChat`
  hook with zero buffering/spinner-then-dump behavior.
- **Hybrid search (dense + keyword, fused by RRF)** - dense-only retrieval
  alone measured **8.5% Context Recall@3** against a held-out test split, a
  real ceiling on 384-dimensional semantic similarity for internal jargon and
  ticket IDs. Fusing a local BM25 index in via weighted Reciprocal Rank
  Fusion lifted that to **12.8%** (~50% relative improvement) with no
  measurable latency cost - see [Engineering Challenges](#engineering-challenges).
- **Semantic guardrails** - a regex-based prompt-injection filter and an
  embedding-based out-of-domain router (cosine similarity against a small set
  of in-domain exemplar questions, calibrated against real measured
  in-domain/out-of-domain scores, not a guessed threshold) run *before* the
  retriever or the LLM are ever invoked, rejecting bad requests cheaply.
- **Arize Phoenix telemetry** - every `/query` call is fully traced: base
  retrieval (20 candidates), reranking (top 3 with scores), and generation
  (latency, prompt/completion token counts), inspectable live at
  `http://localhost:6006`.
- **Idempotent, checkpointed ingestion at scale** - `ingestion/embed_pinecone.py`
  content-hashes every document, skips unchanged ones, batches upserts across
  documents (not per-document), and checkpoints a contiguous confirmed
  prefix so a killed/resumed job never re-processes or silently skips work.
  Verified at 50,000-document / ~300k-vector scale in this project's own
  development history.

## Local Setup Instructions

This repo ships a `docker-compose.prod.yml` for the API, the Next.js
frontend, and Phoenix. MSSQL is provisioned separately via
`infrastructure/docker-compose.yml` (kept independent so its data volume
survives an `api`/`web` rebuild).

### 1. Prerequisites

- Docker Desktop with the NVIDIA Container Toolkit (the `api` service
  requests a GPU; generation and embeddings run on CPU-only hosts too, just
  slower - drop the `deploy.resources.reservations.devices` block in
  `docker-compose.prod.yml` if you have no GPU)
- A Pinecone API key ([pinecone.io](https://www.pinecone.io))

### 2. Configure environment

```bash
cp .env.example .env
# Fill in PINECONE_API_KEY and MSSQL_SA_PASSWORD at minimum.
```

### 3. Start MSSQL and apply the schema

```bash
docker compose -f infrastructure/docker-compose.yml up -d
python -m infrastructure.init_db
```

### 4. Ingest data (one-time, or whenever source documents change)

Requires a local Python 3.11+ environment with `requirements.txt` installed
(these scripts run on the host, not in a container, so they can stream
directly against the local MSSQL/Pinecone connections):

```bash
python -m ingestion.seed_documents          # load documents into MSSQL
python -m ingestion.embed_pinecone          # chunk, embed, upsert to Pinecone
python -m ingestion.build_bm25              # build the local BM25 index
```

Each script accepts a `--limit N` flag for a fast local smoke test before
committing to a full ingestion run.

### 5. Provide a fine-tuned adapter (optional)

The API runs fine without one (falls back to an explicit
"insufficient context" answer for generation, retrieval still works). To
train your own:

```bash
python prepare_data.py                      # builds data/finetune/{train,validation,test}.jsonl
python -m finetuning.train_qlora            # saves the adapter to outputs/qlora
```

### 6. Build and run the stack

```bash
docker compose -f docker-compose.prod.yml up --build
```

| Service          | URL                          |
|-------------------|-------------------------------|
| Chat UI (Next.js) | http://localhost:3000         |
| API docs (FastAPI)| http://localhost:8000/docs    |
| Telemetry (Phoenix)| http://localhost:6006        |

### Running without Docker (local dev)

```bash
# Terminal 1
uvicorn backend.api:app --reload --port 8000
# Terminal 2
cd frontend-next && npm install && npm run dev
```

## Engineering Challenges

### Dense retrieval hit a real recall ceiling - fixed with BM25 + Reciprocal Rank Fusion

An end-to-end evaluation against a held-out test split (`evaluate_model.py`,
real Pinecone retrieval, no simulated/perfect-recall shortcuts) measured
**Context Recall@3 = 8.5%** with dense-only retrieval. Diagnosis: most of
that gap was corpus coverage (only a fraction of the full document set was
ingested at the time), but a real, independently-provable *retrieval
quality* gap existed underneath it too - dense embedding similarity is weak
on exact identifiers. A controlled before/after test made this concrete: a
query referencing a real ticket ID (`"What is the status of build-job
#4572?"`) did not return the target document anywhere in dense-only
retrieval's top 20 candidates at all.

The fix: `ingestion/build_bm25.py` builds a local BM25 (`rank_bm25`) index
over the same chunk boundaries as the Pinecone index (chunking logic shared
between both scripts, so `chunk_id`s line up exactly). `backend/retriever.py`
fuses the two candidate lists with LangChain's `EnsembleRetriever`, which
implements **weighted Reciprocal Rank Fusion** - each retriever's *rank*
for a document (not its raw, differently-scaled score) contributes
`weight / (rank + c)` to a combined score, so a keyword-exact match ranked
#1 by BM25 and a semantically-similar document ranked #1 by dense search are
combined on equal footing before the cross-encoder reranks the fused top 20
down to 3. Result: the ticket-ID query above now returns the correct
document at rank 1 (rerank score 0.524 vs. 0.0005 for the next candidate),
and the measured end-to-end **Context Recall@3 rose from 8.5% to 12.8%**
with no measurable latency cost (the reranker/generation step still
dominates total latency, not the added BM25 lookup).

### Migrating streaming to Next.js meant reading the SDK's actual source, not its README

The original migration plan called for the legacy Vercel "Data Stream
Protocol" (`0:"<token>"\n` line framing over `text/event-stream`). The
installed `ai`/`@ai-sdk/react` major versions (v7/v4) have since replaced
that with a `ChatTransport` abstraction - confirmed by reading the SDK's own
compiled type definitions and source directly rather than assuming the
documented protocol still matched what was installed. Implementing the old
framing literally would have been actively incompatible with this SDK
version.

The actual fix: `TextStreamChatTransport`, which decodes a plain-text
response body directly - meaning the FastAPI backend's existing
`StreamingResponse` (`media_type="text/plain"`) needed **no wire-format
change at all**. Two real problems remained, solved without touching the
streaming format:

- **Request shape** - `useChat` sends `{messages: UIMessage[], id, trigger}`
  by default; the backend expects `{"question": "..."}"`. Bridged with
  `prepareSendMessagesRequest`, a transport hook that reshapes the outgoing
  request body before it's sent.
- **Citations delivery** - the plan called for reading citations from the
  `X-Citations` response header. Verified this is architecturally
  unreachable: `HttpChatTransport.sendMessages()` passes only
  `response.body` into stream processing and never surfaces
  `response.headers` to the calling component. Solved by appending the same
  base64 citation payload *in-band*, as a sentinel
  (`\n\n<!--CITATIONS:...-->`) at the end of the token stream itself, parsed
  and stripped from the displayed answer client-side - the header is still
  sent too, for any other consumer, but the frontend relies on the in-band
  copy, which is actually reachable through the SDK's transport
  abstraction.

## Guardrails, Evaluation, and Quality Gates

- Prompt-injection filtering and an embedding-based out-of-domain router run
  before retrieval or generation.
- `evaluate_model.py` measures Context Recall@k and a lexical-overlap
  faithfulness heuristic against a held-out test split with zero overlap
  against training documents.
- No mocked database or vector-store connections anywhere in the codebase -
  every script and test runs against a real Dockerized MSSQL instance and
  the real Pinecone API.
- `pytest` covers data splitting, chunk metadata, idempotent ingestion, API
  validation, no-context behavior, citation construction, and guardrail
  behavior.

## Repository Layout

See [`docs/repo_structure.txt`](docs/repo_structure.txt) for the full tracked
file tree.

```
backend/        FastAPI app, retriever, generator, guardrails, DB access
frontend-next/  Next.js chat UI (Vercel AI SDK)
ingestion/      MSSQL extraction, Pinecone embedding, BM25 index build
finetuning/     QLoRA training (PEFT/TRL)
infrastructure/ MSSQL Docker Compose + schema
data/           Deterministic train/validation/test splits
tests/          pytest suite
docker-compose.prod.yml   API + web + Phoenix stack
```
