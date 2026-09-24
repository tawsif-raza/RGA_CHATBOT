# Enterprise RAG v1 Implementation Plan

## Objective

Build a local enterprise RAG system for English documents. Streamlit calls a FastAPI service, which retrieves BGE-small embeddings from Pinecone, enriches them with MSSQL metadata, and generates a cited answer through a locally loaded Qwen QLoRA adapter.

## Runtime and Dependencies

- Target Python 3.11+ on Linux with NVIDIA CUDA; Windows is development-only.
- Use one root `requirements.txt`; install Microsoft ODBC Driver 18 separately on API and ingestion hosts.
- Put all credentials in environment variables. `.env.example` documents the values; no secret is committed.
- Use `BAAI/bge-small-en-v1.5`, normalized vectors, cosine similarity, and a 384-dimensional Pinecone index.

## Data and Model Workflow

1. `prepare_data.py` downloads EnterpriseRAG-Bench, creates one Qwen chat example per valid primary document, and deterministically groups source documents into 80/10/10 train/validation/test splits.
2. `finetuning/train_qlora.py` trains Qwen2.5-0.5B-Instruct with NF4 QLoRA, validates on the held-out validation split, and saves adapters under `outputs/qlora`.
3. `finetuning/export_model.py` optionally merges the adapter only for a standalone deployment artifact.
4. `evaluate_model.py` runs only against the held-out test split using real Pinecone retrieval. OpenAI judging is optional; local metrics always run.

## Retrieval and Serving Workflow

1. `infrastructure/init.sql` defines document metadata and idempotent ingestion state.
2. `ingestion/extract_sql.py` reads enabled source records from MSSQL.
3. `ingestion/embed_pinecone.py` chunks source content, hashes it, skips unchanged content, and upserts vectors plus required metadata.
4. FastAPI loads the model once at startup. `POST /query` retrieves up to four chunks, applies optional department/role filters, constructs one token-budgeted Qwen chat prompt, and returns the answer and citations.
5. `GET /health` reports API, model, MSSQL, and Pinecone readiness without revealing credentials. The Streamlit UI shows responses and sources.

## Quality Gates

- Do not train on or evaluate against overlapping source document IDs.
- Train, evaluate, and serve with the same tokenizer chat-template contract.
- Return an explicit insufficient-context answer if retrieval returns no usable context.
- Test splitting, chunk metadata, idempotent ingestion, API validation, no-context behavior, citations, and prompt construction with `pytest`.
- Update `STATUS.md` after each completed milestone with a timestamp, command, and concise result.
