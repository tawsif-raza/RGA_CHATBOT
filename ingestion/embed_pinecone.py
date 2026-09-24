"""Idempotently chunk and upsert MSSQL documents into Pinecone, in parallel batches.

Sized for the full ~512k-document EnterpriseRAG-Bench corpus: vectors are
embedded per-document (batched, not one call per chunk), buffered across
documents into exactly `UPSERT_BATCH_SIZE`-vector batches, and upserted
asynchronously across a Pinecone client thread pool. Progress is checkpointed
to `checkpoint.json` so a crash (e.g. a network timeout mid-run) can resume
from the last confirmed document instead of restarting from scratch.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sqlalchemy import Engine, text
from tqdm import tqdm

from backend.config import Settings
from backend.database import create_sql_engine
from ingestion.chunking import chunk_text
from ingestion.extract_sql import SourceDocument, iter_documents, list_document_ids

# Pinecone enforces a 2 MB payload limit per upsert request. 200 vectors of
# BGE-small (384-dim) embeddings plus chunk-text metadata stays comfortably
# under that ceiling even for large chunks.
UPSERT_BATCH_SIZE = 200

# Size of the Pinecone client's thread pool, used to send multiple upsert
# batches concurrently via `async_req=True`.
POOL_THREADS = 30

CHECKPOINT_PATH = Path("checkpoint.json")


def content_hash(content: str) -> str:
    """Return the stable hash used to skip unchanged source documents."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def document_is_current(engine: Engine, document_id: str, digest: str) -> bool:
    """Return whether MSSQL records an identical successful ingestion."""
    with engine.connect() as connection:
        value = connection.execute(text("SELECT content_hash FROM dbo.ingestion_state WHERE document_id = :document_id"), {"document_id": document_id}).scalar_one_or_none()
    return value == digest


def mark_ingested(engine: Engine, document_id: str, digest: str) -> None:
    """Upsert ingestion state only after Pinecone accepts all vectors."""
    statement = text("""
        MERGE dbo.ingestion_state AS target
        USING (SELECT :document_id AS document_id, :content_hash AS content_hash) AS source
        ON target.document_id = source.document_id
        WHEN MATCHED THEN UPDATE SET content_hash = source.content_hash, ingested_at = SYSUTCDATETIME()
        WHEN NOT MATCHED THEN INSERT (document_id, content_hash) VALUES (source.document_id, source.content_hash);
    """)
    with engine.begin() as connection:
        connection.execute(statement, {"document_id": document_id, "content_hash": digest})


def build_vectors(document: SourceDocument, digest: str, embeddings: HuggingFaceEmbeddings) -> list[dict[str, Any]]:
    """Chunk one document with LangChain and embed its chunks into Pinecone-ready vectors.

    Embeds all chunks in a single `embed_documents` call rather than one
    `embed_query` call per chunk: at this corpus's scale (hundreds of
    thousands of documents), per-chunk calls would lose the batching
    throughput the underlying sentence-transformers model relies on.
    """
    chunks = chunk_text(document.content)
    if not chunks:
        return []
    timestamp = datetime.now(UTC).isoformat()
    chunk_embeddings = embeddings.embed_documents(chunks)
    vectors: list[dict[str, Any]] = []
    for position, (chunk, vector) in enumerate(zip(chunks, chunk_embeddings)):
        chunk_id = f"{document.document_id}:{position}"
        vectors.append({
            "id": chunk_id,
            "values": vector,
            "metadata": {
                "document_id": document.document_id,
                "chunk_id": chunk_id,
                "source_name": document.source_name,
                "department": document.department or "",
                "role": document.role or "",
                "content_hash": digest,
                "ingested_at": timestamp,
                "text": chunk,
            },
        })
    return vectors


def batched(vectors: list[dict[str, Any]], batch_size: int = UPSERT_BATCH_SIZE) -> Iterator[list[dict[str, Any]]]:
    """Yield successive vector batches no larger than `batch_size`."""
    for start in range(0, len(vectors), batch_size):
        yield vectors[start : start + batch_size]


def verify_index_dimension(index_dimension: int, embedding_dimension: int) -> None:
    """Fail fast if the Pinecone index was not created for this embedding model."""
    if index_dimension != embedding_dimension:
        raise RuntimeError(f"Pinecone index dimension {index_dimension} does not match embedding model output dimension {embedding_dimension}. Recreate the index with the correct dimension before ingesting.")


def ensure_index(client: Pinecone, settings: Settings, embedding_dimension: int) -> None:
    """Create the configured serverless index on first run, or verify its dimension if it already exists."""
    existing = {index["name"] for index in client.list_indexes()}
    if settings.pinecone_index_name not in existing:
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        return
    verify_index_dimension(client.describe_index(settings.pinecone_index_name).dimension, embedding_dimension)


# --- Checkpointing -----------------------------------------------------------
#
# The checkpoint stores the last document_id that is *fully confirmed*
# upserted, in the corpus's deterministic `ORDER BY document_id` sequence.
# Because batches are sent concurrently (async_req=True across a thread
# pool), a later document's batch can complete before an earlier document's,
# so the checkpoint only ever advances through a *contiguous* confirmed
# prefix (see `advance_watermark`) rather than jumping to whatever finished
# most recently. That keeps resume-after-crash correct.


def load_checkpoint(path: Path) -> str | None:
    """Return the last confirmed document_id, or None if no checkpoint exists."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("last_document_id")


def save_checkpoint(path: Path, document_id: str) -> None:
    """Atomically write the checkpoint so a crash mid-write can't corrupt it."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps({"last_document_id": document_id}), encoding="utf-8")
    os.replace(tmp_path, path)


def resume_start_index(doc_order: list[str], checkpoint: str | None) -> int:
    """Return the index to resume from: just past the checkpointed document_id."""
    if checkpoint is None:
        return 0
    position = bisect.bisect_right(doc_order, checkpoint)
    return position


def advance_watermark(doc_order: list[str], watermark_index: int, completed: set[str]) -> int:
    """Advance the watermark through a contiguous run of completed documents."""
    while watermark_index < len(doc_order) and doc_order[watermark_index] in completed:
        watermark_index += 1
    return watermark_index


@dataclass
class IngestionStats:
    """Summary counters returned by `run_ingestion`."""

    documents_processed: int = 0
    documents_skipped_unchanged: int = 0
    vectors_upserted: int = 0
    elapsed_seconds: float = 0.0

    @property
    def vectors_per_second(self) -> float:
        return self.vectors_upserted / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def run_ingestion(
    documents: Iterable[SourceDocument],
    doc_order: list[str],
    start_index: int,
    engine: Engine,
    index: Any,
    embeddings: HuggingFaceEmbeddings,
    settings: Settings,
    checkpoint_path: Path = CHECKPOINT_PATH,
    pool_threads: int = POOL_THREADS,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> IngestionStats:
    """Chunk, embed, and upsert `documents` in cross-document batches of exactly `batch_size`.

    `documents` must already be filtered/skipped to start at `start_index`
    within `doc_order` (the full corpus's deterministic id sequence).
    """
    stats = IngestionStats()
    started = time.perf_counter()

    buffer: list[dict[str, Any]] = []
    buffer_doc_ids: list[str] = []
    pending: list[tuple[Any, list[str]]] = []
    remaining_chunks: dict[str, int] = {}
    doc_digest: dict[str, str] = {}
    completed: set[str] = set()
    watermark_index = start_index

    def checkpoint_if_advanced(new_watermark: int) -> int:
        nonlocal watermark_index
        if new_watermark > watermark_index:
            watermark_index = new_watermark
            save_checkpoint(checkpoint_path, doc_order[watermark_index - 1])
        return watermark_index

    def drain_one() -> None:
        """Block on the oldest in-flight batch and account for its completion."""
        future, owners = pending.pop(0)
        future.get()  # raises if the upsert failed; checkpoint stays put for resume
        stats.vectors_upserted += len(owners)
        for document_id in owners:
            remaining_chunks[document_id] -= 1
            if remaining_chunks[document_id] == 0:
                mark_ingested(engine, document_id, doc_digest[document_id])
                completed.add(document_id)
                del remaining_chunks[document_id]
                del doc_digest[document_id]
        checkpoint_if_advanced(advance_watermark(doc_order, watermark_index, completed))

    pbar = tqdm(total=len(doc_order) - start_index, desc="Ingesting documents", unit="doc")
    last_pbar_watermark = start_index

    def sync_pbar() -> None:
        nonlocal last_pbar_watermark
        advanced = watermark_index - last_pbar_watermark
        if advanced > 0:
            elapsed = time.perf_counter() - started
            pbar.set_postfix({"vectors/s": f"{stats.vectors_upserted / elapsed:.1f}" if elapsed > 0 else "0.0"})
            pbar.update(advanced)
            last_pbar_watermark = watermark_index

    try:
        for document in documents:
            digest = content_hash(document.content)
            if document_is_current(engine, document.document_id, digest):
                completed.add(document.document_id)
                stats.documents_skipped_unchanged += 1
                checkpoint_if_advanced(advance_watermark(doc_order, watermark_index, completed))
                sync_pbar()
                continue

            vectors = build_vectors(document, digest, embeddings)
            stats.documents_processed += 1
            if not vectors:
                completed.add(document.document_id)
                checkpoint_if_advanced(advance_watermark(doc_order, watermark_index, completed))
                sync_pbar()
                continue

            doc_digest[document.document_id] = digest
            remaining_chunks[document.document_id] = len(vectors)
            buffer.extend(vectors)
            buffer_doc_ids.extend([document.document_id] * len(vectors))

            while len(buffer) >= batch_size:
                batch = buffer[:batch_size]
                owners = buffer_doc_ids[:batch_size]
                del buffer[:batch_size]
                del buffer_doc_ids[:batch_size]
                future = index.upsert(vectors=batch, namespace=settings.pinecone_namespace, async_req=True)
                pending.append((future, owners))
                if len(pending) >= pool_threads:
                    drain_one()
                    sync_pbar()

        if buffer:
            future = index.upsert(vectors=buffer, namespace=settings.pinecone_namespace, async_req=True)
            pending.append((future, buffer_doc_ids))

        while pending:
            drain_one()
            sync_pbar()
    finally:
        pbar.close()

    stats.elapsed_seconds = time.perf_counter() - started
    return stats


def main(limit: int | None) -> None:
    """Ingest changed enabled MSSQL documents into the configured index, optionally capped by `limit`, resuming from checkpoint.json if present."""
    settings = Settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY must be configured.")
    engine = create_sql_engine(settings)
    client = Pinecone(api_key=settings.pinecone_api_key, pool_threads=POOL_THREADS)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model, encode_kwargs={"normalize_embeddings": True})
    embedding_dimension = len(embeddings.embed_query("dimension check"))
    ensure_index(client, settings, embedding_dimension)
    index = client.Index(settings.pinecone_index_name, pool_threads=POOL_THREADS)

    doc_order = list_document_ids(engine, limit=limit)

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    start_index = resume_start_index(doc_order, checkpoint)
    if start_index >= len(doc_order):
        print(f"Nothing to do: checkpoint {checkpoint!r} already covers all {len(doc_order)} targeted documents.")
        return
    if checkpoint:
        print(f"Resuming from checkpoint after document_id={checkpoint!r} ({start_index}/{len(doc_order)} already confirmed).")

    # Push both the resume-skip and the row-count bound into the SQL query
    # itself (`document_id > :start_after` and `TOP (n)`) rather than
    # streaming the full corpus in Python and discarding rows: at ~512k
    # documents that would waste most of the run's time and bandwidth on
    # rows we already know to skip.
    start_after = doc_order[start_index - 1] if start_index > 0 else None
    remaining_target_count = len(doc_order) - start_index
    remaining_documents = iter_documents(engine, start_after=start_after, limit=remaining_target_count)

    stats = run_ingestion(remaining_documents, doc_order, start_index, engine, index, embeddings, settings)
    print(f"Processed {stats.documents_processed} changed documents ({stats.documents_skipped_unchanged} already current), upserted {stats.vectors_upserted} vectors in {stats.elapsed_seconds:.1f}s ({stats.vectors_per_second:.1f} vectors/sec).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Optional cap on the number of documents to ingest.")
    main(parser.parse_args().limit)
