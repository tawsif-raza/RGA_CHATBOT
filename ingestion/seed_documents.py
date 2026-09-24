"""Seed dbo.documents from the EnterpriseRAG-Bench documents split.

Uses the same Hugging Face split that `prepare_data.py` reads for training
context, so the retrieval corpus stays consistent with what the model was
fine-tuned on.
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from sqlalchemy import Engine, bindparam, text
from sqlalchemy.dialects.mssql import NVARCHAR

from backend.config import Settings
from backend.database import create_sql_engine

DATASET_NAME = "onyx-dot-app/EnterpriseRAG-Bench"

# Explicit NVARCHAR sizes are required: pyodbc otherwise infers each
# parameter's SQL size from the first bound value's length and can truncate
# a later, longer row bound to the same prepared statement in this loop.
UPSERT_SQL = text("""
    MERGE dbo.documents AS target
    USING (SELECT :document_id AS document_id, :source_name AS source_name, :content AS content) AS source
    ON target.document_id = source.document_id
    WHEN MATCHED THEN UPDATE SET source_name = source.source_name, content = source.content, updated_at = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN INSERT (document_id, source_name, content, is_enabled)
        VALUES (source.document_id, source.source_name, source.content, 1);
""").bindparams(
    bindparam("document_id", type_=NVARCHAR(128)),
    bindparam("source_name", type_=NVARCHAR(512)),
    bindparam("content", type_=NVARCHAR(None)),
)


def truncate_utf16(value: str, max_units: int) -> str:
    """Truncate to at most `max_units` UTF-16 code units, matching NVARCHAR(n) sizing.

    A Python `str` slice counts Unicode code points, but SQL Server measures
    NVARCHAR length in UTF-16 code units; a single astral character (e.g. an
    emoji) is 1 code point yet 2 UTF-16 units, so a code-point slice can still
    overflow the column by one unit.
    """
    encoded = value.encode("utf-16-le")
    if len(encoded) <= max_units * 2:
        return value
    return encoded[: max_units * 2].decode("utf-16-le", errors="ignore")


def to_document_row(record: dict[str, object]) -> dict[str, str] | None:
    """Return the upsert parameters for one dataset row, or None if unusable."""
    document_id = record.get("doc_id")
    content = record.get("content")
    if not document_id or not content:
        return None
    source_name = truncate_utf16(str(record.get("title") or document_id), 512)
    return {"document_id": str(document_id), "source_name": source_name, "content": str(content)}


def seed_documents(engine: Engine, limit: int | None = None) -> int:
    """Upsert every valid document row and return the count written."""
    documents = load_dataset(DATASET_NAME, "documents", split="test")
    count = 0
    with engine.begin() as connection:
        for record in documents:
            parameters = to_document_row(record)
            if parameters is None:
                continue
            connection.execute(UPSERT_SQL, parameters)
            count += 1
            if limit and count >= limit:
                break
    return count


def main(limit: int | None) -> None:
    """Seed the configured MSSQL database with real enterprise document content."""
    settings = Settings()
    engine = create_sql_engine(settings)
    written = seed_documents(engine, limit)
    print(f"Seeded {written} documents into dbo.documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Optional cap for a quick local test.")
    main(parser.parse_args().limit)
