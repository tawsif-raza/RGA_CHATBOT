"""MSSQL access helpers for enterprise document metadata."""
from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, text

from backend.config import Settings


def create_sql_engine(settings: Settings) -> Engine:
    """Create a pooled MSSQL engine for the configured Docker or remote host."""
    return create_engine(settings.sql_url, pool_pre_ping=True)


def check_sql_connection(engine: Engine) -> bool:
    """Return whether MSSQL is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def count_ingested_documents(engine: Engine) -> int:
    """Return how many documents are confirmed ingested into Pinecone (`dbo.ingestion_state` row count)."""
    with engine.connect() as connection:
        return connection.execute(text("SELECT COUNT(*) FROM dbo.ingestion_state")).scalar_one()


def get_document_metadata(engine: Engine, document_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch source metadata for retrieved document IDs."""
    if not document_ids:
        return {}
    placeholders = ", ".join(f":document_id_{index}" for index in range(len(document_ids)))
    parameters = {f"document_id_{index}": value for index, value in enumerate(document_ids)}
    query = text(f"SELECT document_id, source_name, department, role_name, updated_at FROM dbo.documents WHERE document_id IN ({placeholders})")
    with engine.connect() as connection:
        rows = connection.execute(query, parameters).mappings().all()
    return {str(row["document_id"]): dict(row) for row in rows}
