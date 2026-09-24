"""Read enabled enterprise documents from MSSQL."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy import Engine, text

from backend.config import Settings
from backend.database import create_sql_engine

# Deterministic ordering is required so a resumable, checkpointed ingestion
# run (see ingestion/embed_pinecone.py) always sees documents in the same
# sequence across runs.
SELECT_DOCUMENTS_SQL = "SELECT document_id, source_name, department, role_name, content FROM dbo.documents WHERE is_enabled = 1 ORDER BY document_id"
SELECT_DOCUMENT_IDS_SQL = "SELECT document_id FROM dbo.documents WHERE is_enabled = 1 ORDER BY document_id"


@dataclass(frozen=True)
class SourceDocument:
    """Document content and metadata required for vector ingestion."""

    document_id: str
    source_name: str
    department: str | None
    role: str | None
    content: str


def _bounded_query(base_sql: str, limit: int | None, start_after: str | None) -> tuple[str, dict[str, str]]:
    """Build a `TOP`-bounded, optionally resume-filtered variant of a base SELECT.

    `TOP (n)` is interpolated directly rather than bound as a parameter
    because MSSQL requires it be a literal/expression, not a parameter
    placeholder in this position; `limit` is always an `int` (argparse-typed
    or internally computed), so this is not user-controlled SQL injection.
    """
    where = "WHERE is_enabled = 1"
    params: dict[str, str] = {}
    if start_after is not None:
        where += " AND document_id > :start_after"
        params["start_after"] = start_after
    top = f"TOP ({int(limit)}) " if limit is not None else ""
    sql = base_sql.format(top=top, where=where)
    return sql, params


def extract_documents(engine: Engine) -> list[SourceDocument]:
    """Fetch all enabled documents from the production MSSQL schema.

    Materializes the full result set. Fine for small/moderate corpora, but at
    full corpus scale (hundreds of thousands of documents) prefer
    `iter_documents`, which streams rows instead of holding every document's
    content in memory at once.
    """
    with engine.connect() as connection:
        rows = connection.execute(text(SELECT_DOCUMENTS_SQL)).mappings().all()
    return [SourceDocument(str(row["document_id"]), str(row["source_name"]), row["department"], row["role_name"], str(row["content"])) for row in rows]


def iter_documents(engine: Engine, yield_per: int = 500, start_after: str | None = None, limit: int | None = None) -> Iterator[SourceDocument]:
    """Stream enabled documents from MSSQL in deterministic `document_id` order.

    Uses a server-side cursor (`stream_results=True` + `yield_per`) so the
    full corpus's content never has to fit in memory at once, unlike
    `extract_documents`. `start_after` and `limit` push resume-skipping and
    row-count bounding into the SQL query itself (`TOP` + a `document_id >`
    filter), rather than fetching and discarding rows in Python.
    """
    sql, params = _bounded_query(
        "SELECT {top}document_id, source_name, department, role_name, content FROM dbo.documents {where} ORDER BY document_id",
        limit,
        start_after,
    )
    with engine.connect().execution_options(stream_results=True, yield_per=yield_per) as connection:
        result = connection.execute(text(sql), params)
        for row in result.mappings():
            yield SourceDocument(str(row["document_id"]), str(row["source_name"]), row["department"], row["role_name"], str(row["content"]))


def list_document_ids(engine: Engine, limit: int | None = None) -> list[str]:
    """Fetch just the ordered document IDs, cheaply, for resume/checkpoint bookkeeping."""
    sql, params = _bounded_query("SELECT {top}document_id FROM dbo.documents {where} ORDER BY document_id", limit, None)
    with engine.connect() as connection:
        rows = connection.execute(text(sql), params).scalars().all()
    return [str(row) for row in rows]


def iter_ingested_documents(engine: Engine, yield_per: int = 500) -> Iterator[SourceDocument]:
    """Stream only documents already confirmed in Pinecone (present in `dbo.ingestion_state`).

    Used to build the BM25 keyword index over exactly the same corpus the
    dense (Pinecone) retriever can actually search, rather than the full
    `dbo.documents` table (most of which isn't ingested yet).
    """
    sql = """
        SELECT d.document_id, d.source_name, d.department, d.role_name, d.content
        FROM dbo.documents d
        INNER JOIN dbo.ingestion_state i ON i.document_id = d.document_id
        WHERE d.is_enabled = 1
        ORDER BY d.document_id
    """
    with engine.connect().execution_options(stream_results=True, yield_per=yield_per) as connection:
        result = connection.execute(text(sql))
        for row in result.mappings():
            yield SourceDocument(str(row["document_id"]), str(row["source_name"]), row["department"], row["role_name"], str(row["content"]))


def main() -> None:
    """Print the count of real documents available for ingestion."""
    print(f"Extracted {len(extract_documents(create_sql_engine(Settings())))} enabled documents.")


if __name__ == "__main__":
    main()
