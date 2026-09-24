from ingestion.build_bm25 import document_to_chunks
from ingestion.extract_sql import SourceDocument


def test_document_to_chunks_matches_pinecone_chunk_id_scheme() -> None:
    document = SourceDocument("doc-1", "Handbook", "HR", None, "short content")
    chunks = document_to_chunks(document)
    assert len(chunks) == 1
    assert chunks[0].page_content == "short content"
    assert chunks[0].metadata["document_id"] == "doc-1"
    assert chunks[0].metadata["chunk_id"] == "doc-1:0"
    assert chunks[0].metadata["source_name"] == "Handbook"
    assert chunks[0].metadata["department"] == "HR"
    assert chunks[0].metadata["role"] == ""


def test_document_to_chunks_splits_long_content_into_multiple_chunks() -> None:
    document = SourceDocument("doc-2", "Runbook", None, None, "word " * 500)
    chunks = document_to_chunks(document)
    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks] == [f"doc-2:{i}" for i in range(len(chunks))]
