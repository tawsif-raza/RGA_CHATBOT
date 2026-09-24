import pytest

from ingestion.embed_pinecone import (
    advance_watermark,
    batched,
    content_hash,
    load_checkpoint,
    resume_start_index,
    save_checkpoint,
    verify_index_dimension,
)
from ingestion.seed_documents import to_document_row, truncate_utf16


def test_content_hash_is_stable_and_sensitive() -> None:
    assert content_hash("same content") == content_hash("same content")
    assert content_hash("same content") != content_hash("changed content")


def test_batched_caps_batch_size_and_preserves_order() -> None:
    vectors = [{"id": str(i)} for i in range(450)]
    batches = list(batched(vectors, batch_size=200))
    assert [len(batch) for batch in batches] == [200, 200, 50]
    assert [v["id"] for batch in batches for v in batch] == [str(i) for i in range(450)]


def test_batched_handles_empty_input() -> None:
    assert list(batched([], batch_size=200)) == []


def test_verify_index_dimension_accepts_matching_dimension() -> None:
    verify_index_dimension(384, 384)


def test_verify_index_dimension_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError):
        verify_index_dimension(1536, 384)


def test_to_document_row_skips_missing_fields() -> None:
    assert to_document_row({"doc_id": "d1", "title": "T", "content": ""}) is None
    assert to_document_row({"doc_id": "", "title": "T", "content": "body"}) is None


def test_to_document_row_truncates_oversized_title() -> None:
    row = to_document_row({"doc_id": "d1", "title": "x" * 1000, "content": "body"})
    assert row is not None
    assert len(row["source_name"].encode("utf-16-le")) // 2 == 512


def test_truncate_utf16_does_not_overflow_on_astral_characters() -> None:
    # An emoji (astral character) is 1 Python code point but 2 UTF-16 units,
    # so a naive Python-length slice can still overflow the SQL column by one unit.
    value = ("x" * 511) + "\U0001f600" + "y"
    result = truncate_utf16(value, 512)
    assert len(result.encode("utf-16-le")) // 2 <= 512


def test_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    assert load_checkpoint(path) is None
    save_checkpoint(path, "dsid_0001")
    assert load_checkpoint(path) == "dsid_0001"
    save_checkpoint(path, "dsid_0002")
    assert load_checkpoint(path) == "dsid_0002"


def test_resume_start_index_with_no_checkpoint() -> None:
    assert resume_start_index(["a", "b", "c"], None) == 0


def test_resume_start_index_skips_past_checkpointed_document() -> None:
    doc_order = ["dsid_0001", "dsid_0002", "dsid_0003"]
    assert resume_start_index(doc_order, "dsid_0002") == 2


def test_resume_start_index_handles_checkpoint_at_end() -> None:
    doc_order = ["dsid_0001", "dsid_0002"]
    assert resume_start_index(doc_order, "dsid_0002") == 2


def test_advance_watermark_stops_at_first_gap() -> None:
    doc_order = ["a", "b", "c", "d"]
    # "d" finished before "b" and "c" (out-of-order async completion), but the
    # watermark must not skip past the still-incomplete "b".
    completed = {"a", "d"}
    assert advance_watermark(doc_order, 0, completed) == 1


def test_advance_watermark_advances_through_contiguous_run() -> None:
    doc_order = ["a", "b", "c", "d"]
    completed = {"a", "b", "c"}
    assert advance_watermark(doc_order, 0, completed) == 3
