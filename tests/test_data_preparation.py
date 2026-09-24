from prepare_data import stable_split


def test_document_split_is_deterministic() -> None:
    assert stable_split("document-42") == stable_split("document-42")


def test_document_split_is_valid() -> None:
    assert stable_split("document-42") in {"train", "validation", "test"}
