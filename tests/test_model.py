from types import SimpleNamespace

from backend.config import Settings
from backend.model import EnterpriseGenerator, INSUFFICIENT_CONTEXT_ANSWER
from backend.retriever import RetrievedChunk


def test_no_context_never_generates() -> None:
    assert EnterpriseGenerator(Settings()).generate("Question", []) == INSUFFICIENT_CONTEXT_ANSWER


def test_prompt_contains_source_and_question() -> None:
    generator = EnterpriseGenerator(Settings(max_context_characters=1000))
    chunk = RetrievedChunk("doc-1", "doc-1:0", "Handbook", "Vacation is 20 days.", 0.9, "HR", None)
    messages = generator.build_messages("How many vacation days?", [chunk])
    assert "Handbook" in messages[1]["content"]
    assert "How many vacation days?" in messages[1]["content"]
