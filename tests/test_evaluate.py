from backend.model import INSUFFICIENT_CONTEXT_ANSWER
from backend.retriever import RetrievedChunk
from evaluate_model import faithfulness_score


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk("doc-1", "doc-1:0", "Handbook", content, 0.9, None, None)


def test_faithfulness_score_declining_to_answer_is_not_a_hallucination() -> None:
    assert faithfulness_score(INSUFFICIENT_CONTEXT_ANSWER, []) == 1.0
    assert faithfulness_score(INSUFFICIENT_CONTEXT_ANSWER, [_chunk("some context")]) == 1.0


def test_faithfulness_score_with_no_chunks_is_one() -> None:
    assert faithfulness_score("Vacation is twenty days per year.", []) == 1.0


def test_faithfulness_score_fully_grounded_answer_is_one() -> None:
    context = [_chunk("Vacation policy grants twenty days of paid leave annually.")]
    assert faithfulness_score("Vacation grants twenty days paid leave.", context) == 1.0


def test_faithfulness_score_penalizes_ungrounded_words() -> None:
    context = [_chunk("Vacation policy grants twenty days of paid leave annually.")]
    score = faithfulness_score("Vacation policy allows unlimited sabbatical trips abroad.", context)
    assert 0.0 < score < 1.0
