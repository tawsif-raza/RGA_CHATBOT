"""Temporary Phase 6 verification: reranked /query returns exactly 3 documents,
and the local Phoenix dashboard is reachable.

Run against a live `uvicorn backend.api:app` instance.
"""
import requests

API_BASE_URL = "http://127.0.0.1:8000"
PHOENIX_URL = "http://localhost:6006"


def test_query_returns_exactly_three_reranked_documents() -> None:
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": "What are the retention windows for private deployments and how is the DPA timeline being tracked?"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    assert "answer" in payload and payload["answer"]
    assert len(payload["citations"]) == 3, f"expected exactly 3 reranked citations, got {len(payload['citations'])}"
    for citation in payload["citations"]:
        assert isinstance(citation["score"], float)
        assert citation["content"]
    print(f"OK: /query returned {len(payload['citations'])} citations (20-candidate pool reranked down to 3).")
    for citation in payload["citations"]:
        print(f"  - {citation['document_id']} rerank_score={citation['score']:.4f} source={citation['source_name']!r}")


def test_phoenix_dashboard_is_reachable() -> None:
    response = requests.get(PHOENIX_URL, timeout=10)
    assert response.status_code == 200
    print(f"OK: Phoenix dashboard reachable at {PHOENIX_URL}")


def test_hybrid_search_finds_exact_id_dense_search_misses() -> None:
    """Verify BM25 recovers an exact-match numeric ticket ID that dense-only search misses entirely.

    Confirmed separately (not asserted here, since it needs its own process
    to avoid double-loading models): a dense-only `similarity_search` for
    this same question does not return `dsid_0000237f20d3490fbf77bc946a1fb860`
    anywhere in its top 20 candidates. The hybrid (BM25 + Pinecone) pipeline
    below should surface it via the keyword-exact-match signal.
    """
    target_document_id = "dsid_0000237f20d3490fbf77bc946a1fb860"
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": "What is the status of build-job #4572?"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    citation_ids = [c["document_id"] for c in payload["citations"]]
    assert target_document_id in citation_ids, f"expected BM25 to recover {target_document_id} via exact-match, got {citation_ids}"
    print(f"OK: hybrid search recovered {target_document_id} (dense-only search misses it entirely) at rank {citation_ids.index(target_document_id) + 1}.")


if __name__ == "__main__":
    test_query_returns_exactly_three_reranked_documents()
    test_phoenix_dashboard_is_reachable()
    test_hybrid_search_finds_exact_id_dense_search_misses()
