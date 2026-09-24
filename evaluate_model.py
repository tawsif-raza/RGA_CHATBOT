"""Evaluate held-out Enterprise RAG examples through the real, live retrieval pipeline.

Uses `backend.retriever.EnterpriseRetriever` end to end (Pinecone + the
BGE cross-encoder reranker), not a simulated or perfect-retrieval stand-in.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import Settings
from backend.model import INSUFFICIENT_CONTEXT_ANSWER, EnterpriseGenerator
from backend.retriever import DEFAULT_TOP_K, EnterpriseRetriever, RetrievedChunk

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def load_examples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read held-out chat-format JSONL examples only."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return records[:limit] if limit else records


def faithfulness_score(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Heuristic groundedness proxy: fraction of the answer's distinct content words that also appear in the retrieved context.

    This is a fast, free, local approximation of whether the answer's
    vocabulary is actually grounded in what was retrieved - it is NOT an
    NLI-based faithfulness judgment. This project's own CLAUDE.md disallows
    OpenAI as a judge model ("use Gemini or custom local models"), and no
    such judge is configured, so a lexical-overlap heuristic is what's
    honestly available today rather than a real semantic entailment check.
    A score of 1.0 for an "insufficient context" answer reflects that
    correctly declining to answer is not a hallucination.
    """
    if answer.strip() == INSUFFICIENT_CONTEXT_ANSWER or not chunks:
        return 1.0
    context_words = set(_WORD_PATTERN.findall(" ".join(chunk.content for chunk in chunks).lower()))
    answer_words = [word for word in _WORD_PATTERN.findall(answer.lower()) if len(word) > 3]
    if not answer_words:
        return 1.0
    grounded = sum(1 for word in answer_words if word in context_words)
    return grounded / len(answer_words)


def main(test_path: Path, limit: int | None) -> None:
    """Run real retrieval (with reranking) and generation against the held-out split."""
    settings = Settings()
    retriever = EnterpriseRetriever(settings)
    generator = EnterpriseGenerator(settings)
    generator.load()
    if not generator.ready:
        raise RuntimeError("A CUDA host and trained LoRA adapter are required for evaluation.")
    results: list[dict[str, Any]] = []
    for example in load_examples(test_path, limit):
        question = example["messages"][1]["content"].rsplit("Question:\n", 1)[-1]
        expected = example["messages"][2]["content"]
        document_id = example["document_id"]
        started = time.perf_counter()
        chunks = retriever.search(question, DEFAULT_TOP_K, None, None)
        answer = generator.generate(question, chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        results.append({
            "question": question,
            "expected_document_id": document_id,
            "expected_answer": expected,
            "context_recall_at_k": int(document_id in {chunk.document_id for chunk in chunks}),
            "answer": answer,
            "latency_ms": elapsed_ms,
            "faithfulness_score": faithfulness_score(answer, chunks),
        })
    frame = pd.DataFrame(results)
    frame.to_csv("evaluation_results.csv", index=False)
    print(f"Examples: {len(frame)}")
    print(f"Context Recall@{DEFAULT_TOP_K} (correct document in reranked context): {frame['context_recall_at_k'].mean():.3f}")
    print(f"Average latency (ms): {frame['latency_ms'].mean():.1f}")
    print(f"Faithfulness (lexical-overlap heuristic, not an NLI judge): {frame['faithfulness_score'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=Path("data/finetune/test.jsonl"))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    main(arguments.test_path, arguments.limit)
