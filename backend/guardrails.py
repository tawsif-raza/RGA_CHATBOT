"""Pre-retrieval guardrails: block prompt-injection attempts and out-of-domain
questions before the LangChain retriever (and therefore Pinecone and the
LLM) is ever invoked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_huggingface import HuggingFaceEmbeddings

GUARDRAIL_MESSAGE = "I am an enterprise knowledge assistant and cannot process this request."

GREETING_RESPONSE = "Hello! I am your enterprise RAG assistant. Ask me a question about our internal documents, tickets, or policies."

# Matches only a standalone greeting - the whole (trimmed) message, not a
# substring - so "hi" short-circuits before any retrieval, but "hi, what's
# our vacation policy?" still goes through retrieval instead of getting
# treated as a bare greeting.
_GREETING_PATTERN = re.compile(
    r"^(hi+|hy+|hello+|hey+|heya+|yo+|sup|howdy|greetings|good\s?(morning|afternoon|evening))\s*(there|team|all)?[!.,\s]*$",
    re.IGNORECASE,
)


def is_pure_greeting(question: str) -> bool:
    """Return whether `question` is just a standalone greeting with no actual question content."""
    return bool(_GREETING_PATTERN.match(question.strip()))

# Fast heuristic filter for common prompt-injection phrasing. Not exhaustive -
# a determined attacker can phrase around any fixed pattern list - but it
# catches common, low-effort attempts cheaply, with no model call at all.
_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore (all|the|any|previous) (previous |prior |above )?instructions",
        r"disregard (the|your|all) (system|previous) (prompt|instructions)",
        r"\byou are now\b",
        r"reveal (your|the) (system prompt|instructions)",
        r"\bact as (?!an? enterprise)",
        r"\bjailbreak\b",
        r"pretend (you|to) (are|be)",
        r"\bDAN\b",
        r"override (your|the) (guardrails|rules|restrictions)",
        r"what (is|are) your (system prompt|instructions)",
    ]
]


def is_prompt_injection(question: str) -> bool:
    """Return whether `question` matches a known prompt-injection pattern."""
    return any(pattern.search(question) for pattern in _INJECTION_PATTERNS)


# Small set of reference sentences describing what this assistant is
# actually for, used as a cheap semantic router: embed once, then compare
# each incoming question's embedding against them. Reuses whichever
# embedding model the caller already has loaded (the retriever's), rather
# than adding a second one.
DOMAIN_EXEMPLARS = [
    "What is our company's policy on employee vacation and leave?",
    "How do I deploy this service to production and roll back if it fails?",
    "What are the data retention and security requirements for this vendor?",
    "Summarize the incident postmortem and the follow-up action items.",
    "What is the process for onboarding a new engineering hire?",
    "Explain the architecture of our internal data pipeline.",
    "Who approved the vendor security questionnaire and when is it due?",
    "What are the terms of the data processing agreement with this partner?",
]

# Below this cosine similarity to every domain exemplar, a question is
# treated as out-of-domain. Calibrated against real BGE-small embeddings
# (not guessed): a small ad-hoc sample of clearly out-of-domain questions
# ("capital of France", "recipe for chocolate cake", ...) scored 0.37-0.47,
# while clearly in-domain enterprise questions scored 0.75-0.89 - except
# unusually-phrased ones close to the exemplar set's blind spots, which is
# why exemplars were added above rather than just raising the threshold.
# This remains a heuristic, not a validated classifier: expect occasional
# false positives/negatives at the margin, especially for in-domain
# questions phrased very differently from every exemplar.
OUT_OF_DOMAIN_THRESHOLD = 0.55


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class DomainRouter:
    """Cheap semantic out-of-domain check, reusing an already-loaded embedding model."""

    embeddings: HuggingFaceEmbeddings
    threshold: float = OUT_OF_DOMAIN_THRESHOLD
    _exemplar_vectors: list[list[float]] | None = field(default=None, init=False, repr=False)

    def _ensure_exemplars(self) -> list[list[float]]:
        if self._exemplar_vectors is None:
            self._exemplar_vectors = self.embeddings.embed_documents(DOMAIN_EXEMPLARS)
        return self._exemplar_vectors

    def max_similarity(self, question: str) -> float:
        """Return the highest cosine similarity between `question` and any domain exemplar."""
        query_vector = self.embeddings.embed_query(question)
        return max(_cosine_similarity(query_vector, exemplar) for exemplar in self._ensure_exemplars())

    def is_out_of_domain(self, question: str) -> bool:
        return self.max_similarity(question) < self.threshold


def check_guardrails(question: str, router: DomainRouter) -> str | None:
    """Return a violation reason ("prompt_injection" / "out_of_domain") if `question` should be blocked, else None."""
    if is_prompt_injection(question):
        return "prompt_injection"
    if router.is_out_of_domain(question):
        return "out_of_domain"
    return None
