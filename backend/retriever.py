"""Hybrid (BM25 + Pinecone) retrieval via LangChain, reranked with a cross-encoder."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from backend.config import Settings

DEFAULT_TOP_K = 3

# The base retrievers each over-fetch this many candidates (20 dense + 20
# keyword, fused down to at most 40 pre-dedup) so the cross-encoder has a
# meaningful pool to rerank before truncating down to the requested top_k.
RERANK_CANDIDATE_K = 20

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Built by ingestion/build_bm25.py from documents already confirmed in
# Pinecone (dbo.ingestion_state), chunked identically to the dense index.
BM25_INDEX_PATH = Path("model_cache/bm25_index.pkl")

# Equal weighting: neither the dense nor the keyword signal is assumed to be
# more reliable a priori. Not empirically tuned.
ENSEMBLE_WEIGHTS = [0.5, 0.5]


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk plus metadata required for a user-visible citation."""

    document_id: str
    chunk_id: str
    source_name: str
    content: str
    score: float
    department: str | None
    role: str | None


class ScoringCrossEncoderReranker(CrossEncoderReranker):
    """CrossEncoderReranker that also attaches its score to each returned document's metadata.

    The stock `CrossEncoderReranker.compress_documents` uses the cross-encoder
    score only to sort and truncate, then discards it - but citations need a
    relevance score per chunk, and Phase 6's telemetry requirement explicitly
    wants reranker scores logged, so the score has to survive past this step.
    """

    def compress_documents(self, documents: Any, query: str, callbacks: Any = None) -> list[Document]:
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        reranked_documents = []
        for document, score in ranked[: self.top_n]:
            reranked_documents.append(Document(page_content=document.page_content, metadata={**document.metadata, "rerank_score": float(score)}))
        return reranked_documents


def load_bm25_retriever(path: Path = BM25_INDEX_PATH, top_k: int = RERANK_CANDIDATE_K) -> BaseRetriever | None:
    """Load the pickled BM25 index built by `ingestion/build_bm25.py`, or None if it hasn't been built yet.

    Missing rather than raising: a fresh checkout that hasn't run
    `build_bm25.py` should still serve dense-only search instead of failing
    to start.
    """
    if not path.exists():
        return None
    with path.open("rb") as handle:
        retriever: BM25Retriever = pickle.load(handle)
    retriever.k = top_k
    return retriever


class EnterpriseRetriever:
    """Hybrid dense (Pinecone) + keyword (BM25) retrieval, reranked via a cross-encoder.

    BM25 is exact-match/keyword-strong where dense embedding similarity is
    weak - internal jargon, ticket IDs, acronyms - so fusing the two before
    reranking should recover cases the embedding model alone misses.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model, encode_kwargs={"normalize_embeddings": True})
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index_name)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace=settings.pinecone_namespace,
        )
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        self.bm25_retriever = load_bm25_retriever()

    def health(self) -> bool:
        """Return whether the configured Pinecone index is accessible."""
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False

    def as_retriever(self, top_k: int = RERANK_CANDIDATE_K, department: str | None = None, role: str | None = None) -> BaseRetriever:
        """Return the raw dense (Pinecone) LangChain retriever, for chain composition."""
        metadata_filter = self._build_filter(department, role)
        search_kwargs: dict[str, Any] = {"k": top_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def hybrid_retriever(self, department: str | None = None, role: str | None = None) -> BaseRetriever:
        """Return the fused dense+keyword retriever, or dense-only if no BM25 index has been built.

        Note: BM25 has no notion of the Pinecone `department`/`role`
        metadata filter, so those constraints only apply to the dense leg
        when both retrievers are ensembled. Not currently exercised in
        practice (nothing in this project calls `search()` with a
        department/role filter yet).
        """
        pinecone_retriever = self.as_retriever(top_k=RERANK_CANDIDATE_K, department=department, role=role)
        if self.bm25_retriever is None:
            return pinecone_retriever
        return EnsembleRetriever(retrievers=[self.bm25_retriever, pinecone_retriever], weights=ENSEMBLE_WEIGHTS)

    def reranking_retriever(self, top_k: int = DEFAULT_TOP_K, department: str | None = None, role: str | None = None) -> ContextualCompressionRetriever:
        """Return a retriever that fuses dense+keyword candidates and reranks them down to `top_k` with the cross-encoder."""
        base_retriever = self.hybrid_retriever(department=department, role=role)
        reranker = ScoringCrossEncoderReranker(model=self.cross_encoder, top_n=top_k)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

    @staticmethod
    def _build_filter(department: str | None, role: str | None) -> dict[str, Any]:
        """Build a Pinecone metadata filter from optional department/role constraints."""
        metadata_filter: dict[str, Any] = {}
        if department:
            metadata_filter["department"] = {"$eq": department}
        if role:
            metadata_filter["role"] = {"$eq": role}
        return metadata_filter

    def search(self, question: str, top_k: int = DEFAULT_TOP_K, department: str | None = None, role: str | None = None) -> list[RetrievedChunk]:
        """Fuse dense+keyword candidates and return the top `top_k` after cross-encoder reranking."""
        retriever = self.reranking_retriever(top_k, department, role)
        documents = retriever.invoke(question)
        chunks: list[RetrievedChunk] = []
        for document in documents:
            metadata = document.metadata or {}
            content = document.page_content
            if content:
                chunks.append(RetrievedChunk(str(metadata["document_id"]), str(metadata["chunk_id"]), str(metadata.get("source_name", "Unknown source")), content, float(metadata["rerank_score"]), metadata.get("department") or None, metadata.get("role") or None))
        return chunks
