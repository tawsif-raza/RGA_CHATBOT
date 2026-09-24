"""Shared document chunking, so the dense (Pinecone) and keyword (BM25)
indexes are built over identical chunks - required for their chunk_ids to
line up when EnsembleRetriever fuses results from both.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_text(content: str) -> list[str]:
    """Split document content into the shared chunk boundaries used by every retrieval index."""
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(content)
