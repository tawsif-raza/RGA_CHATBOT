"""Build a local BM25 keyword index over every already-ingested document's chunks.

Reads from `dbo.documents` joined against `dbo.ingestion_state` (only
documents actually confirmed in Pinecone), chunks them with the exact same
boundaries used for the dense index (`ingestion/chunking.py`), and pickles
the resulting `BM25Retriever` for `backend/retriever.py` to load instantly
at API startup rather than recomputing term frequencies over the whole
corpus every time.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.config import Settings
from backend.database import create_sql_engine
from ingestion.chunking import chunk_text
from ingestion.extract_sql import SourceDocument, iter_ingested_documents

DEFAULT_OUTPUT_PATH = Path("model_cache/bm25_index.pkl")


def document_to_chunks(document: SourceDocument) -> list[Document]:
    """Split one source document into the same chunks the dense index uses, as LangChain Documents.

    Metadata keys match `ingestion/embed_pinecone.py`'s Pinecone vector
    metadata (`document_id`, `chunk_id`, `source_name`, `department`,
    `role`) so `backend.retriever.EnterpriseRetriever.search()` can build a
    `RetrievedChunk` from a BM25 hit exactly the same way it does from a
    Pinecone hit.
    """
    chunks = chunk_text(document.content)
    result: list[Document] = []
    for position, chunk in enumerate(chunks):
        chunk_id = f"{document.document_id}:{position}"
        result.append(Document(
            page_content=chunk,
            metadata={
                "document_id": document.document_id,
                "chunk_id": chunk_id,
                "source_name": document.source_name,
                "department": document.department or "",
                "role": document.role or "",
            },
        ))
    return result


def build_bm25_index(documents: list[SourceDocument]) -> BM25Retriever:
    """Chunk every document and build a BM25Retriever over the resulting chunks."""
    all_chunks: list[Document] = []
    for document in documents:
        all_chunks.extend(document_to_chunks(document))
    return BM25Retriever.from_documents(all_chunks)


def main(output_path: Path) -> None:
    settings = Settings()
    engine = create_sql_engine(settings)
    documents = list(iter_ingested_documents(engine))
    if not documents:
        raise RuntimeError("No ingested documents found in dbo.ingestion_state - run ingestion/embed_pinecone.py first.")
    retriever = build_bm25_index(documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(retriever, handle)
    chunk_count = len(retriever.docs)
    print(f"Built BM25 index over {len(documents)} documents ({chunk_count} chunks); saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    main(arguments.output_path)
