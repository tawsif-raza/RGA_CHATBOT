"""Create leakage-safe Qwen chat datasets from EnterpriseRAG-Bench."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DATASET_NAME = "onyx-dot-app/EnterpriseRAG-Bench"
SPLITS = ("train", "validation", "test")


def stable_split(document_id: str) -> str:
    """Assign a document to a deterministic 80/10/10 split."""
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def build_examples(questions: Dataset, documents: Dataset) -> dict[str, list[dict[str, Any]]]:
    """Return chat examples grouped by a source-document-safe split."""
    content_by_document = {str(item["doc_id"]): str(item["content"]) for item in documents if item.get("doc_id") and item.get("content")}
    examples: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for item in questions:
        document_ids = item.get("expected_doc_ids") or []
        if not document_ids or not item.get("question") or not item.get("gold_answer"):
            continue
        document_id = str(document_ids[0])
        context = content_by_document.get(document_id)
        if context is None:
            continue
        examples[stable_split(document_id)].append({
            "document_id": document_id,
            "messages": [
                {"role": "system", "content": "You are an enterprise AI. Answer only from supplied context. If context is insufficient, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{item['question']}"},
                {"role": "assistant", "content": str(item["gold_answer"])},
            ],
        })
    return examples


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write records to JSONL and return the written count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(output_dir: Path) -> None:
    """Download source data and write train, validation, and test JSONL files."""
    questions = load_dataset(DATASET_NAME, "questions", split="test")
    documents = load_dataset(DATASET_NAME, "documents", split="test")
    for split, records in build_examples(questions, documents).items():
        print(f"{split}: {write_jsonl(records, output_dir / f'{split}.jsonl')} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/finetune"))
    main(parser.parse_args().output_dir)
