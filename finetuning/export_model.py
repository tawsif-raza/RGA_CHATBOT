"""Merge a saved LoRA adapter into Qwen for standalone inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_model(base_model: str, adapter_path: Path, output_path: Path) -> None:
    """Merge and persist adapter weights and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu")
    PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload().save_pretrained(str(output_path), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/qlora"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/qwen-enterprise-merged"))
    options = parser.parse_args()
    export_model(options.base_model, options.adapter_path, options.output_path)
