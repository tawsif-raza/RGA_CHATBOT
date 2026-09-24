"""Train Qwen2.5-0.5B-Instruct with QLoRA on chat-format JSONL data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR = BASE_DIR / "outputs" / "qlora"


def format_example(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Render chat messages with the exact inference template."""
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}


def load_training_data(tokenizer: Any) -> tuple[Dataset, Dataset]:
    """Load prepared train and validation splits."""
    dataset = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl"), "validation": str(DATA_DIR / "validation.jsonl")})
    train = dataset["train"]
    validation = dataset["validation"]
    return train.map(lambda item: format_example(item, tokenizer), remove_columns=train.column_names), validation.map(lambda item: format_example(item, tokenizer), remove_columns=validation.column_names)


def main(dry_run: bool) -> None:
    """Validate inputs or run CUDA QLoRA training."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    train_dataset, validation_dataset = load_training_data(tokenizer)
    print(f"Training examples: {len(train_dataset)}; validation examples: {len(validation_dataset)}")
    if dry_run:
        print(train_dataset[0]["text"][:500])
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training.")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    config = SFTConfig(output_dir=str(OUTPUT_DIR), num_train_epochs=1, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10, save_steps=100, eval_strategy="steps", eval_steps=100, save_total_limit=2, bf16=True, report_to="none", max_length=512, model_init_kwargs={"device_map": "auto"})
    trainer = SFTTrainer(model=MODEL_NAME, args=config, train_dataset=train_dataset, eval_dataset=validation_dataset, processing_class=tokenizer, peft_config=lora, quantization_config=quantization)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
