"""Thread-safe local Qwen plus LoRA inference wrapper."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from backend.config import Settings
from backend.retriever import RetrievedChunk

INSUFFICIENT_CONTEXT_ANSWER = "I do not have sufficient retrieved company context to answer that question."


class EnterpriseGenerator:
    """Keep Qwen and its QLoRA adapter resident for API lifetime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer: object | None = None
        self.model: object | None = None
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        """Return whether model resources have been loaded."""
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load the base model and saved adapter once on a CUDA host."""
        adapter_path = Path(self.settings.lora_adapter_path)
        if not torch.cuda.is_available() or not adapter_path.exists():
            return
        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(self.settings.model_name, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = tokenizer
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))

    def build_messages(self, question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
        """Create the shared training-compatible chat prompt within its context budget."""
        context_parts: list[str] = []
        remaining = self.settings.max_context_characters
        for chunk in chunks:
            excerpt = chunk.content[:remaining]
            if not excerpt:
                break
            context_parts.append(f"[Source: {chunk.source_name}; document: {chunk.document_id}; chunk: {chunk.chunk_id}]\n{excerpt}")
            remaining -= len(excerpt)
        return [
            {"role": "system", "content": "You are an enterprise AI. Answer only from supplied context. If context is insufficient, say so."},
            {"role": "user", "content": f"Context:\n{'\n\n'.join(context_parts)}\n\nQuestion:\n{question}"},
        ]

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a deterministic answer, or report insufficient context."""
        if not chunks:
            return INSUFFICIENT_CONTEXT_ANSWER
        if not self.ready:
            raise RuntimeError("The local model is not ready. Train or configure the LoRA adapter first.")
        tokenizer = self.tokenizer
        model = self.model
        assert tokenizer is not None and model is not None
        prompt = tokenizer.apply_chat_template(self.build_messages(question, chunks), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with self.lock, torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=self.settings.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def generate_stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Generate an answer as a stream of text pieces, or yield the insufficient-context message once.

        Runs `model.generate()` on a background thread feeding a
        `TextIteratorStreamer`, which is the standard pattern for streaming
        with raw `transformers` generation (there is no LangChain LLM
        wrapper in this pipeline to stream through - `EnterpriseGenerator`
        calls `transformers` directly, as noted elsewhere in this project).
        Holds `self.lock` for the full duration, same as `generate()`, so a
        second concurrent request can't also drive the model at once.
        """
        if not chunks:
            yield INSUFFICIENT_CONTEXT_ANSWER
            return
        if not self.ready:
            raise RuntimeError("The local model is not ready. Train or configure the LoRA adapter first.")
        tokenizer = self.tokenizer
        model = self.model
        assert tokenizer is not None and model is not None
        prompt = tokenizer.apply_chat_template(self.build_messages(question, chunks), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.lock.acquire()
        try:
            def run_generation() -> None:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, max_new_tokens=self.settings.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            thread = threading.Thread(target=run_generation)
            thread.start()
            for text_piece in streamer:
                if text_piece:
                    yield text_piece
            thread.join()
        finally:
            self.lock.release()
