from __future__ import annotations

import gc
import hashlib
import re
import time
from dataclasses import dataclass

from retrieval.embeddings import ModelUnavailableError


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    serialized_prompt: str = ""
    serialized_prompt_sha256: str = ""
    reached_token_limit: bool = False


class ExtractiveGenerator:
    """Deterministic test generator that returns the first evidence sentence and locator."""

    model_id = "deterministic-extractive"
    revision = "1"

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        evidence_match = re.search(r"EVIDENCE:\n(.*?)\n\nQUESTION:", prompt, re.DOTALL)
        evidence = evidence_match.group(1).strip() if evidence_match else ""
        if not evidence or evidence == "(empty)":
            text = "INSUFFICIENT_EVIDENCE"
        else:
            sentence = re.split(r"(?<=[.!?])\s+", evidence)[0]
            text = sentence[:500]
        return GenerationResult(
            text=text,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )


class TransformersGenerator:
    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "auto",
        local_files_only: bool = True,
        max_new_tokens: int = 96,
        do_sample: bool = False,
        seed: int = 260226,
    ):
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.seed = seed
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, local_files_only=local_files_only
            )
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = device
            dtype = torch.float16 if device == "mps" else torch.float32
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                self.model.to(device)
            except (RuntimeError, NotImplementedError) as device_exc:
                if device != "mps":
                    raise
                if hasattr(self, "model"):
                    del self.model
                gc.collect()
                torch.mps.empty_cache()
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.model.to("cpu")
                self.device_fallback_reason = str(device_exc)
            self.model.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"Generator model unavailable: {model_id}@{revision}: {exc}"
            ) from exc

    def runtime_metadata(self) -> dict:
        return {
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "torch_version": self.torch.__version__,
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_model_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "device_fallback_reason": getattr(self, "device_fallback_reason", None),
            "seed": self.seed,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "latency_clock": "perf_counter; synchronize device before/after inference",
        }

    def _synchronize(self) -> None:
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def generate(self, prompt: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": "You are a careful closed-corpus research assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        self._synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self._synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        new_tokens = generated[0, prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            latency_ms=elapsed,
            prompt_tokens=prompt_tokens,
            output_tokens=int(new_tokens.shape[0]),
            serialized_prompt=formatted,
            serialized_prompt_sha256=hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
            reached_token_limit=int(new_tokens.shape[0]) >= self.max_new_tokens,
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        if getattr(self, "device", "") == "mps":
            self.torch.mps.empty_cache()
