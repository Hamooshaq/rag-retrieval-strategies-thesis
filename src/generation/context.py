from __future__ import annotations

from typing import Protocol

from storage.models import ContextPack, RetrievalHit


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class WhitespaceTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class HuggingFaceTokenCounter:
    def __init__(self, model_id: str, revision: str, local_files_only: bool = True):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class ContextBuilder:
    def __init__(self, token_counter: TokenCounter, max_tokens: int, locator_prefix: bool = True):
        if max_tokens <= 0:
            raise ValueError("Context budget must be positive")
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.locator_prefix = locator_prefix

    def build(
        self,
        hits: list[RetrievalHit],
        threshold: float,
        apply_threshold: bool = True,
    ) -> ContextPack:
        if not hits:
            return ContextPack(
                text="",
                token_count=0,
                refused_before_generation=apply_threshold,
            )
        if apply_threshold and max(hit.score for hit in hits) < threshold:
            return ContextPack(
                text="",
                token_count=0,
                refused_before_generation=True,
            )

        pieces: list[str] = []
        locators: list[str] = []
        unit_ids: list[str] = []
        truncated = False
        seen_units: set[str] = set()
        for hit in hits:
            if hit.unit_id in seen_units:
                continue
            seen_units.add(hit.unit_id)
            prefix = " ".join(f"[{locator}]" for locator in hit.locators) if self.locator_prefix else ""
            piece = f"{prefix} {hit.text}".strip()
            candidate_text = "\n\n".join([*pieces, piece])
            if self.token_counter.count(candidate_text) <= self.max_tokens:
                pieces.append(piece)
                locators.extend(hit.locators)
                unit_ids.append(hit.unit_id)
                continue

            truncated = True
            # Pack a partial final hit without ever truncating evidence already
            # accepted. Metadata is added only when at least one body token from
            # this hit survives; a locator-only prefix is not valid evidence.
            low = 0
            high = max(1, self.token_counter.count(hit.text))
            best_body = ""
            while low <= high:
                middle = (low + high) // 2
                body = self.token_counter.truncate(hit.text, middle).strip()
                partial_piece = f"{prefix} {body}".strip() if body else ""
                partial_candidate = (
                    "\n\n".join([*pieces, partial_piece]) if partial_piece else ""
                )
                if (
                    body
                    and self.token_counter.count(partial_candidate) <= self.max_tokens
                ):
                    best_body = body
                    low = middle + 1
                else:
                    high = middle - 1
            if best_body:
                pieces.append(f"{prefix} {best_body}".strip())
                locators.extend(hit.locators)
                unit_ids.append(hit.unit_id)
            break
        text = "\n\n".join(pieces)
        actual_tokens = self.token_counter.count(text)
        if actual_tokens > self.max_tokens:
            raise ValueError(
                f"Context exceeds budget after construction: {actual_tokens}>{self.max_tokens}"
            )
        return ContextPack(
            text=text,
            token_count=actual_tokens,
            locators=list(dict.fromkeys(locators)),
            unit_ids=unit_ids,
            truncated=truncated,
            refused_before_generation=not bool(text),
        )
