from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class ModelUnavailableError(RuntimeError):
    pass


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic lightweight embedder used by unit tests, not by the reported pilot."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        self.model_id = f"deterministic-hashing-{dimensions}"
        self.revision = "1"

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        return _normalize(matrix)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = True,
    ):
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Embedding model unavailable: {model_id}@{revision}: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
