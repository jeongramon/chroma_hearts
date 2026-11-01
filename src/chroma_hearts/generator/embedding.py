"""Deterministic hash-based embedding for fast forensic experiments."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

DEFAULT_HASH_DIMENSION = 384


def _digest_to_finite_float(digest: bytes) -> float:
    """Map digest bytes to a finite float in [-1.0, 1.0]."""
    raw = int.from_bytes(digest[:4], "little")
    return (raw / 0xFFFFFFFF) * 2.0 - 1.0


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector; return zeros unchanged."""
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Cannot normalize a vector containing NaN or Infinity.")

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def hash_text_to_vector(text: str, dimension: int) -> list[float]:
    """Map text to a deterministic normalized float vector."""
    if dimension <= 0:
        raise ValueError("dimension must be greater than 0.")

    if not text:
        return [0.0] * dimension

    vector = [0.0] * dimension
    seed = text.encode("utf-8")
    for index in range(dimension):
        digest = hashlib.sha256(seed + index.to_bytes(4, "little")).digest()
        vector[index] = _digest_to_finite_float(digest)
    return _normalize(vector)


@register_embedding_function
class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fast deterministic embedding based on SHA-256 hashing."""

    def __init__(self, dimension: int = DEFAULT_HASH_DIMENSION) -> None:
        self._dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            documents = [input]
        else:
            documents = list(input)
        return [hash_text_to_vector(document, self._dimension) for document in documents]

    @staticmethod
    def name() -> str:
        return "chroma_hearts_hash"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> HashEmbeddingFunction:
        return HashEmbeddingFunction(dimension=int(config.get("dimension", DEFAULT_HASH_DIMENSION)))

    def get_config(self) -> dict[str, Any]:
        return {"dimension": self._dimension}


def build_embedding_function(mode: str, dimension: int = DEFAULT_HASH_DIMENSION) -> EmbeddingFunction[Documents]:
    """Create an embedding function from config.toml settings."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "hash":
        return HashEmbeddingFunction(dimension=dimension)
    if normalized_mode == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()
    raise ValueError(
        f"Unsupported embedding mode '{mode}'. Supported values: default, hash."
    )
