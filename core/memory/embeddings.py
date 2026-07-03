from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    dimension: int

    def embed(self, text: str) -> list[float]:
        ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self._model = None
        self.dimension = 384

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the production embedder. "
                    "Inject a fake embedder in tests or install the dependency."
                ) from exc
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._model = SentenceTransformer(self.model_name, local_files_only=True)

        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


@dataclass(frozen=True)
class HashingEmbedder:
    dimension: int = 16

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return values

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                values[index] += digest[index % len(digest)] / 255.0

        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0.0:
            return values

        return [value / magnitude for value in values]
