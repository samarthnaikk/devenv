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

    @property
    def identifier(self) -> str:
        return f"{type(self).__name__}:{self.model_name}:{self.dimension}"

    def _load_model(self) -> None:
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

    def warm(self) -> None:
        """Load the underlying model now instead of on first ``embed``."""

        if self._model is None:
            self._load_model()

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            self._load_model()

        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


@dataclass(frozen=True)
class HashingEmbedder:
    dimension: int = 16

    @property
    def identifier(self) -> str:
        return f"{type(self).__name__}::{self.dimension}"

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


@dataclass
class BgeSmallEmbedder:
    model_name: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    local_files_only: bool = True

    def __post_init__(self) -> None:
        self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_CARD_EMBEDDER: Embedder | None = None


def build_card_embedder() -> Embedder:
    """Return the interaction-card embedder, loading from the local cache without HF network calls."""
    global _CARD_EMBEDDER
    if _CARD_EMBEDDER is not None:
        return _CARD_EMBEDDER
    for local_files_only in (True, False):
        try:
            embedder: Embedder = BgeSmallEmbedder(local_files_only=local_files_only)
            embedder.embed("warmup")
        except Exception:
            continue
        _CARD_EMBEDDER = embedder
        return embedder
    _CARD_EMBEDDER = build_default_embedder()
    return _CARD_EMBEDDER


class LazyFallbackEmbedder:
    """Defer ``sentence-transformers`` loading until the first embed.

    Construction is cheap, so the TUI can paint its first frame before the
    model loads. The first ``embed`` (or an explicit ``warm``) resolves the real
    embedder and permanently falls back to the deterministic hashing embedder
    if the model cannot be loaded.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._primary = SentenceTransformerEmbedder(model_name)
        self._resolved: Embedder | None = None

    @property
    def _active(self) -> Embedder:
        return self._resolved or self._primary

    @property
    def dimension(self) -> int:
        return int(getattr(self._active, "dimension", 384))

    @property
    def model_name(self) -> str:
        return str(getattr(self._active, "model_name", "all-MiniLM-L6-v2"))

    @property
    def identifier(self) -> str:
        return str(getattr(self._active, "identifier", ""))

    def _ensure(self) -> Embedder:
        if self._resolved is None:
            try:
                self._primary.warm()
            except Exception:
                self._resolved = HashingEmbedder(dimension=384)
            else:
                self._resolved = self._primary
        return self._resolved

    def warm(self) -> None:
        self._ensure()

    def embed(self, text: str) -> list[float]:
        return self._ensure().embed(text)


_DEFAULT_EMBEDDER: Embedder | None = None


def _lazy_embedder_enabled() -> bool:
    return os.getenv("DEVENV_EMBEDDER_LAZY", "1").strip().lower() not in {"0", "false", "no", "off"}


def build_default_embedder() -> Embedder:
    """Return the production embedder, falling back to hashing if unavailable.

    By default the embedder is lazy: construction does not load the model, so
    callers can defer the expensive load to a background warm-up. Set
    ``DEVENV_EMBEDDER_LAZY=0`` to probe eagerly at construction time.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER
    if not _lazy_embedder_enabled():
        try:
            embedder: Embedder = SentenceTransformerEmbedder()
            embedder.warm()
        except Exception:
            _DEFAULT_EMBEDDER = HashingEmbedder(dimension=384)
        else:
            _DEFAULT_EMBEDDER = embedder
        return _DEFAULT_EMBEDDER
    _DEFAULT_EMBEDDER = LazyFallbackEmbedder()
    return _DEFAULT_EMBEDDER


def warm_default_embedder(embedder: Embedder | None = None) -> None:
    """Resolve a lazy embedder (defaults to the shared instance) if it supports warm-up."""

    target = embedder if embedder is not None else _DEFAULT_EMBEDDER
    warmer = getattr(target, "warm", None)
    if callable(warmer):
        warmer()
