from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalModelSelection:
    summary: str
    selected_lines: tuple[str, ...]
    used_fallback: bool
    model_name: str


class LocalSmallModel:
    model_name = "deterministic-fallback"

    def distill(self, prompt: str, candidates: list[str], *, max_lines: int = 6) -> LocalModelSelection:
        raise NotImplementedError


class SentenceTransformerLocalModel(LocalSmallModel):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None
        self._fallback = FallbackLocalModel()

    def distill(self, prompt: str, candidates: list[str], *, max_lines: int = 6) -> LocalModelSelection:
        clean_candidates = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
        if not clean_candidates:
            return LocalModelSelection(summary="", selected_lines=(), used_fallback=False, model_name=self.model_name)

        try:
            if self._model is None:
                self._model = _embedding_model()

            embeddings = self._model.encode(
                [prompt, *clean_candidates],
                normalize_embeddings=True,
            )
            prompt_embedding = embeddings[0]
            scored: list[tuple[float, str]] = []
            for index, candidate in enumerate(clean_candidates, start=1):
                score = float((prompt_embedding * embeddings[index]).sum())
                scored.append((score, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = tuple(line for _score, line in scored[:max_lines] if line)
            return LocalModelSelection(
                summary=_join_summary(selected),
                selected_lines=selected,
                used_fallback=False,
                model_name=self.model_name,
            )
        except Exception as exc:
            logger.warning("Falling back to deterministic local context model during distillation: error=%s", exc)
            return self._fallback.distill(prompt, clean_candidates, max_lines=max_lines)


class FallbackLocalModel(LocalSmallModel):
    model_name = "deterministic-fallback"

    def distill(self, prompt: str, candidates: list[str], *, max_lines: int = 6) -> LocalModelSelection:
        prompt_tokens = {token for token in re.findall(r"[a-z0-9_]+", prompt.lower()) if len(token) >= 3}
        scored: list[tuple[int, str]] = []
        for candidate in candidates:
            text = candidate.strip()
            if not text:
                continue
            lowered = text.lower()
            overlap = sum(1 for token in prompt_tokens if token in lowered)
            scored.append((overlap, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = tuple(text for score, text in scored[:max_lines] if score > 0) or tuple(
            candidate.strip() for candidate in candidates[:max_lines] if candidate.strip()
        )
        return LocalModelSelection(
            summary=_join_summary(selected),
            selected_lines=selected,
            used_fallback=True,
            model_name=self.model_name,
        )


def load_local_small_model() -> LocalSmallModel:
    try:
        return SentenceTransformerLocalModel()
    except Exception as exc:
        logger.warning("Falling back to deterministic local context model: error=%s", exc)
        return FallbackLocalModel()


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    offline = os.getenv("DEVENV_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=False)


def _join_summary(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    joined = " ".join(lines)
    return joined[:700].rstrip()
