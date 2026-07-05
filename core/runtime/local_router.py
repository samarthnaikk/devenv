from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

KNOWLEDGE_PROTOTYPES = (
    "how does this backend work",
    "explain this project architecture",
    "tell me about the previous project we discussed",
    "what do you remember about this repo",
    "how does the system work",
)

REMOTE_PROTOTYPES = (
    "create a new file and implement a feature",
    "fix the bug in this codebase",
    "edit the backend and update the frontend",
    "write code for a new component",
    "remove a file and refactor the module",
)


@dataclass(frozen=True)
class LocalRouteDecision:
    use_local_knowledge: bool
    confidence: float
    knowledge_score: float
    remote_score: float
    reason: str


class LocalIntentRouter:
    def __init__(self, threshold: float = 0.44) -> None:
        self.threshold = threshold

    def decide(self, prompt: str) -> LocalRouteDecision:
        text = prompt.strip()
        if not text:
            return LocalRouteDecision(False, 0.0, 0.0, 0.0, "empty prompt")

        if os.getenv("DEVENV_USE_EMBEDDING_ROUTER") != "1":
            lowered = text.lower()
            knowledge_hits = sum(
                1
                for token in ("how", "why", "what", "explain", "tell", "remember", "backend", "architecture", "project", "repo")
                if token in lowered
            )
            mutation_hits = sum(
                1
                for token in ("create", "make", "add", "fix", "edit", "update", "modify", "remove", "implement", "write")
                if token in lowered
            )
            repo_summary_hint = any(
                phrase in lowered
                for phrase in (
                    "summarize this repo",
                    "summarize the repo",
                    "summarize this repository",
                    "summarize the repository",
                    "explain the repo",
                    "explain this repo",
                    "explain the repository",
                    "explain this repository",
                    "summarize this codebase",
                    "summarize the codebase",
                )
            )
            use_local = (knowledge_hits >= 2 and mutation_hits == 0) or (repo_summary_hint and mutation_hits == 0)
            return LocalRouteDecision(use_local, float(knowledge_hits - mutation_hits), float(knowledge_hits), float(mutation_hits), "heuristic")

        try:
            embeddings = _embedding_model().encode(
                [text, *KNOWLEDGE_PROTOTYPES, *REMOTE_PROTOTYPES],
                normalize_embeddings=True,
            )
            prompt_embedding = embeddings[0]
            knowledge_embeddings = embeddings[1 : 1 + len(KNOWLEDGE_PROTOTYPES)]
            remote_embeddings = embeddings[1 + len(KNOWLEDGE_PROTOTYPES) :]
            knowledge_score = max(_dot(prompt_embedding, candidate) for candidate in knowledge_embeddings)
            remote_score = max(_dot(prompt_embedding, candidate) for candidate in remote_embeddings)
            confidence = knowledge_score - remote_score
            lowered = text.lower()
            knowledge_hint = any(
                phrase in lowered
                for phrase in ("how does", "how do", "explain", "tell me about", "what do you remember", "how does")
            )
            use_local = (knowledge_score >= self.threshold and confidence >= 0.04) or (
                knowledge_hint and confidence >= 0.02
            )
            reason = "embedding classifier"
            return LocalRouteDecision(use_local, confidence, knowledge_score, remote_score, reason)
        except Exception as exc:
            logger.warning("Falling back to heuristic local routing: error=%s", exc)
            lowered = text.lower()
            knowledge_hits = sum(
                1
                for token in ("how", "why", "what", "explain", "tell", "remember", "backend", "architecture", "project", "repo")
                if token in lowered
            )
            mutation_hits = sum(
                1
                for token in ("create", "make", "add", "fix", "edit", "update", "modify", "remove", "implement", "write")
                if token in lowered
            )
            use_local = knowledge_hits >= 2 and mutation_hits == 0
            return LocalRouteDecision(use_local, float(knowledge_hits - mutation_hits), float(knowledge_hits), float(mutation_hits), "heuristic fallback")


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)


def _dot(left, right) -> float:
    return float((left * right).sum())
