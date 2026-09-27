"""Retrieval-selection orchestration layer.

This sits *after* the existing retrieval engine. It reads the engine's ranked
candidates through the engine's own methods (no engine changes), and — when a
selector is configured and enabled — uses an LLM to rerank/select them. With no
selector, or when disabled, it is a byte-for-byte passthrough of
``ContextBuilderService.build_runtime_memory_context``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .context_builder import (
    ContextBuilderService,
    _fuse_provider_session_matches,
)

logger = logging.getLogger(__name__)

_ENABLED_VALUES = {"1", "true", "yes", "on"}


def session_selector_enabled() -> bool:
    return os.getenv("DEVENV_SESSION_SELECTOR", "").strip().lower() in _ENABLED_VALUES


@dataclass(frozen=True)
class SessionCandidate:
    session_id: str
    provider: str
    title: str
    workspace_path: str | None
    score: int
    updated_at: str
    snippet: str = ""


@dataclass(frozen=True)
class SelectionResult:
    session_ids: tuple[str, ...]
    confidence: float = 0.0
    need_more: bool = False
    refined_query: str | None = None
    reason: str = ""
    evidence: dict[str, list[str]] = field(default_factory=dict)


class SessionSelector(Protocol):
    def select(
        self,
        task: str,
        candidates: Sequence[SessionCandidate],
        *,
        workspace_path: str,
    ) -> SelectionResult: ...


class SessionSelectionOrchestrator:
    """Collects engine candidates and (optionally) lets a selector choose among them."""

    def __init__(
        self,
        context_builder: ContextBuilderService,
        *,
        selector: SessionSelector | None = None,
        enabled: bool | None = None,
        max_candidates: int = 12,
    ) -> None:
        self.context_builder = context_builder
        self.selector = selector
        self.enabled = session_selector_enabled() if enabled is None else bool(enabled)
        self.max_candidates = max_candidates

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.selector is not None)

    def collect_candidates(self, task: str) -> list[SessionCandidate]:
        """Return the engine's fused candidates (read-only) with project metadata."""
        builder = self.context_builder
        try:
            provider_names = builder._candidate_provider_names()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Candidate provider lookup failed: error=%s", exc)
            return []

        provider_matches: list[tuple[str, Any, list[dict[str, Any]]]] = []
        for provider_name in provider_names:
            try:
                matches, provider, _metadata = builder._select_runtime_matches_for_provider(
                    task, provider_name=provider_name
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Candidate selection failed for provider=%s: error=%s", provider_name, exc
                )
                continue
            if provider is None or not matches:
                continue
            provider_matches.append((provider_name, provider, matches))

        if not provider_matches:
            return []

        fused = _fuse_provider_session_matches(provider_matches)[: self.max_candidates]
        candidates: list[SessionCandidate] = []
        for match, _provider in fused:
            summary = match.get("summary")
            if summary is None:
                continue
            candidates.append(
                SessionCandidate(
                    session_id=getattr(summary, "session_id", ""),
                    provider=str(getattr(summary, "provider", "") or ""),
                    title=str(getattr(summary, "title", "") or ""),
                    workspace_path=getattr(summary, "workspace_path", None),
                    score=int(match.get("score") or 0),
                    updated_at=str(getattr(summary, "updated_at", "") or ""),
                    snippet=str(getattr(summary, "preview", "") or "")[:400],
                )
            )
        return candidates

    def select(
        self,
        task: str,
        *,
        max_lines: int = 6,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        """Return ``(context, session_ids, metadata)``.

        With no active selector this is an exact passthrough of the engine. The
        selector path is added in a later phase; for now it also delegates.
        """
        if not self.active:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )
        return self._select_with_selector(task, max_lines=max_lines)

    def _select_with_selector(
        self,
        task: str,
        *,
        max_lines: int,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        # Placeholder: implemented alongside the selector prompt (phase B2).
        return self.context_builder.build_runtime_memory_context(
            task, max_lines=max_lines
        )
