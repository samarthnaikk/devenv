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
import re
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


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


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
    degraded: bool = False
    ordered: tuple[str, ...] = ()


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
        return [self._to_candidate(match) for match, _provider in self._collect_fused(task)]

    def _collect_fused(self, task: str) -> list[tuple[dict[str, Any], Any]]:
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
        return _fuse_provider_session_matches(provider_matches)[: self.max_candidates]

    @staticmethod
    def _to_candidate(match: dict[str, Any]) -> SessionCandidate:
        summary = match.get("summary")
        return SessionCandidate(
            session_id=getattr(summary, "session_id", ""),
            provider=str(getattr(summary, "provider", "") or ""),
            title=str(getattr(summary, "title", "") or ""),
            workspace_path=getattr(summary, "workspace_path", None),
            score=int(match.get("score") or 0),
            updated_at=str(getattr(summary, "updated_at", "") or ""),
            snippet=str(getattr(summary, "preview", "") or "")[:400],
        )

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
        fused = self._collect_fused(task)
        if not fused:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )
        candidates = [self._to_candidate(match) for match, _provider in fused]
        workspace_path = getattr(self.context_builder, "workspace_path", "") or ""
        try:
            result = self.selector.select(  # type: ignore[union-attr]
                task, candidates, workspace_path=workspace_path
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Session selector failed; falling back to engine: error=%s", exc)
            result = None

        if result is None or result.degraded:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )

        by_id: dict[str, tuple[dict[str, Any], Any]] = {}
        for match, provider in fused:
            session_id = getattr(match.get("summary"), "session_id", "")
            if session_id:
                by_id.setdefault(session_id, (match, provider))
        selected_fused = [by_id[sid] for sid in result.session_ids if sid in by_id]

        metadata: dict[str, Any] = {
            "context_match_state": "reused_prior_sessions" if selected_fused else "new_context",
            "context_match_reason": result.reason
            or ("selector selected prior sessions" if selected_fused else "selector found no matching session"),
            "context_match_score": 0,
            "selector_applied": True,
            "selector_confidence": result.confidence,
            "selector_need_more": result.need_more,
            "selector_refined_query": result.refined_query or "",
            "selector_candidate_count": len(candidates),
            "selector_session_ids": list(result.session_ids),
            "selector_evidence": {key: list(value) for key, value in result.evidence.items()},
            "index_ready": True,
        }
        if not selected_fused:
            metadata["selector_abstained"] = True
            metadata["selector_evidence_used"] = False
            return "", (), metadata

        selected_ids = tuple(
            getattr(match.get("summary"), "session_id", "")
            for match, _provider in selected_fused
        )
        metadata["selector_abstained"] = False

        evidence_lines: list[str] = []
        for session_id in selected_ids:
            evidence_lines.extend(result.evidence.get(session_id, []))
        body = [
            normalized
            for normalized in (_normalize_evidence(line) for line in evidence_lines)
            if normalized
        ]
        if body:
            metadata["selector_evidence_used"] = True
            context = "\n".join(
                ["## External Session Context", *(f"- {line}" for line in body[:max_lines])]
            )
            return context, selected_ids, metadata

        metadata["selector_evidence_used"] = False
        lines = self.context_builder._context_lines_for_fused_matches(
            task, selected_fused, max_lines=max_lines
        )
        if not lines:
            return "", selected_ids, metadata
        context = "\n".join(["## External Session Context", *(f"- {line}" for line in lines)])
        return context, selected_ids, metadata
