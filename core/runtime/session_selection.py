"""Retrieval-selection orchestration layer.

This sits *after* the existing retrieval engine. It reads the engine's ranked
candidates through the engine's own methods (no engine changes), and — when a
selector is configured and enabled — uses an LLM to rerank/select them. With no
selector, or when disabled, it is a byte-for-byte passthrough of
``ContextBuilderService.build_runtime_memory_context``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .context_builder import (
    ContextBuilderService,
    _fuse_provider_session_matches,
)

logger = logging.getLogger(__name__)

_ENABLED_VALUES = {"1", "true", "yes", "on"}
DEFAULT_SELECTOR_MODEL = "opencode/longcat-2.5-preview-free"


def session_selector_enabled() -> bool:
    return os.getenv("DEVENV_SESSION_SELECTOR", "").strip().lower() in _ENABLED_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


_MAX_EVIDENCE_SEGMENT_CHARS = 600


def _evidence_segments(text: str) -> list[str]:
    """Split raw text into sentence-ish segments, before whitespace collapsing.

    Large tool outputs and markdown are split on newlines and sentence
    boundaries first, then over-long runs are hard-wrapped so a single monster
    block cannot dominate scoring.
    """
    segments: list[str] = []
    for block in re.split(r"\n+", str(text)):
        block = block.strip()
        if not block:
            continue
        for piece in re.split(r"(?<=[.!?;:])\s+", block):
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= _MAX_EVIDENCE_SEGMENT_CHARS:
                segments.append(piece)
                continue
            for start in range(0, len(piece), _MAX_EVIDENCE_SEGMENT_CHARS):
                segments.append(piece[start : start + _MAX_EVIDENCE_SEGMENT_CHARS])
    return segments


def _normalize_project_path(path: str | None) -> str:
    if not path:
        return ""
    return os.path.normpath(str(path).strip()).replace("\\", "/").rstrip("/").lower()


_PROJECT_GATE_MODES = {"off", "demote", "filter"}
_RECALL_FLOOR_MODES = {"off", "soft", "hard"}


def project_gate_mode() -> str:
    mode = os.getenv("DEVENV_SESSION_PROJECT_GATE", "off").strip().lower()
    return mode if mode in _PROJECT_GATE_MODES else "off"


def recall_floor_mode() -> str:
    mode = os.getenv("DEVENV_SESSION_SELECTOR_RECALL_FLOOR", "hard").strip().lower()
    return mode if mode in _RECALL_FLOOR_MODES else "hard"


def apply_project_gate(
    candidates: Sequence[SessionCandidate],
    *,
    workspace_path: str | None,
    query: str = "",
    mode: str = "demote",
) -> list[SessionCandidate]:
    """Soft-demote (or hard-filter) candidates that belong to another project.

    Deterministic backstop for when prompt-only project handling is insufficient.
    Candidates with an unknown workspace are kept; a foreign project is kept when
    the query explicitly names it. ``mode="off"`` is a no-op.
    """
    if mode not in {"demote", "filter"} or not candidates:
        return list(candidates)
    target = _normalize_project_path(workspace_path)
    query_lower = (query or "").lower()
    same_project: list[SessionCandidate] = []
    foreign: list[SessionCandidate] = []
    for candidate in candidates:
        session_workspace = _normalize_project_path(candidate.workspace_path)
        if not session_workspace:
            same_project.append(candidate)
            continue
        if target and session_workspace == target:
            same_project.append(candidate)
            continue
        project_name = session_workspace.rsplit("/", 1)[-1]
        if project_name and project_name in query_lower:
            same_project.append(candidate)
            continue
        foreign.append(candidate)
    if mode == "filter":
        return same_project
    return [*same_project, *foreign]


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
    coverage: tuple[dict[str, Any], ...] = ()


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
        max_attempts: int | None = None,
        min_confidence: float | None = None,
        shadow: bool | None = None,
        recall_floor: str | None = None,
        recall_floor_k: int | None = None,
        max_selected: int | None = None,
        drill_char_budget: int | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.selector = selector
        self.enabled = session_selector_enabled() if enabled is None else bool(enabled)
        self.drill_char_budget = (
            drill_char_budget
            if drill_char_budget is not None
            else _env_int("DEVENV_SESSION_SELECTOR_DRILL_CHARS", 2_000_000)
        )
        self.shadow = (
            os.getenv("DEVENV_SESSION_SELECTOR_SHADOW", "").strip().lower() in _ENABLED_VALUES
            if shadow is None
            else bool(shadow)
        )
        self.recall_floor = (
            recall_floor if recall_floor is not None else recall_floor_mode()
        )
        self.recall_floor_k = max(
            0,
            recall_floor_k
            if recall_floor_k is not None
            else _env_int("DEVENV_SESSION_SELECTOR_RECALL_FLOOR_K", 3),
        )
        self.recall_floor_wide_k = max(
            self.recall_floor_k,
            _env_int("DEVENV_SESSION_SELECTOR_RECALL_FLOOR_WIDE_K", 6),
        )
        self.max_selected = max(
            1,
            max_selected
            if max_selected is not None
            else _env_int("DEVENV_SESSION_SELECTOR_MAX_SELECTED", 5),
        )
        self.max_candidates = max_candidates
        self.max_attempts = max(
            1,
            max_attempts
            if max_attempts is not None
            else _env_int("DEVENV_SESSION_SELECTOR_MAX_ATTEMPTS", 2),
        )
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else _env_float("DEVENV_SESSION_SELECTOR_MIN_CONFIDENCE", 0.4)
        )

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.selector is not None)

    @property
    def uses_selector(self) -> bool:
        """True when calling ``select`` will actually run the selector or shadow it."""
        return self.active or (self.shadow and self.selector is not None)

    def collect_candidates(self, task: str) -> list[SessionCandidate]:
        """Return the engine's fused candidates (read-only) with project metadata."""
        return [
            self._to_candidate(match, task=task, provider=provider)
            for match, provider in self._collect_fused(task)
        ]

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
        fused = _fuse_provider_session_matches(provider_matches)
        gate_mode = project_gate_mode()
        if gate_mode != "off":
            workspace_path = getattr(self.context_builder, "workspace_path", "") or ""
            candidates = [
                self._to_candidate(match, task=task, provider=provider)
                for match, provider in fused
            ]
            gated = apply_project_gate(
                candidates, workspace_path=workspace_path, query=task, mode=gate_mode
            )
            order = {candidate.session_id: index for index, candidate in enumerate(gated)}
            fused = sorted(
                fused,
                key=lambda match_provider: order.get(
                    getattr(match_provider[0].get("summary"), "session_id", ""), 10**6
                ),
            )
        return fused[: self.max_candidates]

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text")
        return re.sub(r"\s+", " ", str(text)).strip() if text else ""

    @classmethod
    def _candidate_snippet(
        cls,
        match: dict[str, Any],
        *,
        task: str = "",
        provider: Any | None = None,
    ) -> str:
        """Prefer the engine's matched chunk text over the short session preview.

        The preview is often unrelated to the query; the ranked chunks are the
        exact passages the engine matched, so they are what the selector needs to
        quote evidence from. When the engine returns no chunks (which happens for
        candidate sessions whose best text was not surfaced), fall back to a
        token-scored slice of the session's own text so the selector is not blind
        to a session that may hold the answer.
        """
        snippets: list[str] = []
        seen: set[str] = set()
        for chunk in list(match.get("chunks") or [])[:5]:
            text = cls._chunk_text(chunk)
            if text and text not in seen:
                seen.add(text)
                snippets.append(text[:300])
        semantic_chunks = list(match.get("semantic_chunks") or [])
        if not semantic_chunks and match.get("semantic_chunk") is not None:
            semantic_chunks = [match["semantic_chunk"]]
        for chunk in semantic_chunks[:3]:
            text = cls._chunk_text(chunk)
            if text and text not in seen:
                seen.add(text)
                snippets.append(text[:300])
        if snippets:
            return "\n    ".join(snippets)[:1500]
        fallback = cls._fallback_snippet(match, task=task, provider=provider)
        if fallback:
            return fallback
        return str(getattr(match.get("summary"), "preview", "") or "")[:400]

    @classmethod
    def _fallback_snippet(
        cls,
        match: dict[str, Any],
        *,
        task: str,
        provider: Any | None,
    ) -> str:
        summary = match.get("summary")
        session_id = getattr(summary, "session_id", "")
        if not provider or not session_id or not task:
            return ""
        try:
            from .context_builder import _normalize_whitespace, _tokenize
        except Exception:  # pragma: no cover - defensive
            return ""
        tokens = _tokenize(task)
        if not tokens:
            return ""
        orchestrator_texts = getattr(provider, "_session_messages", None)
        texts: list[str] = []
        if callable(orchestrator_texts):
            try:
                texts = [
                    str(getattr(message, "content", "") or "")
                    for message in orchestrator_texts(session_id) or ()
                ]
            except Exception:  # pragma: no cover - defensive
                texts = []
        if not texts:
            try:
                detail = provider.get_session(session_id)
                texts = [
                    str(getattr(message, "content", "") or "")
                    for message in getattr(detail, "messages", ()) or ()
                ]
            except Exception:  # pragma: no cover - defensive
                return ""
        scored: list[tuple[int, str]] = []
        budget = 60_000
        consumed = 0
        for content in texts:
            if consumed >= budget:
                break
            consumed += len(content)
            for segment in _evidence_segments(content):
                line = _normalize_whitespace(segment)
                if len(line) < 40:
                    continue
                lowered = line.lower()
                hits = sum(1 for token in tokens if token in lowered)
                if hits:
                    scored.append((hits, line[:300]))
        if not scored:
            # No token overlap: fall back to the first substantial block so the
            # selector still sees real content rather than an empty snippet.
            for content in texts:
                line = _normalize_whitespace(content)
                if len(line) >= 40:
                    return line[:400]
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        return "\n    ".join(line for _score, line in scored[:3])[:1500]

    @classmethod
    def _to_candidate(
        cls,
        match: dict[str, Any],
        *,
        task: str = "",
        provider: Any | None = None,
    ) -> SessionCandidate:
        summary = match.get("summary")
        return SessionCandidate(
            session_id=getattr(summary, "session_id", ""),
            provider=str(getattr(summary, "provider", "") or ""),
            title=str(getattr(summary, "title", "") or ""),
            workspace_path=getattr(summary, "workspace_path", None),
            score=int(match.get("score") or 0),
            updated_at=str(getattr(summary, "updated_at", "") or ""),
            snippet=cls._candidate_snippet(match, task=task, provider=provider),
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
        if self.shadow and self.selector is not None:
            return self._select_shadow(task, max_lines=max_lines)
        if not self.active:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )
        return self._select_with_selector(task, max_lines=max_lines)

    def _select_shadow(
        self,
        task: str,
        *,
        max_lines: int,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        context, session_ids, metadata = self.context_builder.build_runtime_memory_context(
            task, max_lines=max_lines
        )
        result, _fused, attempts, final_query = self._run_selector(task)
        shadow_metadata = dict(metadata)
        shadow_metadata["selector_shadow"] = True
        if result is not None and not result.degraded:
            shadow_metadata["selector_session_ids"] = list(result.session_ids)
            shadow_metadata["selector_confidence"] = result.confidence
            shadow_metadata["selector_engine_ids"] = list(session_ids)
            shadow_metadata["selector_attempts"] = attempts
            shadow_metadata["selector_final_query"] = final_query
            logger.info(
                "Session selector shadow: engine=%s selector=%s confidence=%s",
                list(session_ids),
                list(result.session_ids),
                result.confidence,
            )
        return context, session_ids, shadow_metadata

    def _select_with_selector(
        self,
        task: str,
        *,
        max_lines: int,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        result, fused, attempts, final_query = self._run_selector(task)
        if result is None or result.degraded:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )
        if not fused:
            return self.context_builder.build_runtime_memory_context(
                task, max_lines=max_lines
            )
        candidates = [
            self._to_candidate(match, task=task, provider=provider)
            for match, provider in fused
        ]

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
            "selector_attempts": attempts,
            "selector_final_query": final_query,
            "index_ready": True,
        }
        if not selected_fused:
            metadata["selector_abstained"] = True
            metadata["selector_evidence_used"] = False
            # On a hard floor, never drop below the engine's own top-K: an
            # abstaining selector must not silently reduce recall.
            if self.recall_floor == "hard" and self.recall_floor_k > 0 and fused:
                fallback = fused[: min(self.recall_floor_wide_k, self.max_selected)]
                fallback_ids = tuple(
                    getattr(match.get("summary"), "session_id", "")
                    for match, _provider in fallback
                )
                metadata["selector_floor_applied"] = True
                metadata["selector_floor_ids"] = list(fallback_ids)
                metadata["selector_floor_mode"] = "hard"
                metadata["selector_abstained_fallback"] = True
                drill_lines = self._drill_evidence(
                    task, fallback, max_lines=max_lines
                )
                lines = self.context_builder._context_lines_for_fused_matches(
                    task, fallback, max_lines=max_lines
                )
                combined: list[str] = []
                for line in [*drill_lines, *(lines or ())]:
                    normalized = _normalize_evidence(line)
                    if normalized and normalized not in combined:
                        combined.append(normalized)
                if not combined:
                    return "", fallback_ids, metadata
                context = "\n".join(
                    ["## External Session Context", *(f"- {line}" for line in combined[:max_lines])]
                )
                return context, fallback_ids, metadata
            return "", (), metadata

        selected_fused, floor_ids = self._apply_recall_floor(fused, selected_fused, result)
        metadata["selector_floor_mode"] = self.recall_floor
        metadata["selector_floor_k"] = self.recall_floor_k
        metadata["selector_floor_wide_k"] = self.recall_floor_wide_k
        metadata["selector_floor_ids"] = list(floor_ids)
        metadata["selector_floor_applied"] = bool(floor_ids)

        selected_ids = tuple(
            getattr(match.get("summary"), "session_id", "")
            for match, _provider in selected_fused
        )
        metadata["selector_abstained"] = False

        evidence_lines: list[str] = []
        for session_id in selected_ids:
            evidence_lines.extend(result.evidence.get(session_id, []))
        for entry in result.coverage:
            evidence_lines.extend(entry.get("evidence", []))
        body = [
            normalized
            for normalized in (_normalize_evidence(line) for line in evidence_lines)
            if normalized
        ]

        engine_lines: list[str] = []
        try:
            engine_lines = [
                normalized
                for normalized in (
                    _normalize_evidence(line)
                    for line in self.context_builder._context_lines_for_fused_matches(
                        task, selected_fused, max_lines=max_lines * 2
                    )
                )
                if normalized
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Engine context-line build failed: error=%s", exc)

        drill_query = self._build_drill_query(task, result)
        drill_lines = self._drill_evidence(drill_query, selected_fused, max_lines=max_lines)

        combined: list[str] = []
        for line in [*body, *drill_lines, *engine_lines]:
            if line and line not in combined:
                combined.append(line)
        metadata["selector_evidence_used"] = bool(body)
        metadata["selector_engine_lines_used"] = bool(engine_lines)
        metadata["selector_drill_line_count"] = len(drill_lines)
        metadata["selector_coverage_subquestions"] = len(result.coverage)
        if not combined:
            return "", selected_ids, metadata
        context = "\n".join(
            ["## External Session Context", *(f"- {line}" for line in combined[:max_lines])]
        )
        return context, selected_ids, metadata

    def _apply_recall_floor(
        self,
        fused: list[tuple[dict[str, Any], Any]],
        selected_fused: list[tuple[dict[str, Any], Any]],
        result: SelectionResult,
    ) -> tuple[list[tuple[dict[str, Any], Any]], list[str]]:
        """Keep the engine's top-K sessions selected unless the floor is off.

        Guarantees the selector cannot silently drop the correct session that the
        engine already ranked highly (the Q6/Q7 failure mode).
        """
        mode = self.recall_floor
        if mode == "off" or self.recall_floor_k <= 0 or not fused:
            return selected_fused[: self.max_selected], []
        if mode == "soft" and result.confidence >= self.min_confidence:
            return selected_fused[: self.max_selected], []

        # Widen the floor when the selector is unsure, so a correct session ranked
        # below the narrow K is not silently dropped (the T1 hirex failure mode).
        low_confidence = (
            result.confidence < self.min_confidence
            or len(result.session_ids) == 0
        )
        floor_k = self.recall_floor_wide_k if low_confidence else self.recall_floor_k
        top_ids = [
            getattr(match.get("summary"), "session_id", "")
            for match, _provider in fused[:floor_k]
        ]
        selected_ids = {
            getattr(match.get("summary"), "session_id", "")
            for match, _provider in selected_fused
        }
        by_id: dict[str, tuple[dict[str, Any], Any]] = {}
        for match, provider in fused:
            session_id = getattr(match.get("summary"), "session_id", "")
            if session_id:
                by_id.setdefault(session_id, (match, provider))

        floor_ids = [session_id for session_id in top_ids if session_id and session_id not in selected_ids]
        merged = list(selected_fused)
        for session_id in floor_ids:
            entry = by_id.get(session_id)
            if entry is not None:
                merged.append(entry)
        return merged[: self.max_selected], floor_ids

    @staticmethod
    def _build_drill_query(task: str, result: SelectionResult) -> str:
        """Augment the drill query with the selector's refined query and the
        sub-questions it enumerated, so evidence extraction targets every part
        of a multi-component question (two-stage selection)."""
        parts = [task]
        if result.refined_query:
            parts.append(result.refined_query)
        for entry in result.coverage:
            subquestion = str(entry.get("subquestion") or "").strip()
            if subquestion:
                parts.append(subquestion)
        return "\n".join(dict.fromkeys(part for part in parts if part.strip()))

    def _session_texts(self, provider: Any, session_id: str) -> list[str]:
        """Return every text block in a session, bypassing the detail window.

        ``provider.get_session`` truncates to the last ``MAX_SESSION_MESSAGES``;
        the drill-down needs the whole session, so we prefer the provider's full
        message stream and fall back to the JSONL/raw source when unavailable.
        """
        texts: list[str] = []
        full = getattr(provider, "_session_messages", None)
        if callable(full):
            try:
                for message in full(session_id) or ():
                    content = str(getattr(message, "content", "") or "")
                    if content.strip():
                        texts.append(content)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Full-session read failed (%s): %s", session_id, exc)
        if texts:
            return texts
        try:
            detail = provider.get_session(session_id)
        except Exception:  # pragma: no cover - defensive
            return texts
        for message in getattr(detail, "messages", ()) or ():
            content = str(getattr(message, "content", "") or "")
            if content.strip():
                texts.append(content)
        return texts

    def _drill_evidence(
        self,
        task: str,
        selected_fused: list[tuple[dict[str, Any], Any]],
        *,
        max_lines: int,
    ) -> list[str]:
        """Lexically scan the *whole* session for the query's terms.

        This recovers exact answering lines from very large sessions whose
        retriever-matched chunks (and the truncated detail window) do not contain
        the answer. Runs only for the selected sessions and is bounded by a char
        budget so a huge session cannot stall retrieval.
        """
        if not selected_fused or max_lines <= 0:
            return []
        try:
            from .context_builder import (
                CONTEXT_LINE_WINDOW_CHARS,
                _focus_tokens,
                _normalize_whitespace,
                _tokenize,
                _window_context_line,
            )
        except Exception:  # pragma: no cover - defensive
            return []
        tokens = _tokenize(task)
        if not tokens:
            return []
        focus = _focus_tokens(task)
        # Rare/identifier-ish tokens (matQ5, file paths, error strings) are far
        # more discriminative than common words. Rank by the strongest matching
        # token first (tier), then by breadth of coverage, so a short line that
        # contains the exact identifier beats a long line of generic chatter.
        distinctive = {
            token
            for token in focus
            if (not token.isalpha()) or len(token) >= 8
        }
        budget = self.drill_char_budget
        collected: list[str] = []
        per_session = max(1, max_lines // max(1, len(selected_fused)))
        for match, provider in selected_fused:
            session_id = getattr(match.get("summary"), "session_id", "")
            if not session_id:
                continue
            scored: list[tuple[tuple[int, int, int], str]] = []
            consumed = 0
            for content in self._session_texts(provider, session_id):
                if consumed >= budget:
                    break
                consumed += len(content)
                for raw_segment in _evidence_segments(content):
                    line = _normalize_whitespace(raw_segment)
                    if len(line) < 20:
                        continue
                    lowered = line.lower()
                    distinctive_hits = sum(1 for token in distinctive if token in lowered)
                    focus_hits = sum(1 for token in focus if token in lowered)
                    hits = sum(1 for token in tokens if token in lowered)
                    if not (distinctive_hits or focus_hits or hits):
                        continue
                    windowed = _window_context_line(line, tokens, CONTEXT_LINE_WINDOW_CHARS)
                    scored.append(((distinctive_hits, focus_hits, hits), windowed))
            if not scored:
                continue
            # Greedy set-cover over distinctive tokens: prefer lines that match
            # query tokens not yet covered, so several distinct facts surface
            # instead of several paraphrases of the same one.
            scored.sort(key=lambda item: item[0], reverse=True)
            remaining = list(scored)
            covered_distinctive: set[str] = set()
            covered_focus: set[str] = set()
            taken_lines: list[str] = []

            def matched_tokens(line: str, pool: set[str]) -> set[str]:
                lowered = line.lower()
                return {token for token in pool if token in lowered}

            while remaining and len(taken_lines) < per_session:
                best_index = 0
                best_gain: tuple[int, int, tuple[int, int, int]] | None = None
                for index, (key, line) in enumerate(remaining):
                    new_distinctive = matched_tokens(line, distinctive) - covered_distinctive
                    new_focus = matched_tokens(line, focus) - covered_focus
                    gain = (len(new_distinctive), len(new_focus), key)
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_index = index
                    if new_distinctive and len(new_focus) >= 2:
                        best_index = index
                        break
                key, line = remaining.pop(best_index)
                if key[0] == 0 and key[1] == 0:
                    break
                covered_distinctive |= matched_tokens(line, distinctive)
                covered_focus |= matched_tokens(line, focus)
                if line not in collected:
                    collected.append(line)
                    taken_lines.append(line)
            if len(collected) >= max_lines:
                break
        return collected[:max_lines]

    def _run_selector(
        self,
        task: str,
    ) -> tuple[SelectionResult | None, list[tuple[dict[str, Any], Any]], int, str]:
        """Run the selector, with at most ``max_attempts - 1`` project-anchored re-queries."""
        fused = self._collect_fused(task)
        if not fused or self.selector is None:
            return None, fused, 0, task
        workspace_path = getattr(self.context_builder, "workspace_path", "") or ""
        query = task
        result: SelectionResult | None = None
        attempts = 0
        while True:
            attempts += 1
            candidates = [
                self._to_candidate(match, task=query, provider=provider)
                for match, provider in fused
            ]
            try:
                result = self.selector.select(query, candidates, workspace_path=workspace_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Session selector failed: error=%s", exc)
                return None, fused, attempts, query
            if result.degraded:
                return result, fused, attempts, query
            if attempts >= self.max_attempts or not self._should_requery(result):
                break
            refined = self._refine_query(query, result, workspace_path)
            if refined.strip().lower() == query.strip().lower():
                break
            new_fused = self._collect_fused(refined)
            if not new_fused:
                break
            fused = self._merge_fused(fused, new_fused)
            query = refined
        return result, fused, attempts, query

    def _should_requery(self, result: SelectionResult) -> bool:
        return (
            bool(result.need_more)
            or not result.session_ids
            or result.confidence < self.min_confidence
        )

    @staticmethod
    def _refine_query(
        query: str,
        result: SelectionResult,
        workspace_path: str,
    ) -> str:
        base = (result.refined_query or query).strip() or query
        project = Path(workspace_path).name if workspace_path else ""
        if project and project.lower() not in base.lower():
            base = f"{base} {project}"
        return base

    def _merge_fused(
        self,
        base: list[tuple[dict[str, Any], Any]],
        extra: list[tuple[dict[str, Any], Any]],
    ) -> list[tuple[dict[str, Any], Any]]:
        seen: set[str] = set()
        merged: list[tuple[dict[str, Any], Any]] = []
        for match, provider in [*base, *extra]:
            session_id = getattr(match.get("summary"), "session_id", "")
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            merged.append((match, provider))
        return merged[: self.max_candidates * 2]


def _make_chat_selector(
    ai: Any,
    workspace_path: str,
    model: str,
    *,
    permutations: int,
    timeout_seconds: float | None = None,
) -> Any:
    from .session_selector_llm import LLMSessionSelector

    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _env_float("DEVENV_SESSION_SELECTOR_TIMEOUT", 300.0)
    )
    core = None
    if model:
        try:
            from core.ai.routing import OpenCodeAICore

            core = OpenCodeAICore(workspace_path=workspace_path, model=model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not create selector model core (%s): %s", model, exc)
            core = None

    def _invoke(engine: Any, messages: list[dict[str, str]], schema: dict | None) -> Any:
        try:
            return (
                engine.chat(messages, output_schema=schema)
                if schema is not None
                else engine.chat(messages)
            )
        except TypeError:  # pragma: no cover - callable without schema support
            return engine.chat(messages)

    def chat(messages: list[dict[str, str]], schema: dict | None = None) -> str:
        engines = [core, ai] if core is not None else [ai]
        last_error: Exception | None = None
        for engine in engines:
            try:
                response = _run_with_timeout(
                    lambda: _invoke(engine, messages, schema), timeout
                )
            except Exception as exc:
                last_error = exc
                continue
            structured = dict(getattr(response, "metadata", {}) or {}).get("structured")
            if isinstance(structured, dict) and "selected" in structured:
                return json.dumps(structured)
            return getattr(response, "content", "") or ""
        if last_error is not None:
            raise last_error
        return ""

    return LLMSessionSelector(chat, model=model, permutations=permutations)


def _run_with_timeout(call: Any, timeout_seconds: float) -> Any:
    """Run a blocking selector call with a hard wall-clock cap.

    The selector is a serial LLM round-trip that has been observed to overrun by
    minutes on large candidate sets. On timeout we raise ``TimeoutError`` so the
    caller degrades to the recall floor instead of hanging retrieval.
    """
    if not timeout_seconds or timeout_seconds <= 0:
        return call()
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"selector call exceeded {timeout_seconds:.0f}s"
            ) from exc


def build_session_orchestrator(
    context_builder: ContextBuilderService,
    ai: Any,
    workspace_path: str,
    *,
    selector_model: str | None = None,
    permutations: int | None = None,
    shadow: bool | None = None,
    max_candidates: int = 12,
) -> SessionSelectionOrchestrator:
    """Build an orchestrator for the chosen model, honoring the env flags.

    ``DEVENV_SESSION_SELECTOR`` enables the selector; ``DEVENV_SESSION_SELECTOR_SHADOW``
    runs it in observation-only mode; ``DEVENV_SESSION_SELECTOR_MODEL`` (or the caller's
    ``selector_model``) picks the model. With no flags set this returns a disabled
    orchestrator that is a pure engine passthrough.
    """
    enabled = session_selector_enabled()
    shadow_mode = (
        os.getenv("DEVENV_SESSION_SELECTOR_SHADOW", "").strip().lower() in _ENABLED_VALUES
        if shadow is None
        else bool(shadow)
    )
    if not enabled and not shadow_mode:
        return SessionSelectionOrchestrator(
            context_builder, enabled=False, max_candidates=max_candidates
        )
    model = (
        selector_model
        or os.getenv("DEVENV_SESSION_SELECTOR_MODEL", "")
        or DEFAULT_SELECTOR_MODEL
    ).strip()
    resolved_permutations = (
        permutations
        if permutations is not None
        else _env_int("DEVENV_SESSION_SELECTOR_PERMUTATIONS", 1)
    )
    selector = _make_chat_selector(
        ai, workspace_path, model, permutations=max(1, resolved_permutations)
    )
    return SessionSelectionOrchestrator(
        context_builder,
        selector=selector,
        enabled=enabled,
        shadow=shadow_mode,
        max_candidates=max_candidates,
    )
