from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import (
    ExternalSessionDetail,
    ExternalSessionMessage,
    ExternalSessionProviderConfig,
    ExternalSessionSummary,
    ExternalSourceHealth,
    PreparedPromptRequest,
    PreparedPromptResult,
)
from .workspace import WorkspaceBrowser

logger = logging.getLogger(__name__)

MAX_SESSION_MESSAGES = 10
MAX_CONTEXT_LINES = 8
MAX_WORKSPACE_FACTS = 8
MAX_README_CHARS = 500
MIN_SESSION_CONTENT_SCORE = 6
COMMON_CONTEXT_TOKENS = {
    "about",
    "again",
    "also",
    "any",
    "been",
    "do",
    "from",
    "have",
    "into",
    "just",
    "hello",
    "help",
    "hey",
    "hi",
    "know",
    "previous",
    "project",
    "projects",
    "remember",
    "session",
    "sessions",
    "some",
    "that",
    "them",
    "they",
    "this",
    "talking",
    "there",
    "work",
    "worked",
    "what",
    "were",
    "with",
    "would",
    "you",
    "your",
    "issue",
    "issues",
    "review",
    "reviewer",
    "reviews",
    "fix",
    "fixed",
    "update",
    "get",
}

FOCUS_CONTEXT_TOKENS = COMMON_CONTEXT_TOKENS | {
    "assistant",
    "asked",
    "bug",
    "bugs",
    "context",
    "conversation",
    "exactly",
    "recent",
    "reported",
    "rollout",
    "targeted",
    "those",
    "workspace",
}


class ExternalSessionProvider(ABC):
    def __init__(self, config: ExternalSessionProviderConfig) -> None:
        self.config = config
        self.root = Path(config.root_path).expanduser()

    @property
    def name(self) -> str:
        return self.config.provider

    @abstractmethod
    def health(self) -> ExternalSourceHealth:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> list[ExternalSessionSummary]:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, session_id: str) -> ExternalSessionDetail:
        raise NotImplementedError


class CodexSessionProvider(ExternalSessionProvider):
    def __init__(self, config: ExternalSessionProviderConfig) -> None:
        super().__init__(config)
        self._detail_cache: dict[str, ExternalSessionDetail] = {}
        self._session_file_map: dict[str, Path] | None = None
        self._history_preview_cache: dict[str, str] | None = None
        self._summary_cache: dict[str, ExternalSessionSummary] = {}

    def health(self) -> ExternalSourceHealth:
        available = self.config.enabled and self.root.exists()
        sessions = self.list_sessions() if available else []
        if not self.config.enabled:
            summary = "Disabled"
        elif not self.root.exists():
            summary = "Codex archive not found"
        elif not sessions:
            summary = "No Codex sessions discovered"
        else:
            summary = f"{len(sessions)} Codex session(s) available"
        return ExternalSourceHealth(
            provider=self.name,
            enabled=self.config.enabled,
            available=available,
            root_path=str(self.root),
            summary=summary,
            session_count=len(sessions),
        )

    def list_sessions(self) -> list[ExternalSessionSummary]:
        if not self.config.enabled or not self.root.exists():
            return []

        summaries: list[ExternalSessionSummary] = []
        by_id = self._load_index_records()
        session_files = self._session_files_by_id()
        discovered_ids: set[str] = set()
        if by_id:
            for session_id, record in by_id.items():
                source_path = session_files.get(session_id)
                detail = self._detail_cache.get(session_id)
                summary = detail.summary if detail is not None else self._summary_from_index_record(session_id, record, source_path)
                summaries.append(summary)
                discovered_ids.add(session_id)
            for session_id, session_file in sorted(session_files.items(), key=lambda item: str(item[1]), reverse=True):
                if session_id in discovered_ids:
                    continue
                summary = self._detail_cache.get(session_id)
                if summary is not None:
                    summaries.append(summary.summary)
                    continue
                summaries.append(self._summary_from_session_file(session_file))
            summaries.sort(key=lambda item: item.updated_at, reverse=True)
            return summaries

        for _session_id, session_file in sorted(session_files.items(), key=lambda item: str(item[1]), reverse=True):
            session_id = _session_id_from_file(session_file)
            detail = self._detail_cache.get(session_id)
            if detail is not None:
                summaries.append(detail.summary)
                continue
            summaries.append(self._summary_from_session_file(session_file))
        return summaries

    def get_session(self, session_id: str) -> ExternalSessionDetail:
        cached = self._detail_cache.get(session_id)
        if cached is not None:
            return cached

        session_file = self._find_session_file(session_id)
        if session_file is None:
            raise FileNotFoundError(f"Unknown Codex session: {session_id}")

        detail = self._parse_session_file(session_file)
        index_record = self._load_index_records().get(session_id)
        if index_record is not None:
            summary = ExternalSessionSummary(
                provider=detail.summary.provider,
                session_id=detail.summary.session_id,
                title=str(index_record.get("thread_name") or detail.summary.title),
                updated_at=str(index_record.get("updated_at") or detail.summary.updated_at),
                workspace_path=detail.summary.workspace_path,
                source_path=detail.summary.source_path,
                message_count=detail.summary.message_count,
                preview=detail.summary.preview,
            )
            detail = ExternalSessionDetail(summary=summary, messages=detail.messages, metadata=detail.metadata)

        self._detail_cache[session_id] = detail
        return detail

    def _load_index_records(self) -> dict[str, dict[str, Any]]:
        index_path = self.root / (self.config.index_path or "session_index.jsonl")
        records: dict[str, dict[str, Any]] = {}
        if not index_path.exists():
            return records

        for raw_line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            session_id = str(payload.get("id") or "").strip()
            if session_id:
                records[session_id] = payload
        return records

    def _find_session_file(self, session_id: str) -> Path | None:
        return self._session_files_by_id().get(session_id)

    def _session_files_by_id(self) -> dict[str, Path]:
        if self._session_file_map is not None:
            return self._session_file_map
        mapping: dict[str, Path] = {}
        for path in self.root.glob(self.config.session_glob):
            session_id = _session_id_from_file(path)
            if session_id and session_id not in mapping:
                mapping[session_id] = path
        self._session_file_map = mapping
        return mapping

    def _preview_from_history(self, session_id: str) -> str:
        if self._history_preview_cache is None:
            self._history_preview_cache = {}
        history_path = self.root / "history.jsonl"
        if not history_path.exists():
            return ""
        cached = self._history_preview_cache.get(session_id)
        if cached is not None:
            return cached
        for raw_line in reversed(history_path.read_text(encoding="utf-8").splitlines()):
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("session_id")) != session_id:
                continue
            text = _normalize_whitespace(str(payload.get("text") or ""))
            if text and not _is_noise_message_content(text):
                preview = text[:220]
                self._history_preview_cache[session_id] = preview
                return preview
        self._history_preview_cache[session_id] = ""
        return ""

    def _summary_from_index_record(self, session_id: str, record: dict[str, Any], source_path: Path | None) -> ExternalSessionSummary:
        cached = self._summary_cache.get(session_id)
        if cached is not None:
            return cached
        source_summary = self._summary_from_session_file(source_path) if source_path is not None else None
        summary = ExternalSessionSummary(
            provider=self.name,
            session_id=session_id,
            title=str(record.get("thread_name") or (source_summary.title if source_summary else "Untitled session")),
            updated_at=str(record.get("updated_at") or (source_summary.updated_at if source_summary else "")),
            workspace_path=source_summary.workspace_path if source_summary is not None else None,
            source_path=str(source_path) if source_path else None,
            message_count=source_summary.message_count if source_summary is not None else 0,
            preview=(source_summary.preview if source_summary is not None else "") or self._preview_from_history(session_id),
        )
        self._summary_cache[session_id] = summary
        return summary

    def _summary_from_session_file(self, session_file: Path) -> ExternalSessionSummary:
        session_id = _session_id_from_file(session_file)
        cached = self._summary_cache.get(session_id)
        if cached is not None:
            return cached

        title = session_file.stem
        updated_at = _timestamp_from_path(session_file)
        workspace_path = None
        for raw_line in session_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "session_meta":
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if payload:
                session_id = str(payload.get("session_id") or payload.get("id") or session_id)
                title = str(payload.get("title") or payload.get("thread_name") or title)
                updated_at = str(payload.get("timestamp") or row.get("timestamp") or updated_at)
                workspace_path = payload.get("cwd") or workspace_path
            break

        summary = ExternalSessionSummary(
            provider=self.name,
            session_id=session_id,
            title=_normalize_whitespace(title) or "Untitled session",
            updated_at=updated_at,
            workspace_path=workspace_path,
            source_path=str(session_file),
            message_count=0,
            preview=self._preview_from_history(session_id),
        )
        self._summary_cache[session_id] = summary
        return summary

    def _parse_session_file(self, session_file: Path) -> ExternalSessionDetail:
        session_id = _session_id_from_file(session_file)
        title = session_file.stem
        updated_at = ""
        workspace_path = None
        messages: list[ExternalSessionMessage] = []
        preview = ""
        metadata: dict[str, Any] = {
            "provider": self.name,
            "source_path": str(session_file),
        }
        seen_signatures: set[tuple[str, str]] = set()

        history_fallback = self._history_messages(session_id)

        for raw_line in session_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            timestamp = str(row.get("timestamp") or "")
            row_type = row.get("type")
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}

            if row_type == "session_meta":
                inner = payload or {}
                session_id = str(inner.get("session_id") or inner.get("id") or session_id)
                title = str(inner.get("title") or inner.get("thread_name") or title)
                updated_at = str(inner.get("timestamp") or timestamp or updated_at)
                workspace_path = inner.get("cwd") or workspace_path
                metadata.update(
                    {
                        "source": inner.get("source"),
                        "originator": inner.get("originator"),
                        "model_provider": inner.get("model_provider"),
                    }
                )
                continue

            extracted = _extract_session_messages(row_type=row_type, payload=payload, timestamp=timestamp)
            for message in extracted:
                signature = (message.role, message.content)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                messages.append(message)
                if not preview and message.role == "user":
                    preview = message.content[:220]

        if history_fallback:
            existing_users = {message.content for message in messages if message.role == "user"}
            for message in history_fallback:
                if message.content not in existing_users:
                    messages.insert(0, message)

        messages = [message for message in messages if message.content.strip()]
        messages = messages[:MAX_SESSION_MESSAGES]
        if not preview:
            preview = next((message.content[:220] for message in messages if message.content.strip()), "")
        if not updated_at:
            updated_at = _timestamp_from_path(session_file)

        summary = ExternalSessionSummary(
            provider=self.name,
            session_id=session_id,
            title=_normalize_whitespace(title) or "Untitled session",
            updated_at=updated_at,
            workspace_path=workspace_path,
            source_path=str(session_file),
            message_count=len(messages),
            preview=preview,
        )
        detail = ExternalSessionDetail(summary=summary, messages=tuple(messages), metadata=metadata)
        self._detail_cache[session_id] = detail
        return detail

    def _history_messages(self, session_id: str) -> list[ExternalSessionMessage]:
        history_path = self.root / "history.jsonl"
        if not history_path.exists():
            return []

        messages: list[ExternalSessionMessage] = []
        for raw_line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("session_id")) != session_id:
                continue
            text = _normalize_whitespace(str(payload.get("text") or ""))
            if not text or _is_noise_message_content(text):
                continue
            messages.append(
                ExternalSessionMessage(
                    role="user",
                    content=text,
                    timestamp=_string_timestamp(payload.get("ts")),
                )
            )
        return messages[-3:]


class OpenCodeSessionProvider(ExternalSessionProvider):
    def health(self) -> ExternalSourceHealth:
        available = self.config.enabled and self.root.exists()
        summary = "OpenCode provider ready" if available else "OpenCode archive not configured"
        return ExternalSourceHealth(
            provider=self.name,
            enabled=self.config.enabled,
            available=available,
            root_path=str(self.root),
            summary=summary,
            session_count=0,
        )

    def list_sessions(self) -> list[ExternalSessionSummary]:
        return []

    def get_session(self, session_id: str) -> ExternalSessionDetail:
        raise FileNotFoundError(f"OpenCode session archive is not configured for session: {session_id}")


class ContextBuilderService:
    def __init__(
        self,
        workspace_path: str,
        *,
        memory: Any | None = None,
        provider_configs: tuple[ExternalSessionProviderConfig, ...] = (),
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.workspace = WorkspaceBrowser(self.workspace_path)
        self.memory = memory
        self.provider_configs = provider_configs or _default_provider_configs()
        self.runtime_allowed_providers: set[str] | None = None
        self.providers = {
            config.provider: _provider_from_config(config)
            for config in self.provider_configs
        }

    def set_runtime_allowed_providers(self, providers: set[str] | list[str] | tuple[str, ...] | None) -> None:
        if providers is None:
            self.runtime_allowed_providers = None
            return
        self.runtime_allowed_providers = {provider for provider in providers if provider in self.providers}

    def list_sources(self) -> list[ExternalSourceHealth]:
        return [provider.health() for provider in self.providers.values()]

    def list_sessions(self, provider_name: str) -> list[ExternalSessionSummary]:
        provider = self._get_provider(provider_name)
        return provider.list_sessions()

    def get_session(self, provider_name: str, session_id: str) -> ExternalSessionDetail:
        provider = self._get_provider(provider_name)
        return provider.get_session(session_id)

    def prepare_prompt(self, request: PreparedPromptRequest) -> PreparedPromptResult:
        provider_name = request.provider or self._default_provider_name()
        details: list[ExternalSessionDetail] = []
        selected_session_ids: tuple[str, ...] = request.session_ids
        selection_metadata: dict[str, Any] = {
            "context_match_state": "new_context",
            "context_match_reason": "No strong prior-session match was found.",
        }
        if request.include_prior_context and provider_name:
            provider = self._get_provider(provider_name)
            if request.session_ids:
                selected_session_ids = request.session_ids
                selection_metadata = {
                    "context_match_state": "reused_prior_context" if selected_session_ids else "new_context",
                    "context_match_reason": "Prior-session context was selected manually." if selected_session_ids else "No prior session was selected manually.",
                }
            else:
                selected_matches = self._select_relevant_sessions(provider, request.task)
                selected_session_ids = tuple(match["summary"].session_id for match in selected_matches)
                selection_metadata = self._selection_metadata(selected_matches)
            details = [provider.get_session(session_id) for session_id in selected_session_ids]

        context_lines = _collect_relevant_context_lines(request.task, details, request.output_format)
        workspace_facts = self._workspace_facts(request.task) if request.include_workspace_scan else ()
        memory_facts = self._memory_facts(request.task) if request.include_prior_context else ()
        constraints = _infer_constraints(request.task)

        prompt = _assemble_prompt(
            task=request.task,
            context_lines=context_lines,
            workspace_facts=workspace_facts,
            memory_facts=memory_facts,
            constraints=constraints,
            output_format=request.output_format,
        )
        return PreparedPromptResult(
            prompt=prompt,
            provider=provider_name,
            session_ids=selected_session_ids,
            workspace_facts=workspace_facts,
            prior_context=context_lines,
            constraints=constraints,
            metadata={
                "workspace_path": self.workspace_path,
                "selected_session_count": len(selected_session_ids),
                "selection_mode": "manual" if request.session_ids else "automatic",
                "output_format": request.output_format,
                **selection_metadata,
            },
        )

    def build_runtime_memory_context(
        self,
        task: str,
        *,
        provider_name: str | None = None,
        max_lines: int = 6,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        resolved_provider = provider_name or self._default_provider_name()
        if not resolved_provider:
            return "", (), {"context_match_state": "new_context", "context_match_reason": "No external session provider is available."}
        if self.runtime_allowed_providers is not None and resolved_provider not in self.runtime_allowed_providers:
            return "", (), {"context_match_state": "new_context", "context_match_reason": f"External {resolved_provider} access has not been granted."}
        if self.runtime_allowed_providers == set():
            return "", (), {"context_match_state": "new_context", "context_match_reason": "External session access has not been granted."}
        provider = self._get_provider(resolved_provider)
        selected_matches = self._select_relevant_sessions(provider, task)
        selected_session_ids = tuple(match["summary"].session_id for match in selected_matches)
        selection_metadata = self._selection_metadata(selected_matches)
        if not selected_session_ids:
            return "", (), selection_metadata
        details = [provider.get_session(session_id) for session_id in selected_session_ids]
        context_lines = _collect_relevant_context_lines(task, details, "detailed")[:max_lines]
        if not context_lines:
            return "", selected_session_ids, selection_metadata
        lines = ["## External Session Context", *(f"- {line}" for line in context_lines)]
        return "\n".join(lines), selected_session_ids, selection_metadata

    def _workspace_facts(self, task: str) -> tuple[str, ...]:
        facts: list[str] = []
        try:
            entries = self.workspace.list_entries("")
        except Exception as exc:
            logger.warning("Context builder workspace scan failed: error=%s", exc)
            return ()

        directories = [entry.name for entry in entries if entry.is_dir][:5]
        files = [entry.name for entry in entries if not entry.is_dir][:5]
        if directories:
            facts.append(f"Top-level directories: {', '.join(directories)}.")
        if files:
            facts.append(f"Top-level files: {', '.join(files)}.")

        readme_path = next((entry.path for entry in entries if entry.name.lower().startswith("readme")), None)
        if readme_path:
            try:
                readme = self.workspace.read_text_file(readme_path)
            except Exception:
                readme = ""
            if readme.strip():
                snippet = _normalize_whitespace(readme)[:MAX_README_CHARS]
                facts.append(f"README summary: {snippet}")

        prompt_tokens = _tokenize(task)
        for entry in entries:
            if entry.is_dir:
                continue
            lowered = entry.name.lower()
            if prompt_tokens and any(token in lowered for token in prompt_tokens):
                facts.append(f"Potentially relevant file: {entry.path}")
            if len(facts) >= MAX_WORKSPACE_FACTS:
                break
        return tuple(facts[:MAX_WORKSPACE_FACTS])

    def _memory_facts(self, task: str) -> tuple[str, ...]:
        if self.memory is None or not hasattr(self.memory, "retrieve_context"):
            return ()
        try:
            result = self.memory.retrieve_context(task)
        except Exception as exc:
            logger.warning("Context builder memory retrieval failed: error=%s", exc)
            return ()

        facts: list[str] = []
        for line in str(getattr(result, "markdown_context", "")).splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                facts.append(stripped[2:].strip())
            if len(facts) >= 4:
                break
        return tuple(facts)

    def _get_provider(self, provider_name: str) -> ExternalSessionProvider:
        provider = self.providers.get(provider_name)
        if provider is None:
            raise FileNotFoundError(f"Unknown context provider: {provider_name}")
        return provider

    def _default_provider_name(self) -> str | None:
        for provider_name, provider in self.providers.items():
            if provider.health().available:
                return provider_name
        return next(iter(self.providers), None)

    def _select_relevant_session_ids(self, provider: ExternalSessionProvider, task: str) -> tuple[str, ...]:
        return tuple(match["summary"].session_id for match in self._select_relevant_sessions(provider, task))

    def _select_relevant_sessions(self, provider: ExternalSessionProvider, task: str) -> list[dict[str, Any]]:
        summaries = provider.list_sessions()
        if not summaries:
            return []

        prompt_tokens = _tokenize(task)
        if not prompt_tokens:
            return []
        focus_tokens = _focus_tokens(task)
        preliminary: list[dict[str, Any]] = []
        recent_window = 12
        for index, summary in enumerate(summaries):
            identity_haystacks = _session_identity_haystacks(summary)
            summary_haystacks = identity_haystacks + [summary.preview.lower()]
            summary_token_hits = sum(1 for token in prompt_tokens if any(_token_matches(token, haystack) for haystack in summary_haystacks))
            summary_exact_hits = _exact_prompt_hits(prompt_tokens, summary_haystacks)
            identity_token_hits = sum(1 for token in prompt_tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
            identity_exact_hits = _exact_prompt_hits(prompt_tokens, identity_haystacks)
            identity_focus_hits = sum(1 for token in focus_tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
            issue_bonus = 0
            if any(token in prompt_tokens for token in {"bug", "bugs", "fix", "fixed", "review", "reviews"}):
                issue_terms = ("bug", "bugs", "fix", "fixed", "review", "reviews")
                if any(term in summary.title.lower() for term in issue_terms):
                    issue_bonus += 6
                elif any(term in summary.preview.lower() for term in issue_terms):
                    issue_bonus += 3
            summary_score = (
                (identity_exact_hits * 14)
                + (identity_token_hits * 8)
                + (summary_exact_hits * 4)
                + (summary_token_hits * 2)
                + issue_bonus
            )
            preliminary.append(
                {
                    "summary": summary,
                    "summary_score": summary_score,
                    "summary_token_hits": summary_token_hits,
                    "identity_focus_hits": identity_focus_hits,
                    "recent_rank": index,
                }
            )

        preliminary.sort(key=lambda item: (item["summary_score"], -item["recent_rank"], item["summary"].updated_at), reverse=True)
        candidate_ids: list[str] = []
        for item in preliminary[:12]:
            session_id = item["summary"].session_id
            if session_id not in candidate_ids:
                candidate_ids.append(session_id)
        for item in preliminary:
            if item["summary_score"] <= 0 and item["recent_rank"] >= recent_window:
                continue
            session_id = item["summary"].session_id
            if session_id not in candidate_ids:
                candidate_ids.append(session_id)
            if len(candidate_ids) >= max(recent_window, 12):
                break

        scored: list[dict[str, Any]] = []
        workspace_name = Path(self.workspace_path).name.lower()
        workspace_path = self.workspace_path.lower()

        for session_id in candidate_ids:
            summary = next((item["summary"] for item in preliminary if item["summary"].session_id == session_id), None)
            if summary is None:
                continue
            detail = provider.get_session(summary.session_id)
            identity_haystacks = _session_identity_haystacks(summary)
            haystacks = _session_haystacks(summary, detail)
            token_hits = sum(1 for token in prompt_tokens if any(_token_matches(token, haystack) for haystack in haystacks))
            exact_hits = _exact_prompt_hits(prompt_tokens, haystacks)
            identity_token_hits = sum(1 for token in prompt_tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
            identity_exact_hits = _exact_prompt_hits(prompt_tokens, identity_haystacks)
            identity_focus_hits = sum(1 for token in focus_tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
            best_overlap = _best_message_overlap(prompt_tokens, detail)
            workspace_bonus = 0
            issue_bonus = 0
            session_workspace = (summary.workspace_path or "").lower()
            if session_workspace:
                if session_workspace == workspace_path:
                    workspace_bonus += 5 if identity_focus_hits > 0 or identity_exact_hits > 0 else -12
                elif workspace_name and workspace_name in session_workspace:
                    workspace_bonus += 3 if identity_focus_hits > 0 or identity_exact_hits > 0 else -6
                else:
                    workspace_bonus += 1
            if focus_tokens and session_workspace == workspace_path and identity_focus_hits == 0:
                continue
            if any(token in prompt_tokens for token in {"bug", "bugs", "fix", "fixed", "review", "reviews"}):
                issue_terms = ("bug", "bugs", "fix", "fixed", "review", "reviews")
                if any(term in summary.title.lower() for term in issue_terms):
                    issue_bonus += 10
                elif any(term in haystack for haystack in haystacks for term in issue_terms):
                    issue_bonus += 4

            content_score = (
                (identity_exact_hits * 14)
                + (identity_token_hits * 8)
                + (exact_hits * 3)
                + (token_hits * 2)
                + min(best_overlap * 2, 8)
            )
            strong_match = identity_exact_hits >= 1 or identity_token_hits >= 1 or exact_hits >= 1 or best_overlap >= 2 or token_hits >= 2
            scored.append(
                {
                    "summary": summary,
                    "detail": detail,
                    "content_score": content_score,
                    "score": content_score + workspace_bonus + issue_bonus,
                    "strong_match": strong_match,
                    "exact_hits": exact_hits,
                    "token_hits": token_hits,
                    "identity_exact_hits": identity_exact_hits,
                    "identity_token_hits": identity_token_hits,
                    "identity_focus_hits": identity_focus_hits,
                    "best_overlap": best_overlap,
                }
            )

        scored.sort(key=lambda item: (item["strong_match"], item["score"], item["summary"].updated_at), reverse=True)
        selected = [
            item
            for item in scored
            if item["strong_match"] and item["content_score"] >= MIN_SESSION_CONTENT_SCORE and item["score"] > 0
        ]
        if focus_tokens and any(item["identity_focus_hits"] > 0 for item in selected):
            selected = [item for item in selected if item["identity_focus_hits"] > 0]
        return selected[:3]

    def _selection_metadata(self, selected_matches: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected_matches:
            return {
                "context_match_state": "new_context",
                "context_match_reason": "No strong prior-session match was found.",
                "context_match_score": 0,
            }
        best = selected_matches[0]
        return {
            "context_match_state": "reused_prior_context",
            "context_match_reason": (
                f"Matched {len(selected_matches)} prior session(s) using project identity and token overlap "
                f"({best['identity_token_hits']} identity hits, {best['token_hits']} token hits)."
            ),
            "context_match_score": best["score"],
        }


def _provider_from_config(config: ExternalSessionProviderConfig) -> ExternalSessionProvider:
    if config.provider == "codex":
        return CodexSessionProvider(config)
    if config.provider == "opencode":
        return OpenCodeSessionProvider(config)
    return OpenCodeSessionProvider(config)


def _default_provider_configs() -> tuple[ExternalSessionProviderConfig, ...]:
    return (
        ExternalSessionProviderConfig(
            provider="codex",
            root_path=str(Path.home() / ".codex"),
            index_path="session_index.jsonl",
        ),
        ExternalSessionProviderConfig(
            provider="opencode",
            root_path=str(Path.home() / ".opencode"),
        ),
    )


def _extract_session_messages(*, row_type: str | None, payload: dict[str, Any], timestamp: str) -> list[ExternalSessionMessage]:
    messages: list[ExternalSessionMessage] = []
    if row_type == "event_msg" and payload.get("type") == "agent_message":
        text = _normalize_whitespace(str(payload.get("message") or ""))
        if text and not _is_noise_message_content(text):
            messages.append(ExternalSessionMessage(role="assistant", content=text, timestamp=timestamp))
    elif row_type == "event_msg" and payload.get("type") == "user_message":
        text = _normalize_whitespace(str(payload.get("message") or ""))
        if text and not _is_noise_message_content(text):
            messages.append(ExternalSessionMessage(role="user", content=text, timestamp=timestamp))
    elif row_type == "event_msg" and payload.get("type") == "task_complete":
        text = _normalize_whitespace(str(payload.get("last_agent_message") or ""))
        if text and not _is_noise_message_content(text):
            messages.append(ExternalSessionMessage(role="assistant", content=text, timestamp=timestamp))
    elif row_type == "response_item" and payload.get("type") == "message":
        role = str(payload.get("role") or "assistant")
        if role not in {"user", "assistant"}:
            return messages
        for item in payload.get("content") or ():
            if not isinstance(item, dict):
                continue
            text = _normalize_whitespace(str(item.get("text") or ""))
            if text and not _is_noise_message_content(text):
                messages.append(ExternalSessionMessage(role=role, content=text, timestamp=timestamp))
    elif row_type == "response_item" and payload.get("type") == "custom_tool_call":
        tool_name = str(payload.get("name") or "").strip()
        if tool_name:
            messages.append(
                ExternalSessionMessage(
                    role="tool",
                    content=f"Applied tool call: {tool_name}",
                    timestamp=timestamp,
                )
            )
    elif row_type == "response_item" and payload.get("type") == "function_call_output":
        output_text = _normalize_whitespace(_clean_tool_output(str(payload.get("output") or "")))
        if output_text and not _is_noise_message_content(output_text):
            messages.append(
                ExternalSessionMessage(
                    role="tool",
                    content=_truncate_tool_output(output_text),
                    timestamp=timestamp,
                )
            )
    return messages


def _collect_relevant_context_lines(
    task: str,
    details: list[ExternalSessionDetail],
    output_format: str,
) -> tuple[str, ...]:
    candidates: list[str] = []
    content_candidates: list[str] = []
    seen: set[str] = set()
    for detail in details:
        summary = detail.summary
        if summary.title.strip():
            candidates.append(f"Session '{summary.title}' targeted workspace {summary.workspace_path or 'unknown workspace'}.")
        for message in detail.messages:
            if message.role not in {"user", "assistant"}:
                continue
            content = _compact_context_content(message.role, message.content)
            if not content or content in seen:
                continue
            seen.add(content)
            if message.role == "user":
                prefix = "User asked:"
            else:
                prefix = "Assistant reported:"
            content_candidates.append(f"{prefix} {content}")
    candidates.extend(content_candidates)

    prompt_tokens = _tokenize(task)
    prompt_entities = {
        token.lower()
        for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", task.lower())
        if len(token) >= 3
    }
    lowered_task = task.lower()
    scored: list[tuple[int, str]] = []
    for line in candidates:
        lowered = line.lower()
        overlap = sum(2 for token in prompt_tokens if _token_matches(token, lowered))
        entity_overlap = any(entity in lowered for entity in prompt_entities)
        if line.startswith("User asked:"):
            overlap += 3 if any(token in lowered_task for token in ("bug", "bugs", "review", "reviews", "fix", "fixed")) else 0
        if line.startswith("Assistant reported:"):
            overlap += 3 if any(token in lowered_task for token in ("what was it about", "what was that about", "remember about")) else 1
            if prompt_entities and not entity_overlap and not any(marker in lowered for marker in ("bug", "review", "fix", "workspace", "pipeline chat", "salesforce", "create workspace")):
                overlap -= 5
        if "bug" in lowered or "review" in lowered or "fix" in lowered:
            overlap += 3
        if "i’m grounding" in lowered or "i'm grounding" in lowered:
            overlap -= 6
        if "i’m tracing" in lowered or "i'm tracing" in lowered:
            overlap -= 5
        if "i’m checking" in lowered or "i'm checking" in lowered:
            overlap -= 4
        if "i’m going to" in lowered or "i'm going to" in lowered:
            overlap -= 4
        if "i’ve confirmed this is" in lowered or "i've confirmed this is" in lowered:
            overlap -= 4
        if "i’ve already found concrete anchors" in lowered or "i've already found concrete anchors" in lowered:
            overlap -= 5
        if "decision-complete" in lowered:
            overlap -= 4
        if "this matches the app route" in lowered:
            overlap -= 5
        if "context from my ide setup" in lowered or "## active file:" in lowered:
            overlap -= 5
        if line.startswith("Session '"):
            overlap -= 1
        overlap += 1 if "workspace" in lowered else 0
        overlap += 1 if "file" in lowered or "frontend" in lowered or "backend" in lowered else 0
        scored.append((overlap, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [line for score, line in scored if score > 0][:MAX_CONTEXT_LINES]
    if not selected:
        selected = [line for _score, line in scored[:MAX_CONTEXT_LINES]]
    if any(not line.startswith("Session '") for line in selected):
        session_summaries = [line for line in selected if line.startswith("Session '")]
        detail_lines = [line for line in selected if not line.startswith("Session '")]
        selected = detail_lines + session_summaries[:1]
    if output_format == "detailed":
        return tuple(selected[:MAX_CONTEXT_LINES])
    return tuple(selected[:5])


def _assemble_prompt(
    *,
    task: str,
    context_lines: tuple[str, ...],
    workspace_facts: tuple[str, ...],
    memory_facts: tuple[str, ...],
    constraints: tuple[str, ...],
    output_format: str,
) -> str:
    sections = ["Task:", task.strip()]
    if context_lines:
        sections.extend(["", "Relevant prior session context:"])
        sections.extend(f"- {line}" for line in context_lines)
    if workspace_facts:
        sections.extend(["", "Workspace context:"])
        sections.extend(f"- {line}" for line in workspace_facts)
    if memory_facts:
        sections.extend(["", "Additional Devenv memory/context:"])
        sections.extend(f"- {line}" for line in memory_facts)
    if constraints:
        sections.extend(["", "Constraints:"])
        sections.extend(f"- {line}" for line in constraints)
    sections.extend(
        [
            "",
            "Execution instruction:",
            "Use the context above to act on the current task directly. Do not spend time rediscovering facts already supplied here. Keep changes scoped and preserve existing style unless the task explicitly asks otherwise.",
        ]
    )
    if output_format == "detailed":
        sections.extend(
            [
                "",
                "Verification expectation:",
                "Explain what you changed and verify the result with the smallest relevant checks available in the repo.",
            ]
        )
    return "\n".join(section for section in sections if section is not None).strip()


def _infer_constraints(task: str) -> tuple[str, ...]:
    lowered = task.lower()
    rules: list[str] = []
    if "minimal change" in lowered or "minimal changes" in lowered:
        rules.append("Make minimal changes.")
    if "don't change" in lowered or "dont change" in lowered:
        rules.append("Do not change unrelated logic.")
    if "preserve style" in lowered or "same style" in lowered:
        rules.append("Preserve the existing coding style and patterns.")
    if "verify" in lowered or "test" in lowered:
        rules.append("Run or describe the relevant verification steps.")
    if not rules:
        rules.append("Keep the implementation focused on the current task.")
    return tuple(rules)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", normalized)
        if len(token) >= 3 and token not in COMMON_CONTEXT_TOKENS
    }
    compound_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", normalized)
        if token not in COMMON_CONTEXT_TOKENS
    }
    return tokens | compound_tokens


def _is_noise_message_content(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("<environment_context>") or lowered.startswith("<permissions instructions>") or lowered.startswith("<collaboration_mode>") or lowered.startswith("<skills_instructions>")


def _token_matches(token: str, haystack: str) -> bool:
    if _contains_whole_token(token, haystack):
        return True
    if token.endswith("ers") and token[:-3] and _contains_whole_token(token[:-3], haystack):
        return True
    if token.endswith("er") and token[:-2] and _contains_whole_token(token[:-2], haystack):
        return True
    return False


def _contains_whole_token(token: str, haystack: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", haystack))


def _session_haystacks(summary: ExternalSessionSummary, detail: ExternalSessionDetail) -> list[str]:
    haystacks = _session_identity_haystacks(summary)
    haystacks.append(summary.preview.lower())
    haystacks.extend(_normalize_whitespace(message.content).lower() for message in detail.messages[:MAX_SESSION_MESSAGES])
    return haystacks


def _session_identity_haystacks(summary: ExternalSessionSummary) -> list[str]:
    haystacks = [
        summary.title.lower(),
        (summary.workspace_path or "").lower(),
        os.path.basename(summary.source_path or "").lower(),
    ]
    return haystacks


def _focus_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in _tokenize(text)
        if len(token) >= 5 and token not in FOCUS_CONTEXT_TOKENS
    }
    path_tokens = {
        Path(match).name.lower()
        for match in re.findall(r"/[A-Za-z0-9._/-]+", text)
        if Path(match).name and Path(match).name.lower() not in FOCUS_CONTEXT_TOKENS
    }
    return tokens | path_tokens


def _best_message_overlap(prompt_tokens: set[str], detail: ExternalSessionDetail) -> int:
    best = 0
    for message in detail.messages:
        lowered = _normalize_whitespace(message.content).lower()
        overlap = sum(1 for token in prompt_tokens if _token_matches(token, lowered))
        if overlap > best:
            best = overlap
    return best


def _exact_prompt_hits(prompt_tokens: set[str], haystacks: list[str]) -> int:
    exact_hits = 0
    for token in prompt_tokens:
        pattern = re.compile(rf"\b{re.escape(token)}\b")
        if any(pattern.search(haystack) for haystack in haystacks):
            exact_hits += 1
    return exact_hits


def _truncate_tool_output(text: str, max_chars: int = 900) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _compact_context_content(role: str, text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""
    max_chars = 480 if role == "tool" else 260
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _clean_tool_output(text: str) -> str:
    raw = text or ""
    if "Output:\n" in raw:
        raw = raw.split("Output:\n", 1)[1]
    elif "Output:" in raw:
        raw = raw.split("Output:", 1)[1]

    filtered_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if lowered.startswith("chunk id:"):
            continue
        if lowered.startswith("wall time:"):
            continue
        if lowered.startswith("process exited"):
            continue
        if lowered.startswith("original token count:"):
            continue
        if lowered.startswith("warning: truncated"):
            continue
        if lowered.startswith("total output lines:"):
            continue
        if "operation not permitted: ps" in lowered:
            continue
        filtered_lines.append(stripped)
    return "\n".join(filtered_lines).strip()


def _timestamp_from_path(path: Path) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})", path.name)
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"


def _string_timestamp(value: Any) -> str | None:
    try:
        if value is None:
            return None
        integer = int(value)
        return str(integer)
    except (TypeError, ValueError):
        return None


def _session_id_from_file(path: Path) -> str:
    match = re.search(r"T\d{2}-\d{2}-\d{2}-(.+)$", path.stem)
    if match:
        return match.group(1)
    return path.stem
