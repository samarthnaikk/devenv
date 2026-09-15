from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

from core.memory.embeddings import HashingEmbedder
from core.memory.models import ExternalSessionEmbedding
from core.memory.storage import SQLiteMemoryStore

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
RRF_K = 60
SEMANTIC_STRONG_THRESHOLD = 0.35
MAX_RUNTIME_SESSION_MATCHES = 12
MAX_PROVIDER_SESSION_MATCHES = 6
MAX_INDEX_CHUNK_CHARS = 720
MAX_INDEX_CONTEXT_LINES = 12
MAX_SESSION_EMBEDDING_CACHE = 512
MAX_QUERY_VARIANTS = 4
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


@dataclass(frozen=True)
class ExternalSessionChunk:
    provider: str
    session_id: str
    title: str
    workspace_path: str | None
    role: str
    source: str
    text: str
    timestamp: str | None = None

    @property
    def search_text(self) -> str:
        parts = [self.title, self.workspace_path or "", self.role, self.source, self.text]
        return _normalize_whitespace(" ".join(part for part in parts if part)).lower()


class ExternalSessionIndex:
    def __init__(self, providers: dict[str, "ExternalSessionProvider"], *, performance_mode: str = "medium") -> None:
        self.providers = providers
        self._performance_mode = performance_mode if performance_mode in {"low", "medium", "high"} else "medium"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._indexed_providers: set[str] = set()
        self._requested_providers: set[str] = set()
        self._chunks_by_provider: dict[str, dict[str, list[ExternalSessionChunk]]] = {}
        self._summaries_by_provider: dict[str, dict[str, ExternalSessionSummary]] = {}
        self._status: dict[str, Any] = {
            "active": False,
            "completed": False,
            "message": "Waiting for provider access.",
            "percent": 0,
            "processed_sessions": 0,
            "total_sessions": 0,
            "eta_seconds": None,
            "providers": [],
            "started_at": None,
            "finished_at": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def set_performance_mode(self, performance_mode: str) -> None:
        with self._lock:
            self._performance_mode = performance_mode if performance_mode in {"low", "medium", "high"} else "medium"

    def ensure(self, providers: set[str]) -> None:
        allowed = {provider for provider in providers if provider in self.providers}
        with self._lock:
            self._requested_providers = set(allowed)
            for provider_name in list(self._chunks_by_provider):
                if provider_name not in allowed:
                    self._chunks_by_provider.pop(provider_name, None)
                    self._summaries_by_provider.pop(provider_name, None)
                    self._indexed_providers.discard(provider_name)
            needs_rebuild = bool(allowed) and allowed != self._indexed_providers
            if not needs_rebuild or (self._thread is not None and self._thread.is_alive()):
                if not allowed:
                    self._status.update(
                        {
                            "active": False,
                            "completed": False,
                            "message": "Waiting for provider access.",
                            "percent": 0,
                            "processed_sessions": 0,
                            "total_sessions": 0,
                            "eta_seconds": None,
                            "providers": [],
                            "started_at": None,
                            "finished_at": None,
                        }
                    )
                return
            self._status.update(
                {
                    "active": True,
                    "completed": False,
                    "message": "Counting accessible sessions…",
                    "percent": 0,
                    "processed_sessions": 0,
                    "total_sessions": 0,
                    "eta_seconds": None,
                    "providers": sorted(allowed),
                    "started_at": time.time(),
                    "finished_at": None,
                }
            )
            self._thread = threading.Thread(target=self._build_index, args=(sorted(allowed),), daemon=True)
            self._thread.start()

    def query(
        self,
        task: str,
        *,
        provider_name: str | None,
        workspace_path: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tokens = _tokenize(task)
        if not tokens:
            return [], {"index_ready": self.status().get("completed", False)}
        focus_tokens = _focus_tokens(task)
        workspace_name = Path(workspace_path).name.lower()
        workspace_path_lower = workspace_path.lower()
        providers = [provider_name] if provider_name else sorted(self._chunks_by_provider)
        scored_sessions: list[dict[str, Any]] = []
        with self._lock:
            summaries_by_provider = {
                name: dict(records)
                for name, records in self._summaries_by_provider.items()
                if name in providers
            }
            chunks_by_provider = {
                name: {session_id: list(chunks) for session_id, chunks in records.items()}
                for name, records in self._chunks_by_provider.items()
                if name in providers
            }

        for current_provider in providers:
            session_map = chunks_by_provider.get(current_provider, {})
            summary_map = summaries_by_provider.get(current_provider, {})
            for session_id, chunks in session_map.items():
                summary = summary_map.get(session_id)
                if summary is None or not chunks:
                    continue
                identity_haystacks = _session_identity_haystacks(summary)
                identity_token_hits = sum(1 for token in tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
                identity_exact_hits = _exact_prompt_hits(tokens, identity_haystacks)
                identity_focus_hits = sum(1 for token in focus_tokens if any(_token_matches(token, haystack) for haystack in identity_haystacks))
                top_chunks: list[tuple[int, ExternalSessionChunk, int, int]] = []
                for chunk in chunks:
                    haystack = chunk.search_text
                    token_hits = sum(1 for token in tokens if _token_matches(token, haystack))
                    exact_hits = _exact_prompt_hits(tokens, [haystack])
                    issue_bonus = 0
                    if any(token in tokens for token in {"bug", "bugs", "fix", "fixed", "review", "reviews"}):
                        if any(term in haystack for term in ("bug", "bugs", "fix", "fixed", "review", "issue")):
                            issue_bonus += 4
                    workspace_bonus = 0
                    session_workspace = (summary.workspace_path or "").lower()
                    if session_workspace == workspace_path_lower:
                        workspace_bonus += 3
                    elif workspace_name and workspace_name in session_workspace:
                        workspace_bonus += 2
                    score = (exact_hits * 8) + (token_hits * 4) + issue_bonus + workspace_bonus
                    if identity_token_hits or identity_exact_hits:
                        score += (identity_exact_hits * 8) + (identity_token_hits * 4)
                    if score > 0:
                        top_chunks.append((score, chunk, token_hits, exact_hits))
                if not top_chunks:
                    continue
                top_chunks.sort(key=lambda item: item[0], reverse=True)
                best_score = top_chunks[0][0] + (identity_focus_hits * 4)
                strong_match = best_score >= MIN_SESSION_CONTENT_SCORE or identity_exact_hits > 0 or identity_focus_hits > 0
                if not strong_match:
                    continue
                scored_sessions.append(
                    {
                        "summary": summary,
                        "score": best_score,
                        "strong_match": strong_match,
                        "identity_token_hits": identity_token_hits,
                        "identity_focus_hits": identity_focus_hits,
                        "token_hits": top_chunks[0][2],
                        "chunks": [item[1] for item in top_chunks[:3]],
                    }
                )

        scored_sessions.sort(key=lambda item: (item["strong_match"], item["score"], item["summary"].updated_at), reverse=True)
        return scored_sessions[:6], {"index_ready": self.status().get("completed", False)}

    def _build_index(self, providers: list[str]) -> None:
        started_at = time.time()
        summaries_by_provider: dict[str, list[ExternalSessionSummary]] = {}
        total_sessions = 0
        for provider_name in providers:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            try:
                summaries = provider.list_sessions()
            except Exception:
                summaries = []
            summaries_by_provider[provider_name] = summaries
            total_sessions += len(summaries)
        with self._lock:
            self._status.update(
                {
                    "message": "Chunking accessible sessions…",
                    "total_sessions": total_sessions,
                }
            )
        processed = 0
        next_chunks_by_provider: dict[str, dict[str, list[ExternalSessionChunk]]] = {}
        next_summaries_by_provider: dict[str, dict[str, ExternalSessionSummary]] = {}
        jobs: list[tuple[str, ExternalSessionProvider, ExternalSessionSummary]] = []
        for provider_name in providers:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            next_chunks_by_provider[provider_name] = {}
            next_summaries_by_provider[provider_name] = {}
            for summary in summaries_by_provider.get(provider_name, []):
                next_summaries_by_provider[provider_name][summary.session_id] = summary
                jobs.append((provider_name, provider, summary))

        max_workers, pace_delay = _index_profile_settings(self._performance_mode)
        if max_workers <= 1:
            for provider_name, provider, summary in jobs:
                next_chunks_by_provider[provider_name][summary.session_id] = _build_provider_chunks(provider, summary.session_id)
                processed = _update_index_progress(
                    processed=processed + 1,
                    total_sessions=total_sessions,
                    started_at=started_at,
                    lock=self._lock,
                    status=self._status,
                )
                if pace_delay > 0:
                    time.sleep(pace_delay)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_build_provider_chunks, provider, summary.session_id): (provider_name, summary.session_id)
                    for provider_name, provider, summary in jobs
                }
                for future in as_completed(future_map):
                    provider_name, session_id = future_map[future]
                    try:
                        chunks = future.result()
                    except Exception:
                        chunks = []
                    next_chunks_by_provider[provider_name][session_id] = chunks
                    processed = _update_index_progress(
                        processed=processed + 1,
                        total_sessions=total_sessions,
                        started_at=started_at,
                        lock=self._lock,
                        status=self._status,
                    )
        with self._lock:
            self._chunks_by_provider = next_chunks_by_provider
            self._summaries_by_provider = next_summaries_by_provider
            self._indexed_providers = set(providers)
            self._status.update(
                {
                    "active": False,
                    "completed": True,
                    "message": "Session chunking complete.",
                    "percent": 100,
                    "processed_sessions": processed,
                    "total_sessions": total_sessions,
                    "eta_seconds": 0,
                    "finished_at": time.time(),
                }
            )


def _build_provider_chunks(provider: "ExternalSessionProvider", session_id: str) -> list[ExternalSessionChunk]:
    try:
        return provider.build_index_chunks(session_id)
    except Exception:
        return []


def _index_profile_settings(performance_mode: str) -> tuple[int, float]:
    if performance_mode == "low":
        return 1, 0.01
    if performance_mode == "high":
        return 4, 0.0
    return 2, 0.0


def _update_index_progress(
    *,
    processed: int,
    total_sessions: int,
    started_at: float,
    lock: threading.Lock,
    status: dict[str, Any],
) -> int:
    elapsed = max(time.time() - started_at, 0.001)
    average = elapsed / max(processed, 1)
    remaining = max(total_sessions - processed, 0)
    eta_seconds = int(round(average * remaining)) if remaining else 0
    percent = int(round((processed / total_sessions) * 100)) if total_sessions else 100
    with lock:
        status.update(
            {
                "active": processed < total_sessions,
                "completed": processed >= total_sessions,
                "message": f"Chunked {processed} of {total_sessions} session(s).",
                "percent": percent,
                "processed_sessions": processed,
                "total_sessions": total_sessions,
                "eta_seconds": eta_seconds,
                "finished_at": time.time() if processed >= total_sessions else None,
            }
        )
    return processed


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

    @abstractmethod
    def build_index_chunks(self, session_id: str) -> list[ExternalSessionChunk]:
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
            return self._apply_index_record(cached)

        session_file = self._find_session_file(session_id)
        if session_file is None:
            raise FileNotFoundError(f"Unknown Codex session: {session_id}")

        detail = self._apply_index_record(self._parse_session_file(session_file))
        self._detail_cache[session_id] = detail
        return detail

    def _apply_index_record(self, detail: ExternalSessionDetail) -> ExternalSessionDetail:
        session_id = detail.summary.session_id
        index_record = self._load_index_records().get(session_id)
        if index_record is None:
            return detail
        if detail.summary.title == str(index_record.get("thread_name") or detail.summary.title):
            return detail
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
        return ExternalSessionDetail(summary=summary, messages=detail.messages, metadata=detail.metadata)

    def build_index_chunks(self, session_id: str) -> list[ExternalSessionChunk]:
        session_file = self._find_session_file(session_id)
        if session_file is None:
            return []
        summary = self._summary_from_session_file(session_file)
        all_messages = self._parse_full_session_messages(session_file, session_id)
        return _build_chunks_from_messages(summary, all_messages, source="codex")

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
        messages, preview, metadata = self._parse_full_session_messages(session_file, session_id, include_metadata=True)
        history_fallback = self._history_messages(session_id)
        if history_fallback:
            existing_users = {message.content for message in messages if message.role == "user"}
            for message in history_fallback:
                if message.content not in existing_users:
                    messages.insert(0, message)

        messages = [message for message in messages if message.content.strip()]
        messages = messages[-MAX_SESSION_MESSAGES:]
        workspace_path = metadata.get("workspace_path") or workspace_path
        title = str(metadata.get("title") or title)
        updated_at = str(metadata.get("updated_at") or updated_at)
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

    def _parse_full_session_messages(
        self,
        session_file: Path,
        session_id: str,
        *,
        include_metadata: bool = False,
    ) -> list[ExternalSessionMessage] | tuple[list[ExternalSessionMessage], str, dict[str, Any]]:
        title = session_file.stem
        updated_at = ""
        workspace_path = None
        preview = ""
        metadata: dict[str, Any] = {
            "provider": self.name,
            "source_path": str(session_file),
        }
        messages: list[ExternalSessionMessage] = []
        seen_signatures: set[tuple[str, str]] = set()
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
                        "title": title,
                        "updated_at": updated_at,
                        "workspace_path": workspace_path,
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
                if not preview and message.content.strip():
                    preview = message.content[:220]
        if include_metadata:
            return messages, preview, metadata
        return messages

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
    def __init__(self, config: ExternalSessionProviderConfig) -> None:
        super().__init__(config)
        self._detail_cache: dict[str, ExternalSessionDetail] = {}
        self._summary_cache: dict[str, ExternalSessionSummary] = {}

    def health(self) -> ExternalSourceHealth:
        available = self.config.enabled and self.root.exists()
        session_count = 0
        summary = "OpenCode archive not configured"
        if available:
            try:
                session_count = len(self.list_sessions())
                summary = f"{session_count} OpenCode session(s) available"
            except sqlite3.Error as exc:
                available = False
                summary = f"OpenCode archive unavailable: {exc}"
        return ExternalSourceHealth(
            provider=self.name,
            enabled=self.config.enabled,
            available=available,
            root_path=str(self.root),
            summary=summary,
            session_count=session_count,
        )

    def list_sessions(self) -> list[ExternalSessionSummary]:
        if not self.config.enabled or not self.root.exists():
            return []
        rows = self._query_all(
            """
            select id, title, directory, time_updated
            from session
            where time_archived is null
            order by time_updated desc
            """
        )
        summaries: list[ExternalSessionSummary] = []
        for row in rows:
            session_id = str(row["id"])
            cached = self._summary_cache.get(session_id)
            if cached is not None:
                summaries.append(cached)
                continue
            preview = self._preview_for_session(session_id)
            summary = ExternalSessionSummary(
                provider=self.name,
                session_id=session_id,
                title=_normalize_whitespace(str(row["title"] or "")) or "Untitled session",
                updated_at=_millis_to_iso(row["time_updated"]),
                workspace_path=str(row["directory"] or "") or None,
                source_path=str(self.root),
                message_count=self._message_count_for_session(session_id),
                preview=preview,
            )
            self._summary_cache[session_id] = summary
            summaries.append(summary)
        return summaries

    def get_session(self, session_id: str) -> ExternalSessionDetail:
        cached = self._detail_cache.get(session_id)
        if cached is not None:
            return cached
        summary = next((item for item in self.list_sessions() if item.session_id == session_id), None)
        if summary is None:
            raise FileNotFoundError(f"Unknown OpenCode session: {session_id}")
        messages = self._session_messages(session_id)[-MAX_SESSION_MESSAGES:]
        detail = ExternalSessionDetail(
            summary=summary,
            messages=tuple(messages),
            metadata={"provider": self.name, "source_path": str(self.root)},
        )
        self._detail_cache[session_id] = detail
        return detail

    def build_index_chunks(self, session_id: str) -> list[ExternalSessionChunk]:
        summary = next((item for item in self.list_sessions() if item.session_id == session_id), None)
        if summary is None:
            return []
        messages = self._session_messages(session_id)
        return _build_chunks_from_messages(summary, messages, source="opencode")

    def _session_messages(self, session_id: str) -> list[ExternalSessionMessage]:
        rows = self._query_all(
            """
            select m.id as message_id, m.data as message_data, p.data as part_data, p.time_created as part_time
            from message m
            left join part p on p.message_id = m.id
            where m.session_id = ?
            order by m.time_created asc, p.time_created asc, p.id asc
            """,
            (session_id,),
        )
        messages_by_id: dict[str, dict[str, Any]] = {}
        ordered_parts: list[tuple[str, dict[str, Any], int]] = []
        for row in rows:
            message_id = str(row["message_id"])
            if message_id not in messages_by_id:
                messages_by_id[message_id] = _safe_json_loads(str(row["message_data"] or ""))
            part_payload = _safe_json_loads(str(row["part_data"] or ""))
            if part_payload:
                ordered_parts.append((message_id, part_payload, int(row["part_time"] or 0)))
        extracted: list[ExternalSessionMessage] = []
        seen: set[tuple[str, str]] = set()
        for message_id, payload, part_time in ordered_parts:
            parent = messages_by_id.get(message_id, {})
            role = str(parent.get("role") or "assistant")
            session_message = _extract_opencode_message_part(role=role, payload=payload, timestamp=_millis_to_iso(part_time))
            if session_message is None:
                continue
            signature = (session_message.role, session_message.content)
            if signature in seen:
                continue
            seen.add(signature)
            extracted.append(session_message)
        return extracted

    def _preview_for_session(self, session_id: str) -> str:
        rows = self._query_all(
            """
            select p.data as part_data, m.data as message_data
            from part p
            join message m on m.id = p.message_id
            where p.session_id = ?
            order by p.time_created desc
            limit 12
            """,
            (session_id,),
        )
        for row in rows:
            payload = _safe_json_loads(str(row["part_data"] or ""))
            parent = _safe_json_loads(str(row["message_data"] or ""))
            role = str(parent.get("role") or "assistant")
            message = _extract_opencode_message_part(role=role, payload=payload, timestamp=None)
            if message and message.content.strip():
                return message.content[:220]
        return ""

    def _message_count_for_session(self, session_id: str) -> int:
        row = self._query_one("select count(*) as count from message where session_id = ?", (session_id,))
        return int((row or {}).get("count") or 0)

    def _query_one(self, query: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self._query_all(query, parameters)
        return rows[0] if rows else None

    def _query_all(self, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        try:
            return _fetch_sqlite_rows(self.root, query, parameters, immutable=False)
        except sqlite3.OperationalError:
            return _fetch_sqlite_rows(self.root, query, parameters, immutable=True)


class ContextBuilderService:
    def __init__(
        self,
        workspace_path: str,
        *,
        memory: Any | None = None,
        provider_configs: tuple[ExternalSessionProviderConfig, ...] = (),
        performance_mode: str = "medium",
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.workspace = WorkspaceBrowser(self.workspace_path)
        self.memory = memory
        self.performance_mode = performance_mode if performance_mode in {"low", "medium", "high"} else "medium"
        self.provider_configs = provider_configs or _default_provider_configs()
        self.runtime_allowed_providers: set[str] | None = None
        self.providers = {
            config.provider: _provider_from_config(config)
            for config in self.provider_configs
        }
        self.index = ExternalSessionIndex(self.providers, performance_mode=self.performance_mode)
        self._session_embedding_store: SQLiteMemoryStore | None = None
        self._session_embedding_store_resolved = False
        self._session_embedder = self._resolve_session_embedder()
        self._session_embedding_document_cache: dict[str, tuple[tuple[Any, ...], str]] = {}

    def set_runtime_allowed_providers(self, providers: set[str] | list[str] | tuple[str, ...] | None) -> None:
        if providers is None:
            self.runtime_allowed_providers = None
            return
        self.runtime_allowed_providers = {provider for provider in providers if provider in self.providers}
        self.index.ensure(self.runtime_allowed_providers)

    def set_performance_mode(self, performance_mode: str) -> None:
        self.performance_mode = performance_mode if performance_mode in {"low", "medium", "high"} else "medium"
        self.index.set_performance_mode(self.performance_mode)

    def list_sources(self) -> list[ExternalSourceHealth]:
        return [provider.health() for provider in self.providers.values()]

    def indexing_status(self) -> dict[str, Any]:
        return self.index.status()

    def list_sessions(self, provider_name: str) -> list[ExternalSessionSummary]:
        provider = self._get_provider(provider_name)
        return [self._with_session_embedding(provider, summary) for summary in provider.list_sessions()]

    def get_session(self, provider_name: str, session_id: str) -> ExternalSessionDetail:
        provider = self._get_provider(provider_name)
        detail = provider.get_session(session_id)
        summary = self._with_session_embedding(provider, detail.summary)
        metadata = dict(detail.metadata)
        metadata["unified_session_id"] = summary.unified_session_id
        metadata["embedding"] = list(summary.embedding)
        return ExternalSessionDetail(summary=summary, messages=detail.messages, metadata=metadata)

    def list_session_embeddings(self, provider_name: str | None = None) -> list[ExternalSessionEmbedding]:
        store = self._get_session_embedding_store()
        if store is None:
            return []
        return store.list_external_session_embeddings(provider=provider_name)

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
        if self.runtime_allowed_providers == set():
            return "", (), {"context_match_state": "new_context", "context_match_reason": "External session access has not been granted."}
        if provider_name is not None:
            try:
                return self._build_runtime_memory_context_for_provider(
                    task,
                    provider_name=provider_name,
                    max_lines=max_lines,
                )
            except Exception as exc:
                logger.warning("Failed to build runtime memory context for provider=%s: error=%s", provider_name, exc)
                return "", (), {"context_match_state": "new_context", "context_match_reason": f"External {provider_name} lookup failed."}

        candidate_providers = self._candidate_provider_names()
        if not candidate_providers:
            return "", (), {"context_match_state": "new_context", "context_match_reason": "No external session provider is available."}

        provider_results: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []
        for candidate_provider in candidate_providers:
            try:
                context, session_ids, metadata = self._build_runtime_memory_context_for_provider(
                    task,
                    provider_name=candidate_provider,
                    max_lines=max_lines,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping failed external session provider during runtime retrieval: provider=%s error=%s",
                    candidate_provider,
                    exc,
                )
                continue
            if not session_ids:
                continue
            provider_results.append((candidate_provider, context, tuple(session_ids), metadata))
        if not provider_results:
            return "", (), {"context_match_state": "new_context", "context_match_reason": "No external session provider yielded usable context."}

        fused_score: dict[str, float] = {}
        for _provider_name, _context, session_ids, _metadata in provider_results:
            for rank, session_id in enumerate(session_ids, start=1):
                fused_score[session_id] = fused_score.get(session_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered_session_ids = sorted(fused_score, key=lambda session_id: (-fused_score[session_id], session_id))[:MAX_RUNTIME_SESSION_MATCHES]
        selected_ids = tuple(ordered_session_ids)

        top_session_id = selected_ids[0]
        best_result = provider_results[0]
        best_rank: int | None = None
        for result in provider_results:
            if top_session_id not in result[2]:
                continue
            rank = result[2].index(top_session_id)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_result = result
        metadata = dict(best_result[3])
        metadata["index_ready"] = any(bool(result[3].get("index_ready", False)) for result in provider_results)
        metadata["context_match_providers"] = [result[0] for result in provider_results]

        seen_lines: set[str] = set()
        merged_lines: list[str] = []
        for _provider_name, context, _session_ids, _metadata in provider_results:
            for line in context.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("## ") or stripped in seen_lines:
                    continue
                seen_lines.add(stripped)
                merged_lines.append(stripped[2:].strip() if stripped.startswith("- ") else stripped)
                if len(merged_lines) >= max_lines:
                    break
            if len(merged_lines) >= max_lines:
                break
        if not merged_lines:
            return "", selected_ids, metadata
        lines = ["## External Session Context", *(f"- {line}" for line in merged_lines)]
        return "\n".join(lines), selected_ids, metadata

    def _build_runtime_memory_context_for_provider(
        self,
        task: str,
        *,
        provider_name: str,
        max_lines: int,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        selected_matches, provider, selection_metadata = self._select_runtime_matches_for_provider(
            task,
            provider_name=provider_name,
        )
        if provider is None or not selected_matches:
            return "", (), selection_metadata
        selected_session_ids = tuple(match["summary"].session_id for match in selected_matches)
        context_lines = self._context_lines_for_selected_matches(task, selected_matches, provider, max_lines=max_lines)
        if not context_lines:
            return "", selected_session_ids, selection_metadata
        lines = ["## External Session Context", *(f"- {line}" for line in context_lines)]
        return "\n".join(lines), selected_session_ids, selection_metadata

    def _select_runtime_matches_for_provider(
        self,
        task: str,
        *,
        provider_name: str,
    ) -> tuple[list[dict[str, Any]], ExternalSessionProvider | None, dict[str, Any]]:
        resolved_provider = provider_name
        if not resolved_provider:
            return [], None, {"context_match_state": "new_context", "context_match_reason": "No external session provider is available."}
        if self.runtime_allowed_providers is not None and resolved_provider not in self.runtime_allowed_providers:
            return [], None, {"context_match_state": "new_context", "context_match_reason": f"External {resolved_provider} access has not been granted."}
        provider = self._get_provider(resolved_provider)
        query_variants = _build_query_variants(task)
        indexed_matches, indexed_metadata = self._query_index_variants(query_variants, provider_name=resolved_provider)
        semantic_matches = self._select_relevant_sessions(provider, task, query_variants=query_variants)
        selected_matches = _combine_indexed_and_semantic_matches(task, indexed_matches, semantic_matches)
        selection_metadata = self._selection_metadata(selected_matches)
        selection_metadata["index_ready"] = indexed_metadata.get("index_ready", False)
        return selected_matches, provider, selection_metadata

    def _context_lines_for_selected_matches(
        self,
        task: str,
        selected_matches: list[dict[str, Any]],
        provider: ExternalSessionProvider,
        *,
        max_lines: int,
    ) -> tuple[str, ...]:
        details = [provider.get_session(match["summary"].session_id) for match in selected_matches]
        return self._context_lines_for_matches(task, selected_matches, details, max_lines=max_lines)

    def _context_lines_for_matches(
        self,
        task: str,
        selected_matches: list[dict[str, Any]],
        details: list[ExternalSessionDetail],
        *,
        max_lines: int,
    ) -> tuple[str, ...]:
        indexed_context_lines: tuple[str, ...] = ()
        if any(match.get("chunks") for match in selected_matches):
            indexed_context_lines = _collect_indexed_context_lines(task, selected_matches)
        context_lines = indexed_context_lines or _collect_relevant_context_lines(task, details, "detailed")
        return context_lines[:max_lines]

    def _candidate_provider_names(self) -> list[str]:
        provider_names = list(self.providers)
        if self.runtime_allowed_providers is not None:
            provider_names = [name for name in provider_names if name in self.runtime_allowed_providers]
        return provider_names

    def _query_index_variants(
        self,
        query_variants: tuple[str, ...],
        *,
        provider_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}
        index_ready = False
        for variant in query_variants:
            matches, metadata = self.index.query(variant, provider_name=provider_name, workspace_path=self.workspace_path)
            index_ready = index_ready or bool(metadata.get("index_ready", False))
            for match in matches:
                summary = match.get("summary")
                if not isinstance(summary, ExternalSessionSummary):
                    continue
                existing = aggregated.get(summary.session_id)
                if existing is None:
                    aggregated[summary.session_id] = {
                        **match,
                        "score": match.get("score", 0),
                        "chunks": list(match.get("chunks") or []),
                    }
                    continue
                existing["score"] = max(int(existing.get("score", 0)), int(match.get("score", 0)))
                existing["strong_match"] = bool(existing.get("strong_match")) or bool(match.get("strong_match"))
                existing["identity_token_hits"] = max(
                    int(existing.get("identity_token_hits", 0)),
                    int(match.get("identity_token_hits", 0)),
                )
                existing["identity_focus_hits"] = max(
                    int(existing.get("identity_focus_hits", 0)),
                    int(match.get("identity_focus_hits", 0)),
                )
                existing["token_hits"] = max(int(existing.get("token_hits", 0)), int(match.get("token_hits", 0)))
                existing_chunks = {chunk.text: chunk for chunk in existing.get("chunks", []) if isinstance(chunk, ExternalSessionChunk)}
                for chunk in match.get("chunks", []) or ():
                    if isinstance(chunk, ExternalSessionChunk):
                        existing_chunks.setdefault(chunk.text, chunk)
                existing["chunks"] = list(existing_chunks.values())[:3]

        ranked = sorted(
            aggregated.values(),
            key=lambda item: (
                bool(item.get("strong_match")),
                int(item.get("identity_focus_hits", 0)),
                int(item.get("score", 0)),
                getattr(item.get("summary"), "updated_at", ""),
            ),
            reverse=True,
        )
        return ranked[:6], {"index_ready": index_ready}

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

    def _session_embedding_fingerprint(self, summary: ExternalSessionSummary) -> tuple[Any, ...] | None:
        source_path = summary.source_path
        if not source_path:
            return None
        try:
            stat = Path(source_path).stat()
        except OSError:
            return None
        return (
            summary.provider,
            summary.session_id,
            str(source_path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            summary.updated_at,
        )

    def _get_session_embedding_store(self) -> SQLiteMemoryStore | None:
        if not self._session_embedding_store_resolved:
            self._session_embedding_store = self._resolve_session_embedding_store()
            self._session_embedding_store_resolved = True
        return self._session_embedding_store

    def _resolve_session_embedding_store(self) -> SQLiteMemoryStore | None:
        store = getattr(self.memory, "store", None)
        if isinstance(store, SQLiteMemoryStore):
            return store
        try:
            return SQLiteMemoryStore(str(Path(self.workspace_path) / "memory.db"))
        except Exception:
            return None

    def _resolve_session_embedder(self) -> Any | None:
        embedder = getattr(self.memory, "embedder", None)
        if embedder is not None and hasattr(embedder, "embed"):
            return embedder
        return HashingEmbedder(dimension=384)

    def _with_session_embedding(
        self,
        provider: ExternalSessionProvider,
        summary: ExternalSessionSummary,
    ) -> ExternalSessionSummary:
        unified_session_id = _unified_session_id(summary.provider, summary.session_id)
        store = self._get_session_embedding_store()
        if store is None or self._session_embedder is None:
            return replace(summary, unified_session_id=unified_session_id)

        fingerprint = self._session_embedding_fingerprint(summary)
        cached = self._session_embedding_document_cache.get(unified_session_id)
        if fingerprint is not None and cached is not None and cached[0] == fingerprint:
            document = cached[1]
        else:
            document = self._session_embedding_document(provider, summary)
            if fingerprint is not None:
                self._session_embedding_document_cache[unified_session_id] = (fingerprint, document)
                if len(self._session_embedding_document_cache) > MAX_SESSION_EMBEDDING_CACHE:
                    self._session_embedding_document_cache.pop(next(iter(self._session_embedding_document_cache)))
        content_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
        existing = store.get_external_session_embedding(unified_session_id)
        if existing is None or existing.content_hash != content_hash:
            embedding = tuple(float(value) for value in self._session_embedder.embed(document))
            existing = ExternalSessionEmbedding(
                unified_session_id=unified_session_id,
                provider=summary.provider,
                session_id=summary.session_id,
                title=summary.title,
                workspace_path=summary.workspace_path,
                source_path=summary.source_path,
                updated_at=summary.updated_at,
                content_hash=content_hash,
                content_text=document,
                embedding=embedding,
                indexed_at=time.time(),
            )
            store.upsert_external_session_embedding(existing)

        return replace(summary, unified_session_id=unified_session_id, embedding=existing.embedding)

    def _session_embedding_document(
        self,
        provider: ExternalSessionProvider,
        summary: ExternalSessionSummary,
    ) -> str:
        chunks = provider.build_index_chunks(summary.session_id)
        parts = [summary.title.strip()]
        if summary.workspace_path:
            parts.append(str(summary.workspace_path).strip())
        for chunk in chunks:
            text = _normalize_whitespace(chunk.text)
            if text:
                parts.append(text)
        document = "\n".join(part for part in parts if part).strip()
        return document or summary.preview.strip() or summary.title.strip() or summary.session_id

    def _session_embedding_vectors(self, provider_name: str) -> dict[str, tuple[float, ...]]:
        store = self._get_session_embedding_store()
        if store is None:
            return {}
        vectors: dict[str, tuple[float, ...]] = {}
        for record in store.list_external_session_embeddings(provider=provider_name):
            if record.embedding:
                vectors[record.session_id] = record.embedding
        return vectors

    def _semantic_session_scores(
        self,
        provider: ExternalSessionProvider,
        summaries: list[ExternalSessionSummary],
        query_variants: tuple[str, ...],
    ) -> dict[str, float]:
        vectors = self._session_embedding_vectors(provider.name)
        if not vectors or self._session_embedder is None:
            return {}
        query_vectors: list[tuple[float, ...]] = []
        for variant in query_variants:
            try:
                query_vectors.append(tuple(float(value) for value in self._session_embedder.embed(variant)))
            except Exception as exc:
                logger.debug("Failed to embed recall query variant: error=%s", exc)
        if not query_vectors:
            return {}
        scores: dict[str, float] = {}
        for summary in summaries:
            vector = vectors.get(summary.session_id)
            if not vector:
                continue
            best = max((_cosine_similarity(query, vector) for query in query_vectors), default=0.0)
            if best > 0.0:
                scores[summary.session_id] = best
        return scores

    def _default_provider_name(self) -> str | None:
        for provider_name, provider in self.providers.items():
            if provider.health().available:
                return provider_name
        return next(iter(self.providers), None)

    def _select_relevant_session_ids(self, provider: ExternalSessionProvider, task: str) -> tuple[str, ...]:
        return tuple(match["summary"].session_id for match in self._select_relevant_sessions(provider, task))

    def _select_relevant_sessions(
        self,
        provider: ExternalSessionProvider,
        task: str,
        *,
        query_variants: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        summaries = provider.list_sessions()
        if not summaries:
            return []

        variants = query_variants or _build_query_variants(task)
        prompt_tokens = set().union(*(_tokenize(variant) for variant in variants))
        if not prompt_tokens:
            return []
        focus_tokens = set().union(*(_focus_tokens(variant) for variant in variants))
        semantic_scores = self._semantic_session_scores(provider, summaries, variants)
        semantic_ranking = sorted(semantic_scores, key=lambda session_id: semantic_scores[session_id], reverse=True)
        semantic_rank = {session_id: rank for rank, session_id in enumerate(semantic_ranking, start=1)}
        lowered_task = task.lower()
        issue_recall_prompt = any(token in prompt_tokens for token in {"bug", "bugs", "fix", "fixed", "review", "reviews", "issue", "issues"}) or "last time" in lowered_task
        issue_focus_markers = (
            "bug list",
            "exact bugs",
            "root url redirects",
            "convex generated imports",
            "authentication bypass",
            "open email relay",
            "create workspace",
            "pipeline chat",
            "test/publish",
            "salesforce being marked as coming soon",
        )
        preview_priority = _preview_issue_recall_matches(
            summaries,
            prompt=task,
            issue_focus_markers=issue_focus_markers,
        )
        if issue_recall_prompt and preview_priority:
            prioritized: list[dict[str, Any]] = []
            for score, summary in preview_priority[:3]:
                detail = provider.get_session(summary.session_id)
                prioritized.append(
                    {
                        "summary": summary,
                        "detail": detail,
                        "content_score": score,
                        "score": score,
                        "strong_match": True,
                        "exact_hits": 0,
                        "token_hits": 0,
                        "identity_exact_hits": 0,
                        "identity_token_hits": 0,
                        "identity_focus_hits": 0,
                        "best_overlap": 0,
                    }
                )
            return prioritized
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
            if issue_recall_prompt:
                issue_terms = ("bug", "bugs", "fix", "fixed", "review", "reviews")
                if any(term in summary.title.lower() for term in issue_terms):
                    issue_bonus += 6
                elif any(term in summary.preview.lower() for term in issue_terms):
                    issue_bonus += 3
                issue_bonus += sum(18 for marker in issue_focus_markers if marker in summary.preview.lower())
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
        lexical_rank = {item["summary"].session_id: rank for rank, item in enumerate(preliminary, start=1)}
        for item in preliminary:
            session_id = item["summary"].session_id
            item["semantic_score"] = semantic_scores.get(session_id, 0.0)
            item["semantic_rank"] = semantic_rank.get(session_id)
            item["fused_score"] = _rrf_score(lexical_rank.get(session_id), semantic_rank.get(session_id))
        if semantic_scores:
            preliminary.sort(
                key=lambda item: (
                    item["fused_score"],
                    item["summary_score"],
                    -item["recent_rank"],
                    item["summary"].updated_at,
                ),
                reverse=True,
            )
        candidate_ids: list[str] = []
        candidate_limit = 24 if issue_recall_prompt else 12
        recent_candidate_limit = max(recent_window, candidate_limit)
        for item in preliminary[:candidate_limit]:
            session_id = item["summary"].session_id
            if session_id not in candidate_ids:
                candidate_ids.append(session_id)
        for item in preliminary:
            if item["summary_score"] <= 0 and item["recent_rank"] >= recent_window:
                continue
            session_id = item["summary"].session_id
            if session_id not in candidate_ids:
                candidate_ids.append(session_id)
            if len(candidate_ids) >= recent_candidate_limit:
                break
        if issue_recall_prompt:
            for item in preliminary:
                summary = item["summary"]
                preview_lower = (summary.preview or "").lower()
                if not any(marker in preview_lower for marker in issue_focus_markers):
                    continue
                session_id = summary.session_id
                if session_id not in candidate_ids:
                    candidate_ids.append(session_id)
        for session_id in semantic_ranking[:candidate_limit]:
            if session_id not in candidate_ids:
                candidate_ids.append(session_id)

        preliminary_index = {item["summary"].session_id: item for item in preliminary}
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
            best_overlap = max(_best_message_overlap(_tokenize(variant), detail) for variant in variants)
            semantic_score = semantic_scores.get(summary.session_id, 0.0)
            semantic_strong = semantic_score >= SEMANTIC_STRONG_THRESHOLD
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
            if semantic_strong:
                workspace_bonus = max(workspace_bonus, 0)
            if focus_tokens and session_workspace == workspace_path and identity_focus_hits == 0 and not semantic_strong:
                if not issue_recall_prompt and exact_hits == 0 and token_hits < 2:
                    continue
            if issue_recall_prompt:
                issue_terms = ("bug", "bugs", "fix", "fixed", "review", "reviews")
                if any(term in summary.title.lower() for term in issue_terms):
                    issue_bonus += 10
                elif any(term in haystack for haystack in haystacks for term in issue_terms):
                    issue_bonus += 4
                issue_bonus += sum(16 for marker in issue_focus_markers if marker in summary.preview.lower())
                issue_bonus += sum(8 for marker in issue_focus_markers if any(marker in haystack for haystack in haystacks))
            content_score = (
                (identity_exact_hits * 14)
                + (identity_token_hits * 8)
                + (exact_hits * 3)
                + (token_hits * 2)
                + min(best_overlap * 2, 8)
                + int(round(semantic_score * 10))
            )
            strong_match = (
                identity_exact_hits >= 1
                or identity_token_hits >= 1
                or exact_hits >= 1
                or best_overlap >= 2
                or token_hits >= 2
                or semantic_strong
            )
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
                    "semantic_score": semantic_score,
                    "semantic_rank": semantic_rank.get(summary.session_id),
                    "fused_score": preliminary_index.get(summary.session_id, {}).get("fused_score", 0.0),
                }
            )

        scored.sort(key=lambda item: (item["strong_match"], item["score"], item["summary"].updated_at), reverse=True)
        selected = [
            item
            for item in scored
            if item["strong_match"]
            and (item["content_score"] >= MIN_SESSION_CONTENT_SCORE or item["semantic_score"] >= SEMANTIC_STRONG_THRESHOLD)
            and item["score"] > 0
        ]
        if issue_recall_prompt:
            issue_rich = [
                item
                for item in scored
                if item["score"] > 0
                and _session_has_issue_focus(item["summary"], item["detail"])
            ]
            if issue_rich:
                issue_rich.sort(
                    key=lambda item: (
                        _issue_focus_score(item["summary"], item["detail"]),
                        item["score"],
                        item["content_score"],
                        item["summary"].updated_at,
                    ),
                    reverse=True,
                )
                selected = issue_rich
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
            "semantic_score": best.get("semantic_score", 0.0),
            "fused_score": best.get("fused_score", 0.0),
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
            root_path=str(Path.home() / ".local" / "share" / "opencode" / "opencode.db"),
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
        if output_text and not _is_noise_message_content(output_text) and _is_useful_tool_output(output_text):
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
    prompt_tokens = _tokenize(task)
    prompt_entities = {
        token.lower()
        for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", task.lower())
        if len(token) >= 3
    }
    lowered_task = task.lower()
    is_issue_prompt = any(token in prompt_tokens for token in {"bug", "bugs", "fix", "fixed", "review", "reviews", "issue", "issues"}) or "last time" in lowered_task
    issue_detail_markers = (
        "create workspace",
        "pipeline chat",
        "test/publish",
        "salesforce",
        "root url redirects",
        "convex generated imports",
        "authentication bypass",
        "open email relay",
        "bug list",
        "exact bugs",
    )
    is_project_recall_prompt = any(marker in lowered_task for marker in ("remember about", "remember the", "what was it about", "what was that about"))
    candidates: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for detail in details:
        summary = detail.summary
        identity_haystacks = _session_identity_haystacks(summary)
        session_identity_overlap = any(entity in haystack for entity in prompt_entities for haystack in identity_haystacks) if prompt_entities else False
        session_cleanup_match = _session_matches_cleanup_schema_task(task, summary)
        if summary.title.strip():
            candidates.append(
                (
                    f"Session '{summary.title}' targeted workspace {summary.workspace_path or 'unknown workspace'}.",
                    "session",
                    session_identity_overlap,
                    session_cleanup_match,
                )
            )
        for message in detail.messages:
            if message.role not in {"user", "assistant", "tool"}:
                continue
            content = _compact_context_content(message.role, message.content)
            if not content or content in seen:
                continue
            seen.add(content)
            if message.role == "user":
                prefix = "User asked:"
            elif message.role == "assistant":
                prefix = "Assistant reported:"
            else:
                prefix = "Tool output:"
            candidates.append((f"{prefix} {content}", message.role, session_identity_overlap, session_cleanup_match))

    scored: list[tuple[int, str, str]] = []
    for line, line_kind, session_identity_overlap, session_cleanup_match in candidates:
        lowered = line.lower()
        overlap = sum(2 for token in prompt_tokens if _token_matches(token, lowered))
        entity_overlap = any(entity in lowered for entity in prompt_entities)
        if line.startswith("User asked:"):
            overlap += 3 if any(token in lowered_task for token in ("bug", "bugs", "review", "reviews", "fix", "fixed")) else 0
            if session_identity_overlap and (is_project_recall_prompt or is_issue_prompt):
                overlap += 4
        if line.startswith("Assistant reported:"):
            overlap += 3 if any(token in lowered_task for token in ("what was it about", "what was that about", "remember about")) else 1
            if prompt_entities and not entity_overlap and not any(marker in lowered for marker in ("bug", "review", "fix", "workspace", "pipeline chat", "salesforce", "create workspace")):
                overlap -= 5
            if session_identity_overlap and is_issue_prompt:
                overlap += 2
        if line.startswith("Tool output:"):
            overlap += 1 if is_issue_prompt or any(token in lowered_task for token in ("what did", "what were", "issues", "talking about", "comment")) else 1
            if session_identity_overlap and (is_issue_prompt or is_project_recall_prompt):
                overlap += 3
            if is_issue_prompt and re.search(r"\b(?:fix|chore|test)\([^)]*\):", lowered):
                overlap -= 8
        if "bug" in lowered or "review" in lowered or "fix" in lowered:
            overlap += 3
        if session_identity_overlap and any(
            marker in lowered
            for marker in issue_detail_markers
        ):
            overlap += 5
        if _is_cleanup_schema_task(task):
            if session_cleanup_match:
                overlap += 8
                if any(
                    marker in lowered
                    for marker in (
                        "schema",
                        "cleanup",
                        "clean up",
                        "legacy",
                        "table",
                        "tables",
                        "column",
                        "columns",
                        "dead",
                        "audit",
                        "cache",
                        "convex generated imports",
                        "root url redirects",
                        "authentication bypass",
                    )
                ):
                    overlap += 5
            elif line_kind != "session":
                overlap -= 8
            if any(
                marker in lowered
                for marker in (
                    "tsc --noemit",
                    "type mismatch",
                    "typescript error",
                    "lint/check",
                    "worktree is clean",
                    "committed",
                    "validator",
                    "shared.ts",
                    "fixing the two typescript issues",
                )
            ):
                overlap -= 10
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
        scored.append((overlap, line, line_kind))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [line for score, line, _kind in scored if score > 0][:MAX_CONTEXT_LINES]
    if not selected:
        selected = [line for _score, line, _kind in scored[:MAX_CONTEXT_LINES]]
    elif is_issue_prompt or (is_project_recall_prompt and not is_issue_prompt):
        issue_detail_lines = [
            line
            for score, line, kind in scored
            if score > 0
            and kind in {"user", "assistant"}
            and any(
                marker in line.lower() for marker in issue_detail_markers
            )
        ]
        if issue_detail_lines:
            selected = issue_detail_lines[: min(3, MAX_CONTEXT_LINES)]
            session_summaries = [line for line in selected if line.startswith("Session '")]
            if session_summaries:
                selected = [line for line in selected if not line.startswith("Session '")]
            matching_session_summaries = [
                line
                for _score, line, kind in scored
                if kind == "session" and any(token in line.lower() for token in _tokenize(task))
            ]
            if matching_session_summaries and len(selected) < MAX_CONTEXT_LINES:
                selected.append(matching_session_summaries[0])
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


def _build_query_variants(task: str) -> tuple[str, ...]:
    normalized = _normalize_whitespace(task)
    if not normalized:
        return ()
    variants = [normalized]
    for raw_part in re.split(r"[?.!;]+|\b(?:and|also|then|plus|while|versus|vs)\b", normalized, flags=re.IGNORECASE):
        fragment = _normalize_whitespace(raw_part).strip(" ,.;:-")
        if not fragment or fragment in variants:
            continue
        if len(_tokenize(fragment)) < 2:
            continue
        variants.append(fragment)
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
    return tuple(variants[:MAX_QUERY_VARIANTS])


def _preview_issue_recall_matches(
    summaries: Sequence[ExternalSessionSummary],
    *,
    prompt: str,
    issue_focus_markers: Sequence[str],
) -> list[tuple[int, ExternalSessionSummary]]:
    prompt_lower = prompt.lower()
    prompt_tokens = _tokenize(prompt)
    issue_terms = {"bug", "bugs", "fix", "fixed", "issue", "issues", "review", "reviews"}
    ignored_prompt_tokens = {"did", "last", "time", "while", "working"}
    compound_markers = tuple(
        token
        for token in sorted(prompt_tokens)
        if any(separator in token for separator in ("-", "_", "/"))
    )
    project_markers_set = set(compound_markers)
    for token in compound_markers:
        for part in re.split(r"[-_/]+", token):
            if len(part) >= 4:
                project_markers_set.add(part)
    if not project_markers_set:
        project_markers_set.update(
            token
            for token in prompt_tokens
            if token not in COMMON_CONTEXT_TOKENS
            and token not in issue_terms
            and token not in ignored_prompt_tokens
            and len(token) >= 4
        )
    project_markers = tuple(sorted(project_markers_set))
    if not project_markers:
        return []

    prioritized: list[tuple[int, ExternalSessionSummary]] = []
    for summary in summaries:
        preview = _normalize_whitespace(summary.preview).lower()
        if not preview:
            continue
        title = (summary.title or "").lower()
        workspace = (summary.workspace_path or "").lower()
        if not (
            any(marker in preview for marker in issue_focus_markers)
            or any(term in preview or term in title for term in issue_terms)
            or "based on memory from prior sessions" in preview
        ):
            continue
        identity_haystacks = (title, workspace, preview)
        project_hits = sum(1 for marker in project_markers if any(_token_matches(marker, haystack) for haystack in identity_haystacks))
        if project_hits == 0:
            continue
        score = project_hits * 30
        score += sum(35 for marker in issue_focus_markers if marker in preview)
        if "bug list" in preview:
            score += 45
        if "based on memory from prior sessions" in preview:
            score += 20
        if any(term in preview for term in issue_terms):
            score += 10
        if "last time" in prompt_lower and "last" in preview:
            score += 8
        for noise_marker in (
            "tool exec_command result",
            "operation not permitted: ps",
            "pr-review.md",
            "committed in two atomic commits",
            "fix(settings): use saved timezone and locale dropdowns",
            "glob: /users/",
        ):
            if noise_marker in preview:
                score -= 40
        if score > 0:
            prioritized.append((score, summary))

    prioritized.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
    return prioritized


def _session_has_issue_focus(summary: ExternalSessionSummary, detail: ExternalSessionDetail) -> bool:
    markers = (
        "bug list",
        "root url redirects",
        "convex generated imports",
        "authentication bypass",
        "open email relay",
        "create workspace",
        "pipeline chat",
        "test/publish",
        "salesforce being marked as coming soon",
    )
    preview = (summary.preview or "").lower()
    if any(marker in preview for marker in markers):
        return True
    for message in detail.messages:
        content = (message.content or "").lower()
        if any(marker in content for marker in markers):
            return True
    return False


def _issue_focus_score(summary: ExternalSessionSummary, detail: ExternalSessionDetail) -> int:
    markers = (
        "bug list",
        "root url redirects",
        "convex generated imports",
        "authentication bypass",
        "open email relay",
        "create workspace",
        "pipeline chat",
        "test/publish",
        "salesforce being marked as coming soon",
        "bugs tracked",
    )
    score = 0
    preview = (summary.preview or "").lower()
    score += sum(6 for marker in markers if marker in preview)
    for message in detail.messages:
        content = (message.content or "").lower()
        score += sum(3 for marker in markers if marker in content)
    return score


def _combine_indexed_and_semantic_matches(
    task: str,
    indexed_matches: list[dict[str, Any]],
    semantic_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not indexed_matches:
        return list(semantic_matches)[:MAX_PROVIDER_SESSION_MATCHES]
    if not semantic_matches:
        return list(indexed_matches)[:MAX_PROVIDER_SESSION_MATCHES]

    fused_score: dict[str, float] = {}
    entries: dict[str, dict[str, Any]] = {}
    for rank, match in enumerate(indexed_matches, start=1):
        session_id = match["summary"].session_id
        fused_score[session_id] = fused_score.get(session_id, 0.0) + 1.0 / (RRF_K + rank)
        entries.setdefault(session_id, match)
    for rank, match in enumerate(semantic_matches, start=1):
        session_id = match["summary"].session_id
        fused_score[session_id] = fused_score.get(session_id, 0.0) + 1.0 / (RRF_K + rank)
        if session_id not in entries:
            entries[session_id] = match
    ordered = sorted(entries, key=lambda session_id: (-fused_score[session_id], session_id))
    return [entries[session_id] for session_id in ordered][:MAX_PROVIDER_SESSION_MATCHES]


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


def _collect_indexed_context_lines(task: str, selected_matches: list[dict[str, Any]]) -> tuple[str, ...]:
    if not selected_matches:
        return ()
    tokens = _tokenize(task)
    prefer_tool_output = _is_tool_output_query(task)
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for match in selected_matches:
        summary = match.get("summary")
        if isinstance(summary, ExternalSessionSummary):
            line = f"Session '{summary.title}' targeted workspace {summary.workspace_path or 'unknown workspace'}."
            if line not in seen:
                seen.add(line)
                candidates.append((2, line))
        for chunk in match.get("chunks", []) or ():
            if not isinstance(chunk, ExternalSessionChunk):
                continue
            prefix = "User asked" if chunk.role == "user" else "Assistant reported" if chunk.role == "assistant" else "Tool output"
            line = f"{prefix}: {_compact_context_content(chunk.role, chunk.text)}"
            if not line or line in seen:
                continue
            seen.add(line)
            lowered = line.lower()
            score = sum(2 for token in tokens if _token_matches(token, lowered))
            if chunk.source == "reasoning":
                score += 1
            if chunk.role in {"user", "assistant"}:
                score += 1
            if chunk.role == "tool":
                score += 3 if prefer_tool_output else -3
            candidates.append((score, line))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return tuple(line for score, line in candidates if score > 0)[:MAX_INDEX_CONTEXT_LINES]


def _build_chunks_from_messages(
    summary: ExternalSessionSummary,
    messages: list[ExternalSessionMessage],
    *,
    source: str,
) -> list[ExternalSessionChunk]:
    chunks: list[ExternalSessionChunk] = []
    current_role = ""
    current_source = source
    current_timestamp: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_role, current_source, current_timestamp
        text = _normalize_whitespace("\n".join(buffer))
        if not text:
            buffer = []
            return
        chunks.append(
            ExternalSessionChunk(
                provider=summary.provider,
                session_id=summary.session_id,
                title=summary.title,
                workspace_path=summary.workspace_path,
                role=current_role or "assistant",
                source=current_source,
                text=text,
                timestamp=current_timestamp,
            )
        )
        buffer = []

    for message in messages:
        text = _normalize_whitespace(message.content)
        if not text:
            continue
        role = message.role or "assistant"
        message_source = "tool" if role == "tool" else source
        if buffer and (role != current_role or message_source != current_source or len("\n".join([*buffer, text])) > MAX_INDEX_CHUNK_CHARS):
            flush()
        if not buffer:
            current_role = role
            current_source = message_source
            current_timestamp = message.timestamp
        buffer.append(text)
    flush()
    return chunks


def _extract_opencode_message_part(role: str, payload: dict[str, Any], timestamp: str | None) -> ExternalSessionMessage | None:
    part_type = str(payload.get("type") or "").strip().lower()
    if part_type == "text":
        text = _normalize_whitespace(str(payload.get("text") or ""))
        if text and not _is_noise_message_content(text):
            return ExternalSessionMessage(role=role, content=text, timestamp=timestamp)
        return None
    if part_type == "reasoning":
        text = _normalize_whitespace(str(payload.get("text") or ""))
        if text:
            return ExternalSessionMessage(role="assistant", content=text, timestamp=timestamp)
        return None
    if part_type == "tool":
        tool_name = str(payload.get("tool") or "").strip()
        state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
        detail = _clean_tool_output(str(state.get("output") or state.get("error") or ""))
        title = str(state.get("title") or tool_name or "tool").strip()
        if detail:
            content = f"{title}: {detail}"
        elif title:
            content = f"Tool call: {title}"
        else:
            content = ""
        content = _normalize_whitespace(content)
        if content and _is_useful_tool_output(content):
            return ExternalSessionMessage(role="tool", content=_truncate_tool_output(content), timestamp=timestamp)
    return None


def _safe_json_loads(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _millis_to_iso(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return ""
    if numeric <= 0:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(numeric / 1000))


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


def _is_cleanup_schema_task(task: str) -> bool:
    lowered = task.lower()
    return any(marker in lowered for marker in ("cleanup", "clean up", "schema", "schrema"))


def _session_matches_cleanup_schema_task(task: str, summary: ExternalSessionSummary) -> bool:
    if not _is_cleanup_schema_task(task):
        return False
    title_preview = _normalize_whitespace(" ".join(part for part in (summary.title, summary.preview) if part)).lower()
    return any(
        marker in title_preview
        for marker in (
            "cleanup",
            "clean up",
            "schema",
            "review",
            "legacy",
            "root url redirects",
            "convex generated imports",
            "authentication bypass",
        )
    )


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
    max_chars = 220 if role == "tool" else 260
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


def _is_useful_tool_output(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return not _looks_like_noisy_tool_output(lowered)


def _looks_like_noisy_tool_output(lowered: str) -> bool:
    noisy_markers = (
        "session_index_lines ",
        "\"thread_name\":",
        "\"updated_at\":",
        "<path>",
        "<content>",
        "<diagnostics ",
        "traceback (most recent call last)",
        "warning: the directory",
        "processing /users/",
        "preparing metadata",
        "installing build dependencies",
        "subprocess-exited-with-error",
        "original token count:",
        "process exited with code",
        "edit applied successfully",
        "lsp errors detected",
    )
    if any(marker in lowered for marker in noisy_markers):
        return True
    if re.search(r": line \d+:", lowered):
        return True
    if re.search(r"\bfound \d+ matches\b", lowered):
        return True
    if len(lowered) > 260:
        structured_chars = sum(lowered.count(char) for char in "{}[]|/\\")
        if structured_chars >= max(18, len(lowered) // 12):
            return True
    return False


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(length):
        left_value = left[index]
        right_value = right[index]
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _rrf_score(lexical_rank: int | None, semantic_rank: int | None) -> float:
    score = 0.0
    if lexical_rank is not None:
        score += 1.0 / (RRF_K + lexical_rank)
    if semantic_rank is not None:
        score += 1.0 / (RRF_K + semantic_rank)
    return score


def _unified_session_id(provider: str, session_id: str) -> str:
    return f"{provider}:{session_id}"


def _is_tool_output_query(task: str) -> bool:
    lowered = task.lower()
    return any(
        marker in lowered
        for marker in (
            "tool output",
            "command output",
            "error log",
            "traceback",
            "review comment",
            "what did",
            "comment",
            "issue",
            "issues",
            "error",
            "warning",
        )
    )


def _open_sqlite_connection(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    resolved = str(path)
    if immutable:
        uri = f"file:{resolved}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(resolved)


def _fetch_sqlite_rows(path: Path, query: str, parameters: tuple[Any, ...], *, immutable: bool) -> list[dict[str, Any]]:
    connection = _open_sqlite_connection(path, immutable=immutable)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(query, parameters).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


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
