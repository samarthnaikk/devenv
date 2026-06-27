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
        if by_id:
            for session_id, record in by_id.items():
                source_path = self._find_session_file(session_id)
                preview = self._preview_from_history(session_id)
                summaries.append(
                    ExternalSessionSummary(
                        provider=self.name,
                        session_id=session_id,
                        title=str(record.get("thread_name") or "Untitled session"),
                        updated_at=str(record.get("updated_at") or ""),
                        workspace_path=None,
                        source_path=str(source_path) if source_path else None,
                        preview=preview,
                    )
                )
            summaries.sort(key=lambda item: item.updated_at, reverse=True)
            return summaries

        for session_file in sorted(self.root.glob(self.config.session_glob), reverse=True):
            detail = self._parse_session_file(session_file)
            summaries.append(detail.summary)
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
        pattern = f"*{session_id}*.jsonl"
        for path in self.root.glob(self.config.session_glob):
            if path.match(pattern) or session_id in path.name:
                return path
        return None

    def _preview_from_history(self, session_id: str) -> str:
        history_path = self.root / "history.jsonl"
        if not history_path.exists():
            return ""
        for raw_line in reversed(history_path.read_text(encoding="utf-8").splitlines()):
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("session_id")) != session_id:
                continue
            text = _normalize_whitespace(str(payload.get("text") or ""))
            if text:
                return text[:220]
        return ""

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
            if not text:
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
        self.providers = {
            config.provider: _provider_from_config(config)
            for config in self.provider_configs
        }

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
        if request.include_prior_context and provider_name:
            provider = self._get_provider(provider_name)
            selected_session_ids = request.session_ids or self._select_relevant_session_ids(provider, request.task)
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
            },
        )

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
        summaries = provider.list_sessions()
        if not summaries:
            return ()

        scored: list[tuple[int, ExternalSessionSummary]] = []
        workspace_name = Path(self.workspace_path).name.lower()
        workspace_path = self.workspace_path.lower()
        prompt_tokens = _tokenize(task)

        for summary in summaries:
            score = 0
            haystacks = [
                summary.title.lower(),
                summary.preview.lower(),
                (summary.workspace_path or "").lower(),
                os.path.basename(summary.source_path or "").lower(),
            ]
            for token in prompt_tokens:
                if any(token in haystack for haystack in haystacks):
                    score += 3
            session_workspace = (summary.workspace_path or "").lower()
            if session_workspace:
                if session_workspace == workspace_path:
                    score += 10
                elif workspace_name and workspace_name in session_workspace:
                    score += 6
            if "devenv" in " ".join(haystacks):
                score += 1
            scored.append((score, summary))

        scored.sort(key=lambda item: (-item[0], item[1].updated_at), reverse=False)
        selected = [summary.session_id for score, summary in scored if score > 0][:3]
        if not selected:
            selected = [summary.session_id for _score, summary in sorted(scored, key=lambda item: item[1].updated_at, reverse=True)[:2]]
        return tuple(selected)


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
        if text:
            messages.append(ExternalSessionMessage(role="assistant", content=text, timestamp=timestamp))
    elif row_type == "event_msg" and payload.get("type") == "task_complete":
        text = _normalize_whitespace(str(payload.get("last_agent_message") or ""))
        if text:
            messages.append(ExternalSessionMessage(role="assistant", content=text, timestamp=timestamp))
    elif row_type == "response_item" and payload.get("type") == "message":
        role = str(payload.get("role") or "assistant")
        for item in payload.get("content") or ():
            if not isinstance(item, dict):
                continue
            text = _normalize_whitespace(str(item.get("text") or ""))
            if text:
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
    return messages


def _collect_relevant_context_lines(
    task: str,
    details: list[ExternalSessionDetail],
    output_format: str,
) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()
    for detail in details:
        summary = detail.summary
        if summary.title.strip():
            candidates.append(f"Session '{summary.title}' targeted workspace {summary.workspace_path or 'unknown workspace'}.")
        for message in detail.messages:
            if message.role not in {"user", "assistant"}:
                continue
            content = _normalize_whitespace(message.content)
            if not content or content in seen:
                continue
            seen.add(content)
            prefix = "User asked:" if message.role == "user" else "Assistant reported:"
            candidates.append(f"{prefix} {content}")

    prompt_tokens = _tokenize(task)
    scored: list[tuple[int, str]] = []
    for line in candidates:
        lowered = line.lower()
        overlap = sum(2 for token in prompt_tokens if token in lowered)
        overlap += 1 if "workspace" in lowered else 0
        overlap += 1 if "file" in lowered or "frontend" in lowered or "backend" in lowered else 0
        scored.append((overlap, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [line for score, line in scored if score > 0][:MAX_CONTEXT_LINES]
    if not selected:
        selected = [line for _score, line in scored[:MAX_CONTEXT_LINES]]
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
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 3}


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
