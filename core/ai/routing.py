from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.ai.codex_backend import CodexAICore
from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.ai.ollama_backend import OllamaAICore
from core.ai.opencode_client import (
    OpenCodeClient,
    OpenCodeClientError,
    OpenCodeServerManager,
    default_opencode_server_config,
)
from core.tools.base import BaseTool

DEFAULT_OPENCODE_MODEL = "opencode/claude-sonnet-4"


class OpenCodeAICore:
    provider_label = "OpenCode CLI"
    supports_tool_calls = True

    def __init__(
        self,
        *,
        workspace_path: str,
        model: str | None = None,
        executable: str = "opencode",
        system_instructions: str = "",
        server_manager: OpenCodeServerManager | None = None,
        client: OpenCodeClient | None = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.executable = executable
        self.model = model or os.getenv("OPENCODE_MODEL") or DEFAULT_OPENCODE_MODEL
        self.system_instructions = system_instructions.strip()
        self.last_backend_used = "opencode"
        self.last_backend_reason = ""
        self.last_backend_fallback = ""
        self.last_error = ""
        self._tools: dict[str, BaseTool] = {}
        self.server_manager = server_manager or OpenCodeServerManager(
            config=default_opencode_server_config(),
            executable=self.executable,
        )
        self.client = client or OpenCodeClient(self.server_manager.config)
        self._session_id: str | None = None
        self._synced_message_count = 0
        self._structured_output_supported: bool | None = None
        self._transport_backoff_until = 0.0
        self._transport_backoff_reason = ""

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def status(self) -> AIBackendStatus:
        executable_path = shutil.which(self.executable)
        detail = "Installed" if executable_path else "CLI not found on PATH"
        if self.last_error:
            detail = self.last_error
        server_status = self.server_manager.inspect()
        return AIBackendStatus(
            name="opencode",
            available=bool(executable_path),
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "server": server_status.to_metadata(),
                "transport": "legacy_cli"
                if self._should_use_legacy_cli()
                else "server",
                "session_id": self._session_id or "",
                "synced_message_count": self._synced_message_count,
                "structured_output_supported": self._structured_output_supported,
                "transport_backoff_active": self._transport_backoff_until
                > time.monotonic(),
                "last_error": self.last_error,
            },
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        del temperature
        if not shutil.which(self.executable):
            self.last_error = "OpenCode CLI is not installed."
            raise RuntimeError(self.last_error)
        if self._should_use_legacy_cli():
            return self._legacy_cli_chat(
                messages=messages, memory_context=memory_context, tool_names=tool_names
            )
        self._raise_if_transport_backoff_active()
        return self._server_chat(
            messages=messages, memory_context=memory_context, tool_names=tool_names
        )

    def reset_session(self) -> None:
        self._session_id = None
        self._synced_message_count = 0

    def abort(self) -> bool:
        if not self._session_id:
            return False
        try:
            aborted = self.client.abort_session(self._session_id)
        except OpenCodeClientError as exc:
            self.last_error = f"OpenCode server abort failed: {exc}"
            raise RuntimeError(self.last_error) from exc
        if aborted:
            self.reset_session()
        return aborted

    def _should_use_legacy_cli(self) -> bool:
        explicit = os.getenv("DEVENV_OPENCODE_USE_LEGACY_CLI", "").strip().lower()
        return explicit in {"1", "true", "yes", "on"}

    def _server_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        memory_context: str | None,
        tool_names: Iterable[str] | None,
    ) -> AIResponse:
        resolved_tool_names = [
            name for name in (tool_names or ()) if name in self._tools
        ]
        prompt_messages = messages[self._synced_message_count :] or messages[-1:]
        prompt = (
            self._compile_prompt(prompt_messages, memory_context)
            if not resolved_tool_names
            else self._compile_tool_prompt(
                prompt_messages,
                memory_context,
                resolved_tool_names,
            )
        )
        self.last_backend_fallback = ""
        try:
            session_id = self._ensure_session()
            response = self._send_server_message(
                session_id, prompt=prompt, resolved_tool_names=resolved_tool_names
            )
        except OpenCodeClientError as exc:
            if _should_fallback_to_legacy_cli(exc):
                server_failure = str(exc).strip() or "unknown server failure"
                fallback_note = f"OpenCode server failed ({server_failure}); fell back to CLI transport."
                self.reset_session()
                cli_response = self._legacy_cli_chat(
                    messages=messages,
                    memory_context=memory_context,
                    tool_names=resolved_tool_names,
                )
                self.last_backend_used = "opencode"
                self.last_backend_reason = (
                    "OpenCode CLI fallback handled the turn after server failure."
                )
                self.last_backend_fallback = fallback_note
                self.last_error = ""
                return cli_response
            if _is_transport_backoff_worthy(exc):
                self._transport_backoff_until = (
                    time.monotonic() + _transport_backoff_seconds()
                )
                self._transport_backoff_reason = (
                    str(exc).strip() or "unknown transport failure"
                )
            self.last_error = f"OpenCode server failed: {exc}"
            raise RuntimeError(self.last_error) from exc
        self._synced_message_count = len(messages)
        self._transport_backoff_until = 0.0
        self._transport_backoff_reason = ""
        content, usage, tool_calls = _parse_server_message(
            response, allowed_tools=resolved_tool_names
        )
        self.last_backend_used = "opencode"
        self.last_backend_reason = (
            f"OpenCode server session {session_id} handled the turn."
        )
        self.last_error = ""
        finish_reason = "tool_calls" if tool_calls else "stop"
        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            backend="opencode",
            metadata={
                "transport": "server",
                "session_id": session_id,
            },
        )

    def _legacy_cli_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        memory_context: str | None,
        tool_names: Iterable[str] | None,
    ) -> AIResponse:
        resolved_tool_names = [
            name for name in (tool_names or ()) if name in self._tools
        ]
        prompt = (
            self._compile_prompt(messages, memory_context)
            if not resolved_tool_names
            else self._compile_tool_prompt(
                messages,
                memory_context,
                resolved_tool_names,
            )
        )
        command = [
            self.executable,
            "run",
            "--format",
            "json",
            "--dir",
            self.workspace_path,
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append(prompt)

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.workspace_path,
            )
        except OSError as exc:
            self.last_error = f"OpenCode CLI failed to start: {exc}"
            raise RuntimeError(self.last_error) from exc

        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit status {completed.returncode}"
            )
            self.last_error = f"OpenCode CLI failed: {detail}"
            raise RuntimeError(self.last_error)

        content, usage, tool_calls = _parse_opencode_output(
            completed.stdout, allowed_tools=resolved_tool_names
        )
        self.last_backend_used = "opencode"
        self.last_backend_reason = "OpenCode handled the turn directly."
        self.last_error = ""
        finish_reason = "tool_calls" if tool_calls else "stop"
        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            backend="opencode",
            metadata={
                "transport": "legacy_cli",
            },
        )

    def _ensure_session(self) -> str:
        if self._session_id:
            return self._session_id
        self.server_manager.ensure_server()
        title = f"Devenv: {Path(self.workspace_path).name or self.workspace_path}"
        session = self.client.create_session(title=title)
        self._session_id = session.session_id
        self._synced_message_count = 0
        return self._session_id

    def _send_server_message(
        self, session_id: str, *, prompt: str, resolved_tool_names: list[str]
    ):
        output_format = (
            None
            if self._structured_output_supported is False
            else _opencode_output_format(resolved_tool_names)
        )
        try:
            response = self._send_server_message_once(
                session_id,
                prompt=prompt,
                output_format=output_format,
            )
            if output_format is not None:
                self._structured_output_supported = True
            return response
        except OpenCodeClientError as exc:
            if _is_structured_output_retryable(exc):
                self._structured_output_supported = False
                self.last_backend_fallback = "OpenCode rejected structured output; retried without output schema."
                return self._send_server_message_once(
                    session_id,
                    prompt=prompt,
                    output_format=None,
                )
            if not _is_recoverable_session_error(exc):
                raise
            self.reset_session()
            recovered_session_id = self._ensure_session()
            try:
                response = self._send_server_message_once(
                    recovered_session_id,
                    prompt=prompt,
                    output_format=output_format,
                )
                if output_format is not None:
                    self._structured_output_supported = True
                return response
            except OpenCodeClientError as retry_exc:
                if _is_structured_output_retryable(retry_exc):
                    self._structured_output_supported = False
                    self.last_backend_fallback = "OpenCode rejected structured output after session recovery; retried without output schema."
                    return self._send_server_message_once(
                        recovered_session_id,
                        prompt=prompt,
                        output_format=None,
                    )
                raise

    def _raise_if_transport_backoff_active(self) -> None:
        if self._transport_backoff_until <= time.monotonic():
            return
        remaining = max(int(self._transport_backoff_until - time.monotonic()), 1)
        detail = self._transport_backoff_reason or "recent transport failure"
        self.last_error = f"OpenCode server failed: recent transport failure cached for {remaining}s ({detail})."
        raise RuntimeError(self.last_error)

    def _send_server_message_once(
        self,
        session_id: str,
        *,
        prompt: str,
        output_format: dict[str, Any] | None,
    ):
        return self.client.send_message(
            session_id,
            parts=[{"type": "text", "text": prompt}],
            output_format=output_format,
        )

    def _compile_prompt(
        self, messages: list[dict[str, Any]], memory_context: str | None
    ) -> str:
        sections: list[str] = []
        if self.system_instructions:
            sections.extend(["## System", self.system_instructions])
        if memory_context and memory_context.strip():
            sections.extend(["## Memory", memory_context.strip()])
        for message in messages:
            role = str(message.get("role") or "user").upper()
            content = str(message.get("content") or "").strip()
            if content:
                sections.append(f"{role}: {content}")
        return "\n\n".join(sections).strip()

    def _compile_tool_prompt(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None,
        tool_names: list[str],
    ) -> str:
        sections = [self._compile_prompt(messages, memory_context)]
        tool_payload = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema(),
            }
            for tool_name in tool_names
            for tool in [self._tools[tool_name]]
        ]
        sections.extend(
            [
                "## Available Tools",
                json.dumps(tool_payload, separators=(",", ":"), sort_keys=True),
                "## Required Response Format",
                (
                    "Return exactly one JSON object and nothing else. "
                    'If a tool is needed, return {"type":"tool_call","tool_name":"<tool>","arguments":{...}}. '
                    'If no tool is needed, return {"type":"final","content":"<response>"}. '
                    "Use only one tool call at a time and only from the listed tools."
                ),
                "## Tool Use Policy",
                (
                    "Prefer memory and local workspace tools first. "
                    "Use web_search for current or time-sensitive facts, public officeholders, live package docs, or web pages outside the workspace. "
                    "If the user explicitly asks to search, browse, google, or look something up, call web_search before answering. "
                    "Use knowledge_search when the user wants external references, GitHub repos, forum threads, videos, or broader resource gathering around a topic or feature. "
                    "If the intended search target is unclear, ask one concise follow-up question instead of guessing. "
                    "For large files such as AGENTS.md, guidelines, or generated docs, do not dump the whole file; inspect only relevant sections and summarize them. "
                    "Prefer list_directory, search_text, inspect_symbols, or peek_lines before reading a large file end-to-end. "
                    "Keep final answers short unless the user explicitly asks for a detailed breakdown."
                ),
            ]
        )
        return "\n\n".join(section for section in sections if section).strip()


class RoutingAICore:
    provider_label = "OpenCode CLI"

    def __init__(
        self,
        *,
        workspace_path: str,
        opencode_ai: OpenCodeAICore | None = None,
        ollama_ai: OllamaAICore | None = None,
        codex_ai: Any | None = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.opencode_ai = opencode_ai or OpenCodeAICore(
            workspace_path=self.workspace_path,
            system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        self.ollama_ai = ollama_ai or OllamaAICore(
            workspace_path=self.workspace_path,
            system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        self.codex_ai = codex_ai or CodexAICore(
            workspace_path=self.workspace_path,
            system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
        )
        self.model = self.opencode_ai.model
        self.preferred_backend = "opencode"
        self.opencode_enabled = False
        self.ollama_enabled = False
        self.codex_enabled = False
        self.last_backend_used = "opencode"
        self.last_backend_reason = "OpenCode handled the turn."
        self.last_backend_fallback = ""

    def register_tool(self, tool: BaseTool) -> None:
        self.opencode_ai.register_tool(tool)
        self.ollama_ai.register_tool(tool)
        if self.codex_ai is not None and hasattr(self.codex_ai, "register_tool"):
            self.codex_ai.register_tool(tool)

    def status(self) -> dict[str, AIBackendStatus]:
        opencode_status = self.opencode_ai.status()
        statuses = {
            "opencode": opencode_status,
            "ollama": self.ollama_ai.status(),
        }
        if self.codex_ai is not None and hasattr(self.codex_ai, "status"):
            statuses["codex"] = self.codex_ai.status()
        return statuses

    def set_model(self, model: str) -> None:
        cleaned = model.strip()
        self.model = cleaned
        if self.preferred_backend == "codex" and self.codex_ai is not None and hasattr(self.codex_ai, "set_model"):
            self.codex_ai.set_model(cleaned)
        elif self.preferred_backend == "ollama":
            self.ollama_ai.set_model(cleaned)
        else:
            self.opencode_ai.model = cleaned

    def set_backend_model(self, backend: str, model: str) -> None:
        cleaned_backend = str(backend or "").strip().lower()
        cleaned_model = model.strip()
        if cleaned_backend == "opencode":
            self.opencode_ai.model = cleaned_model
        elif cleaned_backend == "ollama":
            self.ollama_ai.set_model(cleaned_model)
        elif cleaned_backend == "codex" and self.codex_ai is not None and hasattr(self.codex_ai, "set_model"):
            self.codex_ai.set_model(cleaned_model)
        else:
            raise ValueError("backend must be one of: opencode, ollama, codex")
        if self.preferred_backend == cleaned_backend:
            self.model = cleaned_model

    def set_backend_preference(
        self, backend: str, *, opencode_enabled: bool, ollama_enabled: bool = False, codex_enabled: bool = False
    ) -> None:
        cleaned = str(backend or "opencode").strip().lower() or "opencode"
        if cleaned not in {"opencode", "ollama", "codex"}:
            raise ValueError("backend must be one of: opencode, ollama, codex")
        self.preferred_backend = cleaned
        self.opencode_enabled = opencode_enabled
        self.ollama_enabled = ollama_enabled
        self.codex_enabled = codex_enabled

    def reset_session(self) -> None:
        self.opencode_ai.reset_session()
        if hasattr(self.ollama_ai, "reset_session"):
            self.ollama_ai.reset_session()
        if self.codex_ai is not None and hasattr(self.codex_ai, "reset_session"):
            self.codex_ai.reset_session()

    def abort(self) -> bool:
        aborted = self.opencode_ai.abort()
        if self.codex_ai is not None and hasattr(self.codex_ai, "abort"):
            aborted = bool(self.codex_ai.abort()) or aborted
        return aborted

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        if self.preferred_backend == "codex":
            if self.codex_ai is None:
                self.last_backend_fallback = "Codex backend is not configured."
                raise RuntimeError(self.last_backend_fallback)
            if not self.codex_enabled:
                self.last_backend_fallback = (
                    "Codex backend access has not been granted."
                )
                raise RuntimeError(self.last_backend_fallback)
            response = self.codex_ai.chat(
                messages=messages,
                memory_context=memory_context,
                temperature=temperature,
                tool_names=tool_names,
            )
            self.last_backend_used = "codex"
            self.last_backend_reason = getattr(
                self.codex_ai, "last_backend_reason", "Codex handled the turn."
            )
            self.last_backend_fallback = ""
            self.model = getattr(self.codex_ai, "model", self.model)
            return response
        if self.preferred_backend == "ollama":
            if not self.ollama_enabled:
                self.last_backend_fallback = "Ollama backend access has not been granted."
                raise RuntimeError(self.last_backend_fallback)
            response = self.ollama_ai.chat(
                messages=messages,
                memory_context=memory_context,
                temperature=temperature,
                tool_names=tool_names,
            )
            self.last_backend_used = "ollama"
            self.last_backend_reason = self.ollama_ai.last_backend_reason
            self.last_backend_fallback = ""
            self.model = self.ollama_ai.model
            return response
        if not self.opencode_enabled:
            self.last_backend_fallback = "OpenCode backend access has not been granted."
            raise RuntimeError(self.last_backend_fallback)
        response = self.opencode_ai.chat(
            messages=messages,
            memory_context=memory_context,
            temperature=temperature,
            tool_names=tool_names,
        )
        self.last_backend_used = "opencode"
        self.last_backend_reason = self.opencode_ai.last_backend_reason
        self.last_backend_fallback = ""
        self.model = self.opencode_ai.model
        return response


def _is_recoverable_session_error(exc: OpenCodeClientError) -> bool:
    if exc.status_code == 404:
        return True
    lowered = str(exc).lower()
    return (
        "unknown session" in lowered or "session" in lowered and "not found" in lowered
    )


def _is_structured_output_retryable(exc: OpenCodeClientError) -> bool:
    if exc.status_code != 400:
        return False
    detail = str(exc).lower()
    payload = exc.payload
    if isinstance(payload, dict):
        payload_text = json.dumps(payload, sort_keys=True).lower()
        detail = f"{detail} {payload_text}".strip()
    return any(
        token in detail
        for token in (
            "structured",
            "json_schema",
            "json schema",
            "outputformat",
            "output format",
            "response_format",
            "response format",
            "schema",
        )
    )


def _should_fallback_to_legacy_cli(exc: OpenCodeClientError) -> bool:
    explicit = os.getenv("DEVENV_OPENCODE_ALLOW_CLI_FALLBACK", "").strip().lower()
    if explicit == "0" or explicit == "false" or explicit == "off":
        return False
    if exc.status_code in {401, 403}:
        return False
    if exc.status_code is None:
        return True
    return exc.status_code >= 400


def _transport_backoff_seconds() -> float:
    raw = os.getenv("DEVENV_OPENCODE_TRANSPORT_BACKOFF_SECONDS", "").strip()
    if not raw:
        return 20.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 20.0


def _is_transport_backoff_worthy(exc: OpenCodeClientError) -> bool:
    if exc.status_code is not None:
        return exc.status_code >= 500
    lowered = str(exc).lower()
    return any(
        token in lowered
        for token in (
            "unable to reach opencode server",
            "timed out",
            "connection refused",
            "operation not permitted",
            "network is unreachable",
        )
    )


def _opencode_output_format(allowed_tools: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {
            "type": "string",
            "enum": ["final", "tool_call"] if allowed_tools else ["final"],
            "description": "Whether to answer directly or request exactly one Devenv tool call.",
        },
        "content": {
            "type": "string",
            "description": "Required when type is final.",
        },
        "tool_name": {
            "type": "string",
            "enum": allowed_tools or [""],
            "description": "Required when type is tool_call.",
        },
        "arguments": {
            "type": "object",
            "description": "Tool arguments when type is tool_call.",
        },
    }
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": ["type"],
            "additionalProperties": False,
        },
        "retryCount": 2,
    }


def _parse_server_message(
    message: Any, *, allowed_tools: list[str]
) -> tuple[str, dict[str, int], tuple[ToolCallRequest, ...]]:
    payload = message.raw if hasattr(message, "raw") else {}
    info = payload.get("info") if isinstance(payload, dict) else {}
    usage = _extract_opencode_usage(info) if isinstance(info, dict) else {}
    structured_output = None
    if hasattr(message, "structured_output"):
        structured_output = message.structured_output
    if isinstance(structured_output, dict):
        parsed_tool_call = _extract_tool_call_from_payload(
            structured_output, allowed_tools
        )
        if parsed_tool_call is not None:
            return "", usage, (parsed_tool_call,)
        content = str(structured_output.get("content") or "").strip()
        if content:
            return content, usage, ()
    part_lines: list[str] = []
    for part in getattr(message, "parts", ()) or ():
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            part_lines.append(text.strip())
    content = "\n".join(part_lines).strip()
    if content:
        parsed_tool_call = _extract_tool_call_from_text(content, allowed_tools)
        if parsed_tool_call is not None:
            return "", usage, (parsed_tool_call,)
    return content, usage, ()


def _parse_opencode_output(
    stdout: str, *, allowed_tools: list[str] | None = None
) -> tuple[str, dict[str, int], tuple[ToolCallRequest, ...]]:
    content_lines: list[str] = []
    usage: dict[str, int] = {}
    tool_calls: list[ToolCallRequest] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            content_lines.append(line)
            continue
        parsed_tool_call = _extract_tool_call_from_payload(payload, allowed_tools or [])
        if parsed_tool_call is not None:
            tool_calls.append(parsed_tool_call)
        extracted = _extract_opencode_content(payload)
        if extracted:
            content_lines.append(extracted)
        parsed_usage = _extract_opencode_usage(payload)
        for key, value in parsed_usage.items():
            usage[key] = usage.get(key, 0) + value

    content = "\n".join(part for part in content_lines if part).strip()
    if not content:
        content = stdout.strip()
    if not tool_calls:
        parsed_tool_call = _extract_tool_call_from_text(stdout, allowed_tools or [])
        if parsed_tool_call is not None:
            tool_calls.append(parsed_tool_call)
            content = ""
    if tool_calls:
        content = ""
    return content, usage, tuple(tool_calls[:1])


def _extract_tool_call_from_text(
    stdout: str, allowed_tools: list[str]
) -> ToolCallRequest | None:
    candidate = stdout.strip()
    if not candidate:
        return None
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return _extract_tool_call_from_payload(payload, allowed_tools)


def _extract_tool_call_from_payload(
    payload: Any, allowed_tools: list[str]
) -> ToolCallRequest | None:
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("type") or "").strip().lower()
    tool_name = payload.get("tool_name") or payload.get("tool")
    arguments = payload.get("arguments")
    if (
        payload_type not in {"tool_call", "tool"}
        or not isinstance(tool_name, str)
        or tool_name not in allowed_tools
    ):
        return None
    if not isinstance(arguments, dict):
        return None
    return ToolCallRequest(
        call_id=f"opencode-{uuid.uuid4().hex[:12]}",
        tool_name=tool_name,
        arguments=arguments,
    )


def _extract_opencode_content(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("content", "text", "message", "output", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if payload.get("type") == "message":
            content = payload.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                return "\n".join(parts).strip()
    if isinstance(payload, list):
        parts = [_extract_opencode_content(item) for item in payload]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _extract_opencode_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, int] = {}
    for key, value in raw_usage.items():
        if isinstance(value, int):
            usage[str(key)] = value
    return usage
