from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.ai.engine import AICore
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.tools.base import BaseTool


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
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.executable = executable
        self.model = model or os.getenv("OPENCODE_MODEL") or ""
        self.system_instructions = system_instructions.strip()
        self.last_backend_used = "opencode"
        self.last_backend_reason = ""
        self.last_error = ""
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def status(self) -> AIBackendStatus:
        executable_path = shutil.which(self.executable)
        detail = "Installed" if executable_path else "CLI not found on PATH"
        if self.last_error:
            detail = self.last_error
        return AIBackendStatus(
            name="opencode",
            available=bool(executable_path),
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
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
        resolved_tool_names = [name for name in (tool_names or ()) if name in self._tools]
        prompt = self._compile_prompt(messages, memory_context) if not resolved_tool_names else self._compile_tool_prompt(
            messages,
            memory_context,
            resolved_tool_names,
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
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit status {completed.returncode}"
            self.last_error = f"OpenCode CLI failed: {detail}"
            raise RuntimeError(self.last_error)

        content, usage, tool_calls = _parse_opencode_output(completed.stdout, allowed_tools=resolved_tool_names)
        self.last_backend_used = "opencode"
        self.last_backend_reason = "OpenCode handled the turn directly."
        self.last_error = ""
        finish_reason = "tool_calls" if tool_calls else "stop"
        return AIResponse(content=content, tool_calls=tool_calls, finish_reason=finish_reason, usage=usage)

    def _compile_prompt(self, messages: list[dict[str, Any]], memory_context: str | None) -> str:
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
                    "If a tool is needed, return {\"type\":\"tool_call\",\"tool_name\":\"<tool>\",\"arguments\":{...}}. "
                    "If no tool is needed, return {\"type\":\"final\",\"content\":\"<response>\"}. "
                    "Use only one tool call at a time and only from the listed tools."
                ),
            ]
        )
        return "\n\n".join(section for section in sections if section).strip()


class RoutingAICore:
    provider_label = "Groq"

    def __init__(
        self,
        *,
        workspace_path: str,
        groq_ai: AICore | None = None,
        opencode_ai: OpenCodeAICore | None = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.groq_ai = groq_ai or AICore()
        self.opencode_ai = opencode_ai or OpenCodeAICore(
            workspace_path=self.workspace_path,
            system_instructions=getattr(self.groq_ai, "system_instructions", ""),
        )
        self.model = self.groq_ai.model
        self.preferred_backend = "auto"
        self.opencode_enabled = False
        self.last_backend_used = "groq"
        self.last_backend_reason = "Groq handled the turn."
        self.last_backend_fallback = ""

    def register_tool(self, tool: BaseTool) -> None:
        self.groq_ai.register_tool(tool)
        self.opencode_ai.register_tool(tool)

    def status(self) -> dict[str, AIBackendStatus]:
        groq_available = bool(getattr(self.groq_ai, "api_key", None) or os.getenv("GROQ_API_KEY"))
        groq_status = AIBackendStatus(
            name="groq",
            available=groq_available,
            enabled=True,
            model=getattr(self.groq_ai, "model", ""),
            detail="Configured" if groq_available else "Missing GROQ_API_KEY",
            supports_tool_calls=True,
        )
        opencode_status = self.opencode_ai.status()
        return {"groq": groq_status, "opencode": opencode_status}

    def set_model(self, model: str) -> None:
        cleaned = model.strip()
        self.model = cleaned
        self.groq_ai.model = cleaned
        self.opencode_ai.model = cleaned

    def set_backend_preference(self, backend: str, *, opencode_enabled: bool) -> None:
        normalized = backend if backend in {"auto", "groq", "opencode"} else "auto"
        self.preferred_backend = normalized
        self.opencode_enabled = opencode_enabled

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        statuses = self.status()
        wants_opencode = self.opencode_enabled and self.preferred_backend in {"auto", "opencode"}
        if wants_opencode and statuses["opencode"].available:
            try:
                response = self.opencode_ai.chat(
                    messages=messages,
                    memory_context=memory_context,
                    temperature=temperature,
                    tool_names=tool_names,
                )
                self.last_backend_used = "opencode"
                self.last_backend_reason = self.opencode_ai.last_backend_reason
                self.last_backend_fallback = ""
                return response
            except RuntimeError as exc:
                self.last_backend_fallback = str(exc)

        response = self.groq_ai.chat(
            messages=messages,
            memory_context=memory_context,
            temperature=temperature,
            tool_names=tool_names,
        )
        self.last_backend_used = "groq"
        self.last_backend_reason = "Groq handled the turn."
        return response


def _parse_opencode_output(stdout: str, *, allowed_tools: list[str] | None = None) -> tuple[str, dict[str, int], tuple[ToolCallRequest, ...]]:
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


def _extract_tool_call_from_text(stdout: str, allowed_tools: list[str]) -> ToolCallRequest | None:
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


def _extract_tool_call_from_payload(payload: Any, allowed_tools: list[str]) -> ToolCallRequest | None:
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("type") or "").strip().lower()
    tool_name = payload.get("tool_name") or payload.get("tool")
    arguments = payload.get("arguments")
    if payload_type not in {"tool_call", "tool"} or not isinstance(tool_name, str) or tool_name not in allowed_tools:
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
