from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.ai.ollama_backend import PLAN_BLUEPRINT_JSON_SCHEMA, TOOL_OR_FINAL_JSON_SCHEMA
from core.tools.base import BaseTool

DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_LLAMACPP_MODEL = ""


class LlamaCppAICore:
    provider_label = "llama.cpp"
    supports_tool_calls = True

    def __init__(
        self,
        *,
        workspace_path: str,
        model: str | None = None,
        base_url: str | None = None,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.base_url = (
            base_url
            or os.getenv("LLAMA_CPP_BASE_URL")
            or os.getenv("DEVENV_LLAMACPP_BASE_URL")
            or DEFAULT_LLAMACPP_BASE_URL
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("DEVENV_LLAMACPP_MODEL")
            or DEFAULT_LLAMACPP_MODEL
        ).strip()
        self.system_instructions = system_instructions.strip()
        self.timeout_seconds = timeout_seconds
        self.performance_mode = "medium"
        self.last_backend_used = "llama_cpp"
        self.last_backend_reason = ""
        self.last_backend_fallback = ""
        self.last_error = ""
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def set_model(self, model: str) -> None:
        self.model = model.strip()

    def set_performance_mode(self, performance_mode: str) -> None:
        cleaned = str(performance_mode or "").strip().lower()
        if cleaned in {"low", "medium", "high"}:
            self.performance_mode = cleaned

    def reset_session(self) -> None:
        return None

    def abort(self) -> bool:
        return False

    def status(self) -> AIBackendStatus:
        try:
            models = self.list_models()
        except RuntimeError as exc:
            detail = str(exc).strip() or f"llama.cpp server is not running at {self.base_url}."
            if self.last_error:
                detail = self.last_error
            return AIBackendStatus(
                name="llama_cpp",
                available=False,
                enabled=True,
                model=self.model,
                detail=detail,
                supports_tool_calls=True,
                metadata={
                    "runtime": "llama.cpp",
                    "transport": "openai_compatible_http",
                    "base_url": self.base_url,
                    "models": [],
                    "last_error": self.last_error or detail,
                },
            )

        available = True
        if self.model and self.model not in models:
            detail = (
                f"Selected llama.cpp model `{self.model}` is not available. "
                f"Available models: {', '.join(models[:6]) or 'none'}."
            )
            available = False
        elif self.model:
            detail = f"llama.cpp reachable at {self.base_url} with model `{self.model}`."
        elif models:
            detail = f"llama.cpp reachable at {self.base_url} with models: {', '.join(models[:4])}."
        else:
            detail = f"llama.cpp reachable at {self.base_url}, but no models were reported."
            available = False

        if self.last_error and available:
            self.last_error = ""
        return AIBackendStatus(
            name="llama_cpp",
            available=available,
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "runtime": "llama.cpp",
                "transport": "openai_compatible_http",
                "base_url": self.base_url,
                "models": models,
                "last_error": self.last_error,
            },
        )

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/v1/models")
        models: list[str] = []
        for item in payload.get("data", []) or []:
            name = str((item or {}).get("id") or "").strip()
            if name and name not in models:
                models.append(name)
        if self.model and self.model not in models:
            models.insert(0, self.model)
        return models

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        resolved_tool_names = [name for name in (tool_names or ()) if name in self._tools]
        selected_model = self.model.strip()
        if not selected_model:
            models = [name for name in self.list_models() if name]
            if not models:
                self.last_error = f"llama.cpp is reachable at {self.base_url}, but no local models were reported."
                raise RuntimeError(self.last_error)
            selected_model = models[0]
            self.model = selected_model

        prompt_messages = self._compile_messages(
            messages=messages,
            memory_context=memory_context,
            tool_names=resolved_tool_names,
        )
        schema: dict[str, Any] | None = None
        planner_json_mode = any(
            "PLANNER_OUTPUT_MODE: blueprint_json" in str(message.get("content") or "")
            for message in messages
        )
        if planner_json_mode:
            schema = PLAN_BLUEPRINT_JSON_SCHEMA
        elif resolved_tool_names:
            schema = TOOL_OR_FINAL_JSON_SCHEMA

        body = self._chat_request_body(
            model=selected_model,
            messages=prompt_messages,
            temperature=temperature,
            schema=schema,
        )
        try:
            payload = self._request_json("POST", "/v1/chat/completions", body)
        except RuntimeError as exc:
            if schema is not None and _looks_like_response_format_failure(str(exc)):
                self.last_backend_fallback = "llama.cpp rejected structured format; retried with json_object mode."
                body = self._chat_request_body(
                    model=selected_model,
                    messages=prompt_messages,
                    temperature=temperature,
                    schema="json_object",
                )
                payload = self._request_json("POST", "/v1/chat/completions", body)
            else:
                raise

        choice = ((payload.get("choices") or [{}])[0] or {})
        message_payload = choice.get("message") or {}
        content = str(message_payload.get("content") or "").strip()
        usage = _extract_llamacpp_usage(payload)
        self.last_backend_used = "llama_cpp"
        self.last_backend_reason = f"llama.cpp model {selected_model} handled the turn."
        self.last_error = ""
        if resolved_tool_names:
            parsed = _parse_structured_llamacpp_response(content, resolved_tool_names)
            if parsed is not None:
                return parsed
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage=usage,
            backend="llama_cpp",
            metadata={
                "runtime": "llama.cpp",
                "transport": "openai_compatible_http",
                "base_url": self.base_url,
                "model": selected_model,
                "performance_mode": self.performance_mode,
            },
        )

    def _compile_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        memory_context: str | None,
        tool_names: list[str],
    ) -> list[dict[str, str]]:
        compiled: list[dict[str, str]] = []
        system_sections: list[str] = []
        if self.system_instructions:
            system_sections.append(self.system_instructions)
        if memory_context and memory_context.strip():
            system_sections.append(f"## Memory\n{memory_context.strip()}")
        if tool_names:
            tool_payload = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema(),
                }
                for tool_name in tool_names
                for tool in [self._tools[tool_name]]
            ]
            system_sections.extend(
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
                ]
            )
        if system_sections:
            compiled.append({"role": "system", "content": "\n\n".join(system_sections).strip()})
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(message.get("content") or "").strip()
            if content:
                compiled.append({"role": role, "content": content})
        return compiled

    def _chat_request_body(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        schema: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if schema == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif isinstance(schema, dict):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "devenv_response",
                    "schema": schema,
                },
            }
        return body

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            self.last_error = f"llama.cpp request failed: {detail or exc.reason or exc.code}"
            raise RuntimeError(self.last_error) from exc
        except OSError as exc:
            self.last_error = f"llama.cpp is not running at {self.base_url}: {exc}"
            raise RuntimeError(self.last_error) from exc
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            self.last_error = "llama.cpp returned invalid JSON."
            raise RuntimeError(self.last_error) from exc


def _extract_llamacpp_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _parse_structured_llamacpp_response(
    content: str, allowed_tools: list[str]
) -> AIResponse | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    response_type = str(payload.get("type") or "").strip().lower()
    if response_type == "tool_call":
        tool_name = str(payload.get("tool_name") or "").strip()
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        if tool_name and tool_name in allowed_tools:
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCallRequest(
                        call_id="llama_cpp-tool-call",
                        tool_name=tool_name,
                        arguments=arguments,
                    ),
                ),
                usage={},
                backend="llama_cpp",
            )
    if response_type == "final":
        final_content = str(payload.get("content") or "").strip()
        return AIResponse(
            content=final_content,
            finish_reason="stop",
            usage={},
            backend="llama_cpp",
        )
    return None


def _looks_like_response_format_failure(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return any(
        token in lowered
        for token in (
            "response_format",
            "json_schema",
            "json schema",
            "structured",
            "schema",
        )
    )
