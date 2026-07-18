from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.tools.base import BaseTool

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = ""
DEFAULT_OLLAMA_NUM_CTX = 4096
DEFAULT_OLLAMA_KEEP_ALIVE = "2m"
PLAN_BLUEPRINT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "description": {"type": "string"},
                    "level": {"type": "integer", "minimum": 0},
                },
                "required": ["task_id", "description", "level"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["tasks", "edges"],
}
TOOL_OR_FINAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tool_call", "final"],
        },
        "tool_name": {"type": "string"},
        "arguments": {"type": "object"},
        "content": {"type": "string"},
    },
    "required": ["type"],
}


class OllamaAICore:
    provider_label = "Ollama"
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
            or os.getenv("OLLAMA_HOST")
            or os.getenv("DEVENV_OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("DEVENV_OLLAMA_MODEL")
            or DEFAULT_OLLAMA_MODEL
        ).strip()
        self.system_instructions = system_instructions.strip()
        self.timeout_seconds = timeout_seconds
        self.performance_mode = "medium"
        self.last_backend_used = "ollama"
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
        profile = self._performance_profile()
        try:
            models = self.list_models()
        except RuntimeError as exc:
            detail = str(exc).strip() or f"Ollama is not running at {self.base_url}."
            if self.last_error:
                detail = self.last_error
            return AIBackendStatus(
                name="ollama",
                available=False,
                enabled=True,
                model=self.model,
                detail=detail,
                supports_tool_calls=True,
                metadata={
                    "runtime": "ollama",
                    "transport": "http_api",
                    "base_url": self.base_url,
                    "models": [],
                    "keep_alive": profile["keep_alive"],
                    "threads": profile["threads"],
                    "ctx_size": profile["ctx_size"],
                    "last_error": self.last_error or detail,
                },
            )

        detail = f"Ollama reachable at {self.base_url}."
        available = True
        if self.model and self.model not in models:
            detail = (
                f"Selected Ollama model `{self.model}` is not installed. "
                f"Available models: {', '.join(models[:6]) or 'none'}."
            )
            available = False
        elif self.model:
            detail = f"Ollama reachable at {self.base_url} with model `{self.model}`."
        elif models:
            detail = f"Ollama reachable at {self.base_url} with models: {', '.join(models[:4])}."
        else:
            detail = f"Ollama reachable at {self.base_url}, but no models are installed."
            available = False

        if self.last_error and available:
            self.last_error = ""
        return AIBackendStatus(
            name="ollama",
            available=available,
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "runtime": "ollama",
                "transport": "http_api",
                "base_url": self.base_url,
                "models": models,
                "keep_alive": profile["keep_alive"],
                "threads": profile["threads"],
                "ctx_size": profile["ctx_size"],
                "last_error": self.last_error,
            },
        )

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags")
        models: list[str] = []
        for item in payload.get("models", []) or []:
            name = str((item or {}).get("name") or "").strip()
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
                self.last_error = f"Ollama is reachable at {self.base_url}, but no local models are installed."
                raise RuntimeError(self.last_error)
            selected_model = models[0]
            self.model = selected_model

        prompt_messages = self._compile_messages(messages=messages, memory_context=memory_context, tool_names=resolved_tool_names)
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
            payload = self._request_json("POST", "/api/chat", body)
        except RuntimeError as exc:
            if schema is not None and _looks_like_schema_failure(str(exc)):
                self.last_backend_fallback = "Ollama rejected structured format; retried with plain JSON mode."
                body = self._chat_request_body(
                    model=selected_model,
                    messages=prompt_messages,
                    temperature=temperature,
                    schema="json",
                )
                payload = self._request_json("POST", "/api/chat", body)
            else:
                raise

        content = str(((payload.get("message") or {}).get("content")) or "").strip()
        usage = _extract_ollama_usage(payload)
        self.last_backend_used = "ollama"
        self.last_backend_reason = f"Ollama model {selected_model} handled the turn."
        self.last_error = ""
        if resolved_tool_names:
            parsed = _parse_structured_ollama_response(content, resolved_tool_names)
            if parsed is not None:
                return parsed
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage=usage,
            backend="ollama",
            metadata={
                "runtime": "ollama",
                "transport": "http_api",
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
        prepared: list[dict[str, str]] = []
        system_text = self.system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS
        planner_json_mode = any(
            "PLANNER_OUTPUT_MODE: blueprint_json" in str(message.get("content") or "")
            for message in messages
        )
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
            system_text = "\n\n".join(
                [
                    system_text,
                    "Available tools:",
                    json.dumps(tool_payload, separators=(",", ":"), sort_keys=True),
                    "If tool use is required, respond with exactly one JSON object and no prose.",
                    'For tool use return {"type":"tool_call","tool_name":"<tool>","arguments":{...}}.',
                    (
                        "For direct answers in planner mode, return the blueprint JSON object itself with tasks and edges."
                        if planner_json_mode
                        else 'For direct answers return {"type":"final","content":"<response>"}.'
                    ),
                    f"Allowed tools: {', '.join(tool_names)}.",
                    "Use only one tool call at a time and only from the listed tools.",
                    "Do not say the tools are unavailable when the needed file inspection or file editing tools are listed above.",
                ]
            )
        if memory_context and memory_context.strip():
            system_text = "\n\n".join([system_text, "Memory context:", memory_context.strip()])
        prepared.append({"role": "system", "content": system_text})
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = str(message.get("content") or "").strip()
            if content:
                prepared.append({"role": role, "content": content})
        return prepared

    def _chat_request_body(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        schema: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        profile = self._performance_profile()
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": profile["keep_alive"],
            "options": {
                "temperature": max(float(temperature), 0.0),
                "num_ctx": profile["ctx_size"],
                "num_thread": profile["threads"],
                "num_predict": profile["n_predict"],
            },
        }
        if schema is not None:
            body["format"] = schema
        return body

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            self.last_error = f"Ollama request failed ({exc.code}): {detail or exc.reason}."
            raise RuntimeError(self.last_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_error = f"Ollama is not running at {self.base_url}: {exc}."
            raise RuntimeError(self.last_error) from exc
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            self.last_error = "Ollama returned invalid JSON."
            raise RuntimeError(self.last_error) from exc
        if not isinstance(payload, dict):
            self.last_error = "Ollama returned an unexpected response shape."
            raise RuntimeError(self.last_error)
        return payload

    def _performance_profile(self) -> dict[str, int | str]:
        cpu_count = os.cpu_count() or 2
        mode = self.performance_mode
        if mode == "low":
            return {
                "threads": _env_int("DEVENV_OLLAMA_THREADS_LOW", 1),
                "ctx_size": _env_int("DEVENV_OLLAMA_CTX_LOW", 2048),
                "n_predict": _env_int("DEVENV_OLLAMA_N_PREDICT_LOW", 384),
                "keep_alive": os.getenv("DEVENV_OLLAMA_KEEP_ALIVE_LOW", "90s").strip() or "90s",
            }
        if mode == "high":
            return {
                "threads": _env_int("DEVENV_OLLAMA_THREADS_HIGH", max(cpu_count // 2, 1)),
                "ctx_size": _env_int("DEVENV_OLLAMA_CTX_HIGH", 8192),
                "n_predict": _env_int("DEVENV_OLLAMA_N_PREDICT_HIGH", 1024),
                "keep_alive": os.getenv("DEVENV_OLLAMA_KEEP_ALIVE_HIGH", "10m").strip() or "10m",
            }
        return {
            "threads": _env_int("DEVENV_OLLAMA_THREADS_MEDIUM", max(cpu_count // 3, 1)),
            "ctx_size": _env_int("DEVENV_OLLAMA_CTX_MEDIUM", DEFAULT_OLLAMA_NUM_CTX),
            "n_predict": _env_int("DEVENV_OLLAMA_N_PREDICT_MEDIUM", 768),
            "keep_alive": os.getenv("DEVENV_OLLAMA_KEEP_ALIVE_MEDIUM", DEFAULT_OLLAMA_KEEP_ALIVE).strip() or DEFAULT_OLLAMA_KEEP_ALIVE,
        }


def _extract_ollama_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, int] = {}
    prompt_tokens = payload.get("prompt_eval_count")
    completion_tokens = payload.get("eval_count")
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return usage


def _parse_structured_ollama_response(
    content: str,
    allowed_tools: list[str],
) -> AIResponse | None:
    payload = _load_relaxed_json_object(content)
    if not isinstance(payload, dict):
        return None
    response_type = str(payload.get("type") or "final").strip().lower()
    if response_type == "tool_call":
        tool_name = str(payload.get("tool_name") or "").strip()
        arguments = payload.get("arguments")
        if tool_name not in allowed_tools or not isinstance(arguments, dict):
            return None
        return AIResponse(
            content="",
            tool_calls=(ToolCallRequest(call_id="ollama_tool_call", tool_name=tool_name, arguments=arguments),),
            finish_reason="tool_calls",
            usage={},
            backend="ollama",
            metadata={"transport": "http_api", "runtime": "ollama"},
        )
    if "tasks" in payload or "nodes" in payload:
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={},
            backend="ollama",
            metadata={"transport": "http_api", "runtime": "ollama"},
        )
    final_content = str(payload.get("content") or "").strip()
    return AIResponse(
        content=final_content,
        finish_reason="stop",
        usage={},
        backend="ollama",
        metadata={"transport": "http_api", "runtime": "ollama"},
    )


def _load_relaxed_json_object(content: str) -> dict[str, Any] | None:
    candidate = str(content or "").strip()
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
        return _normalize_relaxed_json_object(payload)
    except json.JSONDecodeError:
        repaired = re.sub(r'"([A-Za-z0-9_]+):"\s*:', r'"\1":', candidate)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return None
        return _normalize_relaxed_json_object(payload)


def _normalize_relaxed_json_object(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        cleaned_key = key[:-1] if isinstance(key, str) and key.endswith(":") else key
        normalized[str(cleaned_key)] = value
    return normalized


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _looks_like_schema_failure(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return "format" in lowered and any(
        marker in lowered for marker in ("unsupported", "invalid", "schema", "json")
    )
