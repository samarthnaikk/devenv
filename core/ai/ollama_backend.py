from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib import error, request

from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.tools.base import BaseTool

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_KEEP_ALIVE = "2m"
DEFAULT_OLLAMA_NUM_CTX = 4096


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
        self.model = model or os.getenv("DEVENV_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.base_url = (base_url or os.getenv("DEVENV_OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.system_instructions = system_instructions.strip()
        self.timeout_seconds = timeout_seconds
        self.last_backend_used = "ollama"
        self.last_backend_reason = ""
        self.last_backend_fallback = ""
        self.last_error = ""
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def set_model(self, model: str) -> None:
        self.model = model.strip()

    def reset_session(self) -> None:
        return None

    def abort(self) -> bool:
        return False

    def status(self) -> AIBackendStatus:
        try:
            models = self.list_models()
            detail = f"Ollama reachable at {self.base_url}."
            self.last_error = ""
            available = True
        except RuntimeError as exc:
            models = []
            detail = str(exc)
            self.last_error = detail
            available = False
        return AIBackendStatus(
            name="ollama",
            available=available,
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "base_url": self.base_url,
                "models": models,
                "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
                "num_thread": _default_num_threads(),
                "last_error": self.last_error,
            },
        )

    def list_models(self) -> list[str]:
        payload = self._request_json("/api/tags", {"Content-Type": "application/json"})
        models = payload.get("models")
        if not isinstance(models, list):
            return [self.model] if self.model else []
        ordered: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name not in ordered:
                ordered.append(name)
        if self.model and self.model not in ordered:
            ordered.insert(0, self.model)
        return ordered

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        resolved_tool_names = [name for name in (tool_names or ()) if name in self._tools]
        payload = {
            "model": self.model,
            "stream": True,
            "keep_alive": os.getenv("DEVENV_OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE),
            "options": {
                "temperature": temperature,
                "num_ctx": _env_int("DEVENV_OLLAMA_NUM_CTX", DEFAULT_OLLAMA_NUM_CTX),
                "num_thread": _env_int("DEVENV_OLLAMA_NUM_THREAD", _default_num_threads()),
            },
            "messages": self._compile_messages(
                messages=messages,
                memory_context=memory_context,
                tool_names=resolved_tool_names,
            ),
        }
        if resolved_tool_names:
            payload["format"] = "json"
        streamed = self._stream_chat(payload)
        self.last_backend_used = "ollama"
        self.last_backend_reason = f"Ollama model {self.model} handled the turn."
        self.last_backend_fallback = ""
        self.last_error = ""
        content = streamed["content"]
        usage = streamed["usage"]
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
                "transport": "http_stream",
                "base_url": self.base_url,
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
        system_text = self.system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS
        if tool_names:
            planner_json_mode = any(
                "PLANNER_OUTPUT_MODE: blueprint_json" in str(message.get("content") or "")
                for message in messages
            )
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
                    "For workspace code changes, inspect files with list_directory/read_file first, then use edit_file or write_file to make the change.",
                ]
            )
        compiled.append({"role": "system", "content": system_text})
        if memory_context and memory_context.strip():
            compiled.append(
                {
                    "role": "system",
                    "content": f"Retrieved memory context:\n{memory_context.strip()}",
                }
            )
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            compiled.append({"role": role, "content": content})
        return compiled

    def _stream_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
                "User-Agent": "devenv/0.1",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                chunks: list[str] = []
                final_payload: dict[str, Any] | None = None
                while True:
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    message = event.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                    if event.get("done"):
                        final_payload = event
                        break
                usage = _ollama_usage(final_payload or {})
                return {"content": "".join(chunks).strip(), "usage": usage}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"Ollama request failed with HTTP {exc.code}: {detail}"
            raise RuntimeError(self.last_error) from exc
        except error.URLError as exc:
            reason = str(exc.reason)
            self.last_error = (
                f"Ollama is not running at {self.base_url}. Start Ollama and try again. ({reason})"
            )
            raise RuntimeError(self.last_error) from exc

    def _request_json(self, path: str, headers: dict[str, str]) -> dict[str, Any]:
        req = request.Request(
            url=f"{self.base_url}{path}",
            headers={
                **headers,
                "Accept": "application/json",
                "User-Agent": "devenv/0.1",
            },
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Ollama is not running at {self.base_url}. Start Ollama and try again. ({exc.reason})"
            ) from exc
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned a malformed response.")
        return payload


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
            metadata={"transport": "http_stream"},
        )
    if "tasks" in payload or "nodes" in payload:
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={},
            backend="ollama",
            metadata={"transport": "http_stream"},
        )
    final_content = str(payload.get("content") or "").strip()
    return AIResponse(
        content=final_content,
        finish_reason="stop",
        usage={},
        backend="ollama",
        metadata={"transport": "http_stream"},
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


def _ollama_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, int] = {}
    prompt_eval_count = payload.get("prompt_eval_count")
    eval_count = payload.get("eval_count")
    if isinstance(prompt_eval_count, int):
        usage["prompt_tokens"] = prompt_eval_count
    if isinstance(eval_count, int):
        usage["completion_tokens"] = eval_count
    if isinstance(prompt_eval_count, int) and isinstance(eval_count, int):
        usage["total_tokens"] = prompt_eval_count + eval_count
    return usage


def _default_num_threads() -> int:
    cpu_count = os.cpu_count() or 2
    return max(cpu_count // 2, 1)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default
