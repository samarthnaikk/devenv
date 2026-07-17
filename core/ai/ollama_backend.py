from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIResponse, ToolCallRequest
from core.tools.base import BaseTool

DEFAULT_OLLAMA_EXECUTABLE = "llama-cli"
DEFAULT_OLLAMA_MODEL = ""
DEFAULT_OLLAMA_NUM_CTX = 4096
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
        del base_url
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.executable = (
            os.getenv("DEVENV_LLAMA_CPP_EXECUTABLE", "").strip()
            or DEFAULT_OLLAMA_EXECUTABLE
        )
        self.model = (
            model
            or os.getenv("DEVENV_LLAMA_CPP_MODEL")
            or os.getenv("DEVENV_OLLAMA_MODEL")
            or DEFAULT_OLLAMA_MODEL
        ).strip()
        self.model_dir = str(
            Path(
                os.getenv("DEVENV_LLAMA_CPP_MODELS_DIR", "")
                or os.getenv("DEVENV_OLLAMA_MODELS_DIR", "")
                or ""
            ).expanduser()
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
        executable_path = shutil.which(self.executable)
        available = bool(executable_path)
        detail = (
            f"llama.cpp CLI available at {executable_path}."
            if executable_path
            else f"llama.cpp CLI `{self.executable}` not found on PATH."
        )
        model_metadata = self._model_metadata()
        if available and not self.model:
            detail = "llama.cpp CLI is installed, but no model is selected."
        elif available and self.model and not model_metadata["model_exists"]:
            detail = (
                f"Selected llama.cpp model `{self.model}` was not found as a local file. "
                "Set DEVENV_LLAMA_CPP_MODEL to a valid GGUF path or filename inside DEVENV_LLAMA_CPP_MODELS_DIR."
            )
            available = False
        if self.last_error:
            detail = self.last_error
            available = False
        return AIBackendStatus(
            name="ollama",
            available=available,
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "runtime": "llama.cpp",
                "transport": "subprocess_cli",
                "executable": executable_path or self.executable,
                "models": self.list_models(),
                "model_path": model_metadata["resolved_model_path"],
                "model_exists": model_metadata["model_exists"],
                "performance_mode": self.performance_mode,
                "threads": self._performance_profile()["threads"],
                "batch_size": self._performance_profile()["batch_size"],
                "ubatch_size": self._performance_profile()["ubatch_size"],
                "ctx_size": self._performance_profile()["ctx_size"],
                "n_gpu_layers": self._performance_profile()["n_gpu_layers"],
                "mmap": True,
                "last_error": self.last_error,
            },
        )

    def list_models(self) -> list[str]:
        ordered: list[str] = []
        if self.model:
            ordered.append(self.model)
        model_dir = Path(self.model_dir).expanduser() if self.model_dir else None
        if model_dir and model_dir.exists():
            for candidate in sorted(model_dir.rglob("*.gguf")):
                relative = str(candidate.relative_to(model_dir))
                if relative not in ordered:
                    ordered.append(relative)
        return ordered

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        executable_path = shutil.which(self.executable)
        if not executable_path:
            self.last_error = f"llama.cpp CLI `{self.executable}` is not installed."
            raise RuntimeError(self.last_error)
        resolved_model = self._resolve_model_path()
        if not resolved_model:
            self.last_error = (
                "llama.cpp backend requires DEVENV_LLAMA_CPP_MODEL to point to a GGUF model path "
                "or a filename inside DEVENV_LLAMA_CPP_MODELS_DIR."
            )
            raise RuntimeError(self.last_error)
        resolved_tool_names = [name for name in (tool_names or ()) if name in self._tools]
        planner_json_mode = any(
            "PLANNER_OUTPUT_MODE: blueprint_json" in str(message.get("content") or "")
            for message in messages
        )
        prompt = self._compile_prompt(
            messages=messages,
            memory_context=memory_context,
            tool_names=resolved_tool_names,
        )
        schema: dict[str, Any] | None = None
        if planner_json_mode:
            schema = PLAN_BLUEPRINT_JSON_SCHEMA
        elif resolved_tool_names:
            schema = TOOL_OR_FINAL_JSON_SCHEMA
        command = self._build_command(
            executable_path=executable_path,
            model_path=resolved_model,
            prompt=prompt,
            temperature=temperature,
            schema=schema,
        )
        completed = self._run_command(command)
        if completed.returncode != 0 and schema is not None and _looks_like_schema_flag_failure(completed.stderr):
            self.last_backend_fallback = (
                "llama.cpp rejected --json-schema; retried without schema enforcement."
            )
            command = self._build_command(
                executable_path=executable_path,
                model_path=resolved_model,
                prompt=prompt,
                temperature=temperature,
                schema=None,
            )
            completed = self._run_command(command)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit status {completed.returncode}"
            self.last_error = f"llama.cpp CLI failed: {detail}"
            raise RuntimeError(self.last_error)
        content = _strip_ansi(completed.stdout).strip()
        self.last_backend_used = "ollama"
        self.last_backend_reason = (
            f"llama.cpp model {Path(resolved_model).name if Path(resolved_model).suffix else resolved_model} handled the turn."
        )
        self.last_error = ""
        if resolved_tool_names:
            parsed = _parse_structured_ollama_response(content, resolved_tool_names)
            if parsed is not None:
                return parsed
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={},
            backend="ollama",
            metadata={
                "runtime": "llama.cpp",
                "transport": "subprocess_cli",
                "executable": executable_path,
                "model_path": resolved_model,
                "performance_mode": self.performance_mode,
            },
        )

    def _compile_prompt(
        self,
        *,
        messages: list[dict[str, Any]],
        memory_context: str | None,
        tool_names: list[str],
    ) -> str:
        sections: list[str] = []
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
        sections.extend(["## System", system_text])
        if memory_context and memory_context.strip():
            sections.extend(["## Memory", memory_context.strip()])
        for message in messages:
            role = str(message.get("role") or "user").upper()
            content = str(message.get("content") or "").strip()
            if content:
                sections.append(f"{role}: {content}")
        return "\n\n".join(sections).strip()

    def _build_command(
        self,
        *,
        executable_path: str,
        model_path: str,
        prompt: str,
        temperature: float,
        schema: dict[str, Any] | None,
    ) -> list[str]:
        profile = self._performance_profile()
        command = [
            executable_path,
            "--model",
            model_path,
            "--prompt",
            prompt,
            "--single-turn",
            "--no-display-prompt",
            "--no-conversation",
            "--simple-io",
            "--log-disable",
            "--threads",
            str(profile["threads"]),
            "--ctx-size",
            str(profile["ctx_size"]),
            "--batch-size",
            str(profile["batch_size"]),
            "--ubatch-size",
            str(profile["ubatch_size"]),
            "--n-predict",
            str(profile["n_predict"]),
            "--temp",
            str(max(float(temperature), 0.0)),
            "--poll",
            str(profile["poll"]),
        ]
        if profile["n_gpu_layers"] != "0":
            command.extend(["--n-gpu-layers", str(profile["n_gpu_layers"])])
        device = os.getenv("DEVENV_LLAMA_CPP_DEVICE", "").strip()
        if device:
            command.extend(["--device", device])
        if schema is not None:
            command.extend(["--json-schema", json.dumps(schema, separators=(",", ":"))])
        if os.getenv("DEVENV_LLAMA_CPP_MLOCK", "").strip().lower() in {"1", "true", "yes", "on"}:
            command.append("--mlock")
        return command

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.workspace_path,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self.last_error = f"llama.cpp CLI timed out after {self.timeout_seconds:.0f}s."
            raise RuntimeError(self.last_error) from exc
        except OSError as exc:
            self.last_error = f"llama.cpp CLI failed to start: {exc}"
            raise RuntimeError(self.last_error) from exc

    def _resolve_model_path(self) -> str | None:
        candidate = self.model.strip()
        if not candidate:
            return None
        model_path = Path(candidate).expanduser()
        if model_path.exists():
            return str(model_path.resolve())
        if self.model_dir:
            joined = Path(self.model_dir).expanduser() / candidate
            if joined.exists():
                return str(joined.resolve())
        return candidate if "/" in candidate or ":" in candidate else None

    def _model_metadata(self) -> dict[str, Any]:
        resolved = self._resolve_model_path()
        if not resolved:
            return {"resolved_model_path": "", "model_exists": False}
        candidate = Path(resolved).expanduser()
        return {
            "resolved_model_path": resolved,
            "model_exists": candidate.exists(),
        }

    def _performance_profile(self) -> dict[str, int | str]:
        cpu_count = os.cpu_count() or 2
        mode = self.performance_mode
        default_gpu_layers = os.getenv("DEVENV_LLAMA_CPP_GPU_LAYERS", "").strip()
        if mode == "low":
            profile = {
                "threads": _env_int("DEVENV_LLAMA_CPP_THREADS_LOW", 1),
                "batch_size": _env_int("DEVENV_LLAMA_CPP_BATCH_SIZE_LOW", 64),
                "ubatch_size": _env_int("DEVENV_LLAMA_CPP_UBATCH_SIZE_LOW", 16),
                "ctx_size": _env_int("DEVENV_LLAMA_CPP_CTX_LOW", 2048),
                "n_predict": _env_int("DEVENV_LLAMA_CPP_N_PREDICT_LOW", 512),
                "poll": _env_int("DEVENV_LLAMA_CPP_POLL_LOW", 0),
                "n_gpu_layers": default_gpu_layers or "999",
            }
        elif mode == "high":
            profile = {
                "threads": _env_int("DEVENV_LLAMA_CPP_THREADS_HIGH", max(cpu_count // 2, 1)),
                "batch_size": _env_int("DEVENV_LLAMA_CPP_BATCH_SIZE_HIGH", 512),
                "ubatch_size": _env_int("DEVENV_LLAMA_CPP_UBATCH_SIZE_HIGH", 128),
                "ctx_size": _env_int("DEVENV_LLAMA_CPP_CTX_HIGH", 8192),
                "n_predict": _env_int("DEVENV_LLAMA_CPP_N_PREDICT_HIGH", 1024),
                "poll": _env_int("DEVENV_LLAMA_CPP_POLL_HIGH", 50),
                "n_gpu_layers": default_gpu_layers or "999",
            }
        else:
            profile = {
                "threads": _env_int("DEVENV_LLAMA_CPP_THREADS_MEDIUM", max(cpu_count // 3, 1)),
                "batch_size": _env_int("DEVENV_LLAMA_CPP_BATCH_SIZE_MEDIUM", 256),
                "ubatch_size": _env_int("DEVENV_LLAMA_CPP_UBATCH_SIZE_MEDIUM", 64),
                "ctx_size": _env_int("DEVENV_LLAMA_CPP_CTX_MEDIUM", DEFAULT_OLLAMA_NUM_CTX),
                "n_predict": _env_int("DEVENV_LLAMA_CPP_N_PREDICT_MEDIUM", 768),
                "poll": _env_int("DEVENV_LLAMA_CPP_POLL_MEDIUM", 10),
                "n_gpu_layers": default_gpu_layers or "999",
            }
        return profile


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
            metadata={"transport": "subprocess_cli", "runtime": "llama.cpp"},
        )
    if "tasks" in payload or "nodes" in payload:
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={},
            backend="ollama",
            metadata={"transport": "subprocess_cli", "runtime": "llama.cpp"},
        )
    final_content = str(payload.get("content") or "").strip()
    return AIResponse(
        content=final_content,
        finish_reason="stop",
        usage={},
        backend="ollama",
        metadata={"transport": "subprocess_cli", "runtime": "llama.cpp"},
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


def _looks_like_schema_flag_failure(stderr: str) -> bool:
    lowered = str(stderr or "").lower()
    return "json-schema" in lowered and any(
        marker in lowered for marker in ("unknown", "unrecognized", "invalid option")
    )


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text or ""))
