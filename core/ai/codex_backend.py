from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.ai.engine import DEFAULT_SYSTEM_INSTRUCTIONS
from core.ai.models import AIBackendStatus, AIExecutedToolStep, AIResponse
from core.runtime.mcp_http import MCPHTTPServerManager, default_mcp_http_server_config


@dataclass(frozen=True)
class CodexRunResult:
    content: str | None
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    executed_steps: tuple[AIExecutedToolStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""


class CodexRunner(Protocol):
    def run_turn(
        self,
        *,
        model: str,
        prompt: str,
        mcp_server_url: str,
        mcp_auth_token: str | None,
        allowed_tools: list[str],
        base_url: str | None,
        api_key: str,
        timeout_seconds: float,
    ) -> CodexRunResult: ...

    def abort_run(self, run_id: str) -> bool: ...


class CodexAICore:
    provider_label = "Codex via OpenAI"
    supports_tool_calls = True

    def __init__(
        self,
        *,
        workspace_path: str,
        model: str | None = None,
        system_instructions: str = "",
        runner: CodexRunner | None = None,
        mcp_server_manager: MCPHTTPServerManager | None = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.model = model or os.getenv("DEVENV_CODEX_MODEL") or ""
        self.system_instructions = (system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS).strip()
        self.runner = runner or _AgentsCodexRunner()
        self.mcp_server_manager = mcp_server_manager or MCPHTTPServerManager(
            workspace_path=self.workspace_path,
            db_path="memory.db",
            vector_dir="vectors",
            config=default_mcp_http_server_config(),
        )
        self.last_backend_used = "codex"
        self.last_backend_reason = ""
        self.last_backend_fallback = ""
        self.last_error = ""
        self._last_run_id = ""

    def register_tool(self, tool) -> None:
        del tool
        return None

    def set_model(self, model: str) -> None:
        self.model = model.strip()

    def status(self) -> AIBackendStatus:
        endpoint_status = self.mcp_server_manager.inspect()
        available = bool(os.getenv("OPENAI_API_KEY")) and bool(self.model)
        detail = self.last_error or (
            "Configured" if available else "Codex requires OPENAI_API_KEY and DEVENV_CODEX_MODEL."
        )
        return AIBackendStatus(
            name="codex",
            available=available,
            enabled=True,
            model=self.model,
            detail=detail,
            supports_tool_calls=True,
            metadata={
                "transport": "responses_mcp",
                "mcp_server": endpoint_status.to_metadata(),
                "run_id": self._last_run_id,
                "last_error": self.last_error,
            },
        )

    def reset_session(self) -> None:
        self._last_run_id = ""

    def abort(self) -> bool:
        if not self._last_run_id:
            return False
        aborted = bool(self.runner.abort_run(self._last_run_id))
        if aborted:
            self._last_run_id = ""
        return aborted

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: Iterable[str] | None = None,
    ) -> AIResponse:
        del temperature
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self.last_error = "Codex backend requires OPENAI_API_KEY."
            raise RuntimeError(self.last_error)
        if not self.model:
            self.last_error = "Codex backend requires DEVENV_CODEX_MODEL."
            raise RuntimeError(self.last_error)

        endpoint_status = self.mcp_server_manager.ensure_server()
        prompt = self._compile_prompt(messages, memory_context)
        result = self.runner.run_turn(
            model=self.model,
            prompt=prompt,
            mcp_server_url=endpoint_status.base_url,
            mcp_auth_token=self.mcp_server_manager.config.auth_token,
            allowed_tools=[name for name in (tool_names or ()) if isinstance(name, str) and name.strip()],
            base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
            api_key=api_key,
            timeout_seconds=_float_env("DEVENV_CODEX_TIMEOUT_SECONDS", 60.0),
        )
        self._last_run_id = result.run_id
        self.last_backend_reason = "Codex handled the turn through the OpenAI MCP backend."
        self.last_error = ""
        return AIResponse(
            content=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
            backend="codex",
            metadata={
                "transport": "responses_mcp",
                "run_id": result.run_id,
                **result.metadata,
            },
            executed_steps=result.executed_steps,
        )

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


class _AgentsCodexRunner:
    def run_turn(
        self,
        *,
        model: str,
        prompt: str,
        mcp_server_url: str,
        mcp_auth_token: str | None,
        allowed_tools: list[str],
        base_url: str | None,
        api_key: str,
        timeout_seconds: float,
    ) -> CodexRunResult:
        del model, prompt, mcp_server_url, mcp_auth_token, allowed_tools, base_url, api_key, timeout_seconds
        _load_openai_agents_sdk()
        raise RuntimeError(
            "The OpenAI Agents SDK MCP execution path is not available in this environment yet. "
            "Install the OpenAI Agents SDK to enable the Codex backend."
        )

    def abort_run(self, run_id: str) -> bool:
        del run_id
        return False


def _load_openai_agents_sdk():
    try:
        import agents  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional OpenAI Agents SDK is not installed. Install it to enable the Codex backend."
        ) from exc
    return agents


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
