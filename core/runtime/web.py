from __future__ import annotations

import json
import logging
import os
import inspect
import re
import sysconfig
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.ai.models import AIExecutedToolStep, AIResponse, ToolCallRequest
from core.logging_utils import configure_logging

from .context_builder import ContextBuilderService
from .kernel import DevenvKernel
from .mcp_http import MCPHTTPServerManager, default_mcp_http_server_config
from .models import (
    PlanningMode,
    PreparedPromptRequest,
    PrivacyModeState,
    RunConfig,
    ToolReadiness,
)
from .response_sanitizer import sanitize_replay_text
from .setup import inspect_setup
from .tooling import build_runtime_tools
from .workspace import WorkspaceBrowser

logger = logging.getLogger(__name__)
DEFAULT_WEB_MODELS = (
    "opencode/claude-sonnet-4",
    "opencode/deepseek-v4-flash-free",
    "opencode/claude-sonnet-5",
    "opencode/claude-haiku-4-5",
    "opencode/north-mini-code-free",
)
DEFAULT_OLLAMA_MODELS = ("qwen2.5:3b",)
READ_ONLY_PLAN_TOOLS = (
    "list_directory",
    "locate_files",
    "read_file",
    "peek_lines",
    "inspect_symbols",
    "search_text",
    "track_symbol",
)
PLAN_MEMORY_CHAR_LIMIT = 1200
PLAN_BLUEPRINT_REPAIR_LIMIT = 2
PLAN_TEMPERATURE = 0.0
PLAN_SYSTEM_RULE = """You are Devenv's planning engine.

PLANNER_OUTPUT_MODE: blueprint_json

Your job is to inspect the workspace if needed using read-only tools, then return only a multi-node execution graph as raw JSON.

Never modify files. Never ask to apply changes. Never call mutation tools.

Return exactly one JSON object with this structure:
{
  "tasks": [
    {
      "task_id": "short-kebab-id",
      "description": "Concrete implementation step",
      "level": 0
    }
  ],
  "edges": [
    { "from": "task-a", "to": "task-b" }
  ]
}

Rules:
- Return raw JSON only. No markdown fences. No prose before or after the JSON.
- Use multiple tasks when the work can be broken down.
- Use multiple levels when later tasks depend on earlier tasks.
- Every task must have a unique string task_id.
- Every task must have a non-empty description.
- Every task must have an integer level >= 0.
- Include edges whenever there is more than one task.
- Prefer 3-8 tasks for non-trivial implementation requests.
"""


class AccessPolicy:
    def __init__(self) -> None:
        self.session_access: dict[str, bool] = {"codex": False, "opencode": False}
        self.backend_access: dict[str, bool] = {"opencode": False, "ollama": False, "codex": False}

    def set_session_access(self, provider: str, allowed: bool) -> dict[str, object]:
        self.session_access[provider] = allowed
        return self.snapshot()

    def set_backend_access(self, backend: str, allowed: bool) -> dict[str, object]:
        self.backend_access[backend] = allowed
        return self.snapshot()

    def can_access_provider(self, provider: str) -> bool:
        return bool(self.session_access.get(provider, False))

    def can_use_backend(self, backend: str) -> bool:
        return bool(self.backend_access.get(backend, False))

    def snapshot(self) -> dict[str, object]:
        return {
            "session_access": dict(self.session_access),
            "backend_access": dict(self.backend_access),
        }


class DevenvWebApp:
    def __init__(
        self, config: RunConfig, port: int = 4173, *, memory=None, ai=None
    ) -> None:
        self.config = config
        self.port = port
        self.static_root = _resolve_static_root()
        self.performance_mode = (
            config.performance_mode
            if config.performance_mode in {"low", "medium", "high"}
            else "medium"
        )
        self.privacy_mode = {
            "no_memory": bool(config.no_memory),
            "incognito": bool(config.incognito),
        }
        self.kernel = DevenvKernel(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
            memory=memory,
            ai=ai,
        )
        self.workspace = WorkspaceBrowser(config.workspace_path)
        self.context_builder = ContextBuilderService(
            config.workspace_path,
            memory=self.kernel.memory,
            provider_configs=config.external_session_configs,
            performance_mode=self.performance_mode,
        )
        self.context_builder.set_runtime_allowed_providers(set())
        self.kernel.context_builder = self.context_builder
        for tool in build_runtime_tools(
            self.kernel.memory, context_builder=self.context_builder
        ):
            self.kernel.register_tool(tool)
        self.access_policy = AccessPolicy()
        self._setup_cache: dict[str, object] | None = None
        self._setup_cache_ttl_seconds = 20.0
        self._mcp_server_manager = MCPHTTPServerManager(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
            config=default_mcp_http_server_config(),
        )

    def create_handler(self):
        return partial(DevenvRequestHandler, app=self)

    def create_server(self) -> ThreadingHTTPServer:
        return ThreadingHTTPServer(("127.0.0.1", self.port), self.create_handler())

    def serve(self) -> None:
        server = self.create_server()
        logger.info(
            "Starting Devenv web server: url=http://127.0.0.1:%s workspace=%s",
            self.port,
            self.config.workspace_path,
        )
        print(f"Devenv website running at http://127.0.0.1:{server.server_address[1]}")
        try:
            server.serve_forever()
        finally:
            self._mcp_server_manager.close()
            self.kernel.close()

    def build_health_payload(self) -> dict[str, object]:
        model = getattr(self.kernel.ai, "model", "unknown")
        ai_statuses = getattr(self.kernel.ai, "status", lambda: {})()
        active_backend = getattr(self.kernel.ai, "last_backend_used", "opencode")
        preferred_backend = getattr(self.kernel.ai, "preferred_backend", "opencode")
        active_provider_label = {
            "opencode": "OpenCode CLI",
            "ollama": "Ollama",
            "codex": "Codex via OpenAI",
        }.get(active_backend, getattr(self.kernel.ai, "provider_label", "OpenCode CLI"))
        model_catalog = self._model_catalog(ai_statuses=ai_statuses, active_backend=active_backend, current_model=model)
        setup = self._cached_setup_readiness()
        privacy = PrivacyModeState(
            no_memory=self.privacy_mode["no_memory"],
            incognito=self.privacy_mode["incognito"],
        )
        tool_readiness = self._build_tool_readiness()
        mcp_server = self._mcp_server_manager.inspect().to_metadata()
        opencode_server = {}
        codex_backend = {}
        if isinstance(ai_statuses, dict):
            opencode_status = ai_statuses.get("opencode")
            if opencode_status is not None:
                opencode_server = dict(
                    getattr(opencode_status, "metadata", {}) or {}
                ).get("server", {})
            codex_status = ai_statuses.get("codex")
            if codex_status is not None:
                codex_backend = dict(getattr(codex_status, "metadata", {}) or {})
        return {
            "workspace_path": self.config.workspace_path,
            "port": self.port,
            "tools": sorted(self.kernel.tools),
            "status": "ok",
            "ai_provider": active_provider_label,
            "ai_model": model,
            "available_models": list(model_catalog.get(active_backend, [])),
            "available_models_by_backend": model_catalog,
            "selected_models_by_backend": self._selected_models_by_backend(ai_statuses, fallback_model=model, preferred_backend=preferred_backend),
            "context_builder_enabled": True,
            "context_sources": [
                source.to_dict() for source in self.context_builder.list_sources()
            ],
            "access_policy": self.access_policy.snapshot(),
            "ai_backends": {
                name: status.to_dict() for name, status in ai_statuses.items()
            },
            "opencode_server": opencode_server,
            "codex_backend": codex_backend,
            "active_backend": active_backend,
            "preferred_backend": preferred_backend,
            "indexing": self.context_builder.indexing_status(),
            "setup": setup.to_dict(),
            "performance_mode": self.performance_mode,
            "privacy": privacy.to_dict(),
            "tool_readiness": {
                name: readiness.to_dict() for name, readiness in tool_readiness.items()
            },
            "mcp_server": mcp_server,
        }

    def _cached_setup_readiness(self):
        now = time.time()
        cached = self._setup_cache or {}
        expires_at = float(cached.get("expires_at", 0) or 0)
        readiness = cached.get("readiness")
        if readiness is not None and expires_at > now:
            return readiness
        readiness = inspect_setup(self.config, include_optional=True)
        self._setup_cache = {
            "readiness": readiness,
            "expires_at": now + self._setup_cache_ttl_seconds,
        }
        return readiness

    def _build_tool_readiness(self) -> dict[str, ToolReadiness]:
        return {
            "web_search": ToolReadiness(
                name="web_search",
                ready=False,
                detail="Tool contract reserved. Provider readiness is not implemented yet.",
            ),
            "generate_prompt": ToolReadiness(
                name="generate_prompt",
                ready=True,
                detail="Prompt-preparation primitives are available and will be exposed as a runtime tool.",
            ),
            "generate_pdf": ToolReadiness(
                name="generate_pdf",
                ready=False,
                detail="Tool contract reserved. LaTeX pipeline is not implemented yet.",
            ),
        }

    def _available_models(self, *, current_model: str) -> list[str]:
        configured = os.getenv("DEVENV_AVAILABLE_MODELS", "")
        configured_models = [
            item.strip() for item in configured.split(",") if item.strip()
        ]
        ordered: list[str] = []
        for model_name in [current_model, *configured_models, *DEFAULT_WEB_MODELS]:
            if model_name and model_name not in ordered:
                ordered.append(model_name)
        return ordered

    def _model_catalog(
        self,
        *,
        ai_statuses: dict[str, object],
        active_backend: str,
        current_model: str,
    ) -> dict[str, list[str]]:
        catalog: dict[str, list[str]] = {
            "opencode": self._available_models(current_model=current_model if active_backend == "opencode" else getattr(getattr(self.kernel.ai, "opencode_ai", None), "model", "")),
            "ollama": list(DEFAULT_OLLAMA_MODELS),
            "codex": [],
        }
        if isinstance(ai_statuses, dict):
            for backend, status in ai_statuses.items():
                metadata = dict(getattr(status, "metadata", {}) or {})
                model_name = str(getattr(status, "model", "") or "").strip()
                if backend == "ollama":
                    models = [str(item).strip() for item in metadata.get("models", []) if str(item).strip()]
                    ordered: list[str] = []
                    for candidate in [model_name, *models, *DEFAULT_OLLAMA_MODELS]:
                        if candidate and candidate not in ordered:
                            ordered.append(candidate)
                    catalog["ollama"] = ordered
                elif backend == "codex":
                    catalog["codex"] = [model_name] if model_name else []
        return catalog

    def _selected_models_by_backend(
        self,
        ai_statuses: dict[str, object],
        *,
        fallback_model: str,
        preferred_backend: str,
    ) -> dict[str, str]:
        selected = {
            "opencode": str(getattr(getattr(self.kernel.ai, "opencode_ai", None), "model", "") or ""),
            "ollama": str(getattr(getattr(self.kernel.ai, "ollama_ai", None), "model", "") or ""),
            "codex": str(getattr(getattr(self.kernel.ai, "codex_ai", None), "model", "") or ""),
        }
        if isinstance(ai_statuses, dict):
            for backend, status in ai_statuses.items():
                model_name = str(getattr(status, "model", "") or "").strip()
                if model_name:
                    selected[backend] = model_name
        if preferred_backend in selected and not selected[preferred_backend]:
            selected[preferred_backend] = fallback_model
        return selected

    def build_files_payload(self, relative_path: str = "") -> dict[str, object]:
        return {
            "path": relative_path,
            "entries": [
                entry.to_dict() for entry in self.workspace.list_entries(relative_path)
            ],
        }

    def build_file_payload(self, relative_path: str) -> dict[str, object]:
        return {
            "path": relative_path,
            **self.workspace.read_file_preview(relative_path),
        }

    def build_context_sources_payload(self) -> dict[str, object]:
        return {
            "sources": [
                source.to_dict() for source in self.context_builder.list_sources()
            ],
            "access_policy": self.access_policy.snapshot(),
        }

    def build_context_sessions_payload(self, provider_name: str) -> dict[str, object]:
        self._require_provider_access(provider_name)
        return {
            "provider": provider_name,
            "sessions": [
                session.to_dict()
                for session in self.context_builder.list_sessions(provider_name)
            ],
        }

    def build_context_session_payload(
        self, provider_name: str, session_id: str
    ) -> dict[str, object]:
        self._require_provider_access(provider_name)
        detail = self.context_builder.get_session(provider_name, session_id)
        return detail.to_dict()

    def build_prepared_prompt_payload(
        self, payload: dict[str, object]
    ) -> dict[str, object]:
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Missing required field: task")
        provider = payload.get("provider")
        session_ids_raw = payload.get("session_ids") or []
        if not isinstance(session_ids_raw, list) or any(
            not isinstance(item, str) for item in session_ids_raw
        ):
            raise ValueError("session_ids must be a list of strings")
        include_workspace_scan = bool(payload.get("include_workspace_scan", True))
        include_prior_context = bool(payload.get("include_prior_context", True))
        output_format = str(payload.get("output_format", "compact"))
        request = PreparedPromptRequest(
            task=task.strip(),
            provider=provider
            if isinstance(provider, str) and provider.strip()
            else None,
            session_ids=tuple(session_ids_raw),
            include_workspace_scan=include_workspace_scan,
            include_prior_context=include_prior_context,
            output_format="detailed" if output_format == "detailed" else "compact",
        )
        return self.context_builder.prepare_prompt(request).to_dict()

    def run_turn(
        self,
        prompt: str,
        max_consecutive_tools: int | None = None,
        planning_mode: PlanningMode = PlanningMode.AUTO,
        continue_plan: bool = False,
        local_only: bool = False,
        selected_tools: list[str] | None = None,
        backend_preference: str = "opencode",
        session_budget_tokens: int | None = None,
    ) -> dict[str, object]:
        execute_turn = self.kernel.execute_turn
        kwargs = {
            "max_consecutive_tools": max_consecutive_tools
            or self.config.max_consecutive_tools,
            "planning_mode": planning_mode,
            "continue_plan": continue_plan,
            "local_only": local_only,
        }
        parameters = inspect.signature(execute_turn).parameters
        if "backend_preference" in parameters:
            kwargs["backend_preference"] = backend_preference
        if "opencode_enabled" in parameters:
            kwargs["opencode_enabled"] = self.access_policy.can_use_backend("opencode")
        if "ollama_enabled" in parameters:
            kwargs["ollama_enabled"] = self.access_policy.can_use_backend("ollama")
        if "codex_enabled" in parameters:
            kwargs["codex_enabled"] = self.access_policy.can_use_backend("codex")
        if "session_budget_tokens" in parameters:
            kwargs["session_budget_tokens"] = session_budget_tokens
        if "selected_tools" in parameters:
            kwargs["selected_tools"] = selected_tools or []
        if "no_memory" in parameters:
            kwargs["no_memory"] = self.privacy_mode["no_memory"]
        if "incognito" in parameters:
            kwargs["incognito"] = self.privacy_mode["incognito"]
        result = execute_turn(prompt, **kwargs).to_dict()
        result["final_response"] = sanitize_replay_text(result.get("final_response"))
        result["error_message"] = sanitize_replay_text(result.get("error_message"))
        metadata = dict(result.get("metadata") or {})
        result["backend_used"] = metadata.get("backend_used", "opencode")
        result["budget_state"] = metadata.get("budget_state")
        result["usage_sample"] = {
            "prompt_tokens": int(
                result.get("total_usage", {}).get("prompt_tokens", 0) or 0
            ),
            "completion_tokens": int(
                result.get("total_usage", {}).get("completion_tokens", 0) or 0
            ),
            "total_tokens": int(
                result.get("total_usage", {}).get("total_tokens", 0) or 0
            ),
        }
        return result

    def run_plan(
        self,
        prompt: str,
        *,
        max_consecutive_tools: int | None = None,
        selected_tools: list[str] | None = None,
        backend_preference: str = "opencode",
        local_only: bool = False,
    ) -> dict[str, object]:
        turn_started_at = time.perf_counter()
        steps = []
        total_usage: dict[str, int] = {}
        ai_logs = [f"Queued plan prompt: {prompt}"]
        system_logs = [f"Workspace: {self.config.workspace_path}", "Plan-only mode active"]
        metadata: dict[str, object] = {
            "backend_preference": backend_preference,
            "backend_used": "local",
            "selected_tools": [],
            "plan_mode": "explicit",
        }
        if hasattr(self.kernel.ai, "set_backend_preference"):
            self.kernel.ai.set_backend_preference(
                backend_preference,
                opencode_enabled=self.access_policy.can_use_backend("opencode"),
                ollama_enabled=self.access_policy.can_use_backend("ollama"),
                codex_enabled=self.access_policy.can_use_backend("codex"),
            )
        try:
            memory_context, retrieval_metadata = self.kernel._retrieve_memory_context(
                prompt, local_only=local_only
            )
        except Exception as exc:
            logger.warning("Plan mode memory retrieval failed: error=%s", exc)
            memory_context = ""
            retrieval_metadata = {
                "external_context_state": "new_context",
                "external_context_reason": "Planning continued without memory context.",
                "external_context_session_count": 0,
                "external_context_session_ids": [],
            }
        metadata.update(retrieval_metadata)
        planning_memory = _trim_plan_memory_context(memory_context)
        if planning_memory:
            system_logs.append(f"Planning memory chars sent: {len(planning_memory)}")
        allowed_tools = self._resolve_plan_tools(selected_tools)
        metadata["selected_tools"] = list(allowed_tools)
        system_logs.append(f"Planning tool scope size: {len(allowed_tools)}")
        conversation = [
            {"role": "system", "content": PLAN_SYSTEM_RULE},
            {"role": "user", "content": prompt},
        ]
        max_tools = max_consecutive_tools or self.config.max_consecutive_tools
        repair_attempts = 0

        while True:
            ai_response = self.kernel.ai.chat(
                messages=list(conversation),
                memory_context=planning_memory,
                temperature=PLAN_TEMPERATURE,
                tool_names=list(allowed_tools),
            )
            _merge_usage_counts(total_usage, ai_response.usage)
            metadata["backend_used"] = ai_response.backend or getattr(
                self.kernel.ai, "last_backend_used", "opencode"
            )
            ai_logs.append(
                f"Plan response: finish_reason={ai_response.finish_reason}, tool_calls={len(ai_response.tool_calls)}, total_tokens={ai_response.usage.get('total_tokens', 0)}"
            )
            if ai_response.executed_steps:
                converted_steps = [
                    _runtime_step_from_ai_step(step)
                    for step in ai_response.executed_steps
                ]
                for step in converted_steps:
                    steps.append(step)
                    system_logs.append(
                        f"Tool step {len(steps)}: {step.tool_name} success={step.success}"
                    )
                if ai_response.content:
                    blueprint = _parse_plan_blueprint(ai_response.content)
                    if blueprint is not None:
                        return _build_plan_result(
                            final_response=ai_response.content,
                            blueprint=blueprint,
                            steps=steps,
                            total_usage=total_usage,
                            ai_logs=ai_logs,
                            system_logs=system_logs,
                            metadata=metadata,
                            elapsed_ms=int(
                                (time.perf_counter() - turn_started_at) * 1000
                            ),
                        )
            if ai_response.tool_calls:
                tool_call = ai_response.tool_calls[0]
                if tool_call.tool_name not in allowed_tools:
                    ai_logs.append(
                        f"Blocked non-read-only planning tool: {tool_call.tool_name}"
                    )
                    system_logs.append(
                        f"Blocked planning tool call: {tool_call.tool_name}"
                    )
                    conversation.append(
                        _assistant_tool_call_message(ai_response, [tool_call])
                    )
                    conversation.append(
                        _tool_message(
                            tool_call.call_id,
                            tool_call.tool_name,
                            "Plan mode is read-only. Use only the listed inspection tools and then return raw plan JSON.",
                        )
                    )
                    continue
                if len(steps) >= max_tools:
                    raise RuntimeError(
                        "Planning tool limit reached before a valid blueprint could be produced."
                    )
                step = self.kernel._execute_tool_call(tool_call)
                steps.append(step)
                ai_logs.append(f"Planning tool requested: {tool_call.tool_name}")
                system_logs.append(
                    f"Tool step {len(steps)}: {tool_call.tool_name} success={step.success}"
                )
                conversation.append(
                    _assistant_tool_call_message(ai_response, [tool_call])
                )
                conversation.append(
                    _tool_message(
                        tool_call.call_id,
                        tool_call.tool_name,
                        _format_tool_output(step.output, step.data),
                    )
                )
                continue

            content = ai_response.content or ""
            blueprint = _parse_plan_blueprint(content)
            if blueprint is not None:
                return _build_plan_result(
                    final_response=content,
                    blueprint=blueprint,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    metadata=metadata,
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )
            if repair_attempts >= PLAN_BLUEPRINT_REPAIR_LIMIT:
                return _build_plan_result(
                    final_response=content,
                    blueprint=None,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs
                    + ["Planner failed to return a valid JSON blueprint."],
                    metadata=metadata,
                    error_message="Planner did not return a valid multi-node JSON blueprint.",
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )
            repair_attempts += 1
            ai_logs.append(
                "Planner returned invalid JSON blueprint; requesting repair"
            )
            conversation.append({"role": "assistant", "content": content})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response was not a valid planning blueprint. "
                        "Return only raw JSON with a 'tasks' array of objects containing "
                        "'task_id', 'description', and integer 'level', plus an 'edges' array when there are multiple tasks."
                    ),
                }
            )

    def _resolve_plan_tools(
        self, selected_tools: list[str] | None
    ) -> tuple[str, ...]:
        requested = [
            tool_name
            for tool_name in (selected_tools or READ_ONLY_PLAN_TOOLS)
            if isinstance(tool_name, str) and tool_name in READ_ONLY_PLAN_TOOLS
        ]
        resolved = []
        for tool_name in requested:
            if tool_name in self.kernel.tools and tool_name not in resolved:
                resolved.append(tool_name)
        return tuple(resolved)

    def run_tool(
        self, tool_name: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        tool = self.kernel.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        result = tool.execute(**arguments)
        return {
            "tool_name": tool_name,
            "success": result.success,
            "output": result.output,
            "data": result.data,
        }

    def set_model(self, model: str, backend: str | None = None) -> dict[str, object]:
        cleaned = model.strip()
        if not cleaned:
            raise ValueError("Missing required field: model")
        cleaned_backend = str(backend or getattr(self.kernel.ai, "preferred_backend", "") or "").strip().lower() or None
        if cleaned_backend and hasattr(self.kernel.ai, "set_backend_model"):
            self.kernel.ai.set_backend_model(cleaned_backend, cleaned)
        elif hasattr(self.kernel.ai, "set_model"):
            self.kernel.ai.set_model(cleaned)
        else:
            self.kernel.ai.model = cleaned
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        active_backend = getattr(self.kernel.ai, "last_backend_used", cleaned_backend or "opencode")
        preferred_backend = getattr(self.kernel.ai, "preferred_backend", cleaned_backend or "opencode")
        model_catalog = self._model_catalog(ai_statuses=statuses, active_backend=active_backend, current_model=cleaned)
        return {
            "ai_provider": getattr(self.kernel.ai, "provider_label", "OpenCode CLI"),
            "ai_model": cleaned,
            "available_models": list(model_catalog.get(cleaned_backend or preferred_backend or active_backend, [])),
            "available_models_by_backend": model_catalog,
            "selected_models_by_backend": self._selected_models_by_backend(statuses, fallback_model=cleaned, preferred_backend=preferred_backend),
        }

    def update_session_access(self, provider: str, allowed: bool) -> dict[str, object]:
        if provider not in {"codex", "opencode"}:
            raise ValueError("provider must be one of: codex, opencode")
        snapshot = self.access_policy.set_session_access(provider, allowed)
        allowed_providers = {
            name
            for name, permitted in self.access_policy.session_access.items()
            if permitted
        }
        self.context_builder.set_runtime_allowed_providers(allowed_providers)
        return snapshot

    def update_backend_access(self, backend: str, allowed: bool) -> dict[str, object]:
        if backend not in {"opencode", "ollama", "codex"}:
            raise ValueError("backend must be one of: opencode, ollama, codex")
        return self.access_policy.set_backend_access(backend, allowed)

    def update_performance_mode(self, performance_mode: str) -> dict[str, object]:
        cleaned = performance_mode.strip().lower()
        if cleaned not in {"low", "medium", "high"}:
            raise ValueError("performance_mode must be one of: low, medium, high")
        self.performance_mode = cleaned
        if hasattr(self.context_builder, "set_performance_mode"):
            self.context_builder.set_performance_mode(cleaned)
        return {"performance_mode": self.performance_mode}

    def update_privacy_mode(
        self, *, no_memory: bool, incognito: bool
    ) -> dict[str, object]:
        self.privacy_mode["incognito"] = bool(incognito)
        self.privacy_mode["no_memory"] = bool(no_memory or incognito)
        return {"privacy": dict(self.privacy_mode)}

    def reset_thread(self) -> dict[str, object]:
        session_id = self.kernel.reset_conversation()
        return {
            "session_id": session_id,
            "state": self.kernel.state.name,
            "usage": dict(self.kernel.session_usage_totals),
        }

    def _require_provider_access(self, provider_name: str) -> None:
        if not self.access_policy.can_access_provider(provider_name):
            raise PermissionError(
                f"Access to {provider_name} sessions requires explicit user permission."
            )


class DevenvRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: DevenvWebApp, **kwargs):
        self.app = app
        super().__init__(*args, directory=str(app.static_root), **kwargs)

    def guess_type(self, path: str | os.PathLike[str]) -> str:
        path_str = str(path)
        if path_str.endswith(".js") or path_str.endswith(".mjs"):
            return "application/javascript"
        if path_str.endswith(".css"):
            return "text/css"
        return super().guess_type(path)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._write_json(HTTPStatus.OK, self.app.build_health_payload())
                return
            if parsed.path == "/api/files":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(
                    HTTPStatus.OK, self.app.build_files_payload(relative_path)
                )
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(
                    HTTPStatus.OK, self.app.build_file_payload(relative_path)
                )
                return
            if parsed.path == "/api/context-sources":
                self._write_json(
                    HTTPStatus.OK, self.app.build_context_sources_payload()
                )
                return
            if parsed.path.startswith("/api/context-sources/"):
                payload = self._match_context_source_path(parsed.path)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                provider_name, session_id = payload
                if session_id is None:
                    self._write_json(
                        HTTPStatus.OK,
                        self.app.build_context_sessions_payload(provider_name),
                    )
                    return
                self._write_json(
                    HTTPStatus.OK,
                    self.app.build_context_session_payload(provider_name, session_id),
                )
                return
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
        ) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/context-builder/prepare":
            try:
                prepared = self.app.build_prepared_prompt_payload(payload)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except FileNotFoundError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, prepared)
            return
        if parsed.path == "/api/session-access":
            provider = payload.get("provider")
            allowed = payload.get("allowed")
            if not isinstance(provider, str) or not isinstance(allowed, bool):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "provider and allowed are required"},
                )
                return
            try:
                result = self.app.update_session_access(provider, allowed)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/backend-access":
            backend = payload.get("backend")
            allowed = payload.get("allowed")
            if not isinstance(backend, str) or not isinstance(allowed, bool):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "backend and allowed are required"},
                )
                return
            try:
                result = self.app.update_backend_access(backend, allowed)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/model":
            model = payload.get("model")
            backend = payload.get("backend")
            if not isinstance(model, str):
                self._write_json(
                    HTTPStatus.BAD_REQUEST, {"error": "Missing required field: model"}
                )
                return
            if backend is not None and not isinstance(backend, str):
                self._write_json(
                    HTTPStatus.BAD_REQUEST, {"error": "backend must be a string when provided"}
                )
                return
            try:
                result = self.app.set_model(model, backend)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/tool":
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if not isinstance(tool_name, str):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Missing required field: tool_name"},
                )
                return
            if not isinstance(arguments, dict):
                self._write_json(
                    HTTPStatus.BAD_REQUEST, {"error": "arguments must be an object"}
                )
                return
            try:
                result = self.app.run_tool(tool_name, arguments)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/performance":
            performance_mode = payload.get("performance_mode")
            if not isinstance(performance_mode, str):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Missing required field: performance_mode"},
                )
                return
            try:
                result = self.app.update_performance_mode(performance_mode)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/privacy":
            no_memory = payload.get("no_memory", False)
            incognito = payload.get("incognito", False)
            if not isinstance(no_memory, bool) or not isinstance(incognito, bool):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "no_memory and incognito must be booleans"},
                )
                return
            result = self.app.update_privacy_mode(
                no_memory=no_memory, incognito=incognito
            )
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/thread/reset":
            self._write_json(HTTPStatus.OK, self.app.reset_thread())
            return
        if parsed.path not in {"/api/turn", "/api/plan"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "Missing required field: prompt"}
            )
            return

        max_consecutive_tools = payload.get("max_consecutive_tools")
        if max_consecutive_tools is not None and not isinstance(
            max_consecutive_tools, int
        ):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "max_consecutive_tools must be an integer"},
            )
            return
        planning_mode_value = payload.get("planning_mode", PlanningMode.AUTO.value)
        if not isinstance(planning_mode_value, str):
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "planning_mode must be a string"}
            )
            return
        try:
            planning_mode = PlanningMode(planning_mode_value)
        except ValueError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "planning_mode must be one of: auto, force_plan, force_direct"
                },
            )
            return
        continue_plan = payload.get("continue_plan", False)
        if not isinstance(continue_plan, bool):
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "continue_plan must be a boolean"}
            )
            return
        local_only = payload.get("local_only", False)
        if not isinstance(local_only, bool):
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "local_only must be a boolean"}
            )
            return
        selected_tools = payload.get("selected_tools", [])
        if not isinstance(selected_tools, list) or any(
            not isinstance(item, str) for item in selected_tools
        ):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "selected_tools must be a list of strings"},
            )
            return
        backend_preference = payload.get("backend_preference", "opencode")
        if not isinstance(backend_preference, str):
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "backend_preference must be a string"}
            )
            return
        session_budget_tokens = payload.get("session_budget_tokens")
        if session_budget_tokens is not None and not isinstance(
            session_budget_tokens, int
        ):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "session_budget_tokens must be an integer"},
            )
            return

        try:
            if parsed.path == "/api/plan":
                result = self.app.run_plan(
                    prompt=prompt,
                    max_consecutive_tools=max_consecutive_tools,
                    selected_tools=selected_tools,
                    backend_preference=backend_preference,
                    local_only=local_only,
                )
            else:
                result = self.app.run_turn(
                    prompt=prompt,
                    max_consecutive_tools=max_consecutive_tools,
                    planning_mode=planning_mode,
                    continue_plan=continue_plan,
                    local_only=local_only,
                    selected_tools=selected_tools,
                    backend_preference=backend_preference,
                    session_budget_tokens=session_budget_tokens,
                )
        except (RuntimeError, PermissionError) as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args) -> None:
        logger.info("web request: " + format, *args)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _match_context_source_path(self, path: str) -> tuple[str, str | None] | None:
        parts = [part for part in path.split("/") if part]
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "context-sources"
            and parts[3] == "sessions"
        ):
            return parts[2], None
        if (
            len(parts) == 5
            and parts[0] == "api"
            and parts[1] == "context-sources"
            and parts[3] == "sessions"
        ):
            return parts[2], parts[4]
        return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Devenv website runtime.")
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace path to sandbox the runtime within.",
    )
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--max-consecutive-tools", type=int, default=5)
    parser.add_argument(
        "--performance-mode", default="low", choices=("low", "medium", "high")
    )
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = RunConfig(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
        max_consecutive_tools=args.max_consecutive_tools,
        performance_mode=args.performance_mode,
    )
    app = DevenvWebApp(config=config, port=args.port)
    app.serve()
    return 0


def _resolve_static_root() -> Path:
    env_override = os.getenv("DEVENV_STATIC_ROOT", "").strip()
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if _is_valid_static_root(candidate):
            return candidate

    package_root = Path(__file__).resolve().parents[2]
    source_candidate = package_root / "interface" / "website"
    if _is_valid_static_root(source_candidate):
        return source_candidate

    data_candidate = (
        Path(sysconfig.get_path("data")).resolve()
        / "share"
        / "devenv"
        / "interface"
        / "website"
    )
    if _is_valid_static_root(data_candidate):
        return data_candidate

    raise FileNotFoundError(
        "Devenv web static assets were not found. "
        "Set DEVENV_STATIC_ROOT or reinstall the package with bundled web assets."
    )


def _is_valid_static_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "index.html").is_file()
        and (path / "styles.css").is_file()
    )


def _merge_usage_counts(target: dict[str, int], usage: dict[str, int] | None) -> None:
    for key, value in dict(usage or {}).items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _assistant_tool_call_message(
    ai_response: AIResponse,
    tool_calls: list[ToolCallRequest] | None = None,
) -> dict[str, object]:
    selected_tool_calls = tool_calls or list(ai_response.tool_calls)
    return {
        "role": "assistant",
        "content": ai_response.content,
        "tool_calls": [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in selected_tool_calls
        ],
    }


def _tool_message(call_id: str, tool_name: str, output: str) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": output,
    }


def _format_tool_output(output: str, data: dict[str, object]) -> str:
    if not data:
        return output
    return f"{output}\n{json.dumps(data, sort_keys=True)}"


def _runtime_step_from_ai_step(step: AIExecutedToolStep):
    from .models import ToolExecutionStep

    return ToolExecutionStep(
        step_id=step.step_id,
        tool_name=step.tool_name,
        arguments=dict(step.arguments),
        output=step.output,
        success=step.success,
        is_sandboxed_violation=False,
        data=dict(step.data),
    )


def _trim_plan_memory_context(memory_context: str) -> str:
    stripped = str(memory_context or "").strip()
    if not stripped:
        return ""
    return stripped[:PLAN_MEMORY_CHAR_LIMIT]


def _parse_plan_blueprint(content: str | None) -> dict[str, object] | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    for candidate in _plan_json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        blueprint = _coerce_plan_blueprint(parsed)
        if blueprint is not None:
            return blueprint
    return _coerce_text_plan_blueprint(raw)


def _plan_json_candidates(raw: str) -> list[str]:
    candidates = [raw]
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence_match and fence_match.group(1).strip():
        candidates.append(fence_match.group(1).strip())
    balanced = _extract_balanced_json_object(raw)
    if balanced:
        candidates.append(balanced)
    return list(dict.fromkeys(item for item in candidates if item))


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _coerce_plan_blueprint(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("content"), str):
        nested = _parse_plan_blueprint(value["content"])
        if nested is not None:
            return nested
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        for alias in ("steps", "plan", "items"):
            aliased = value.get(alias)
            if isinstance(aliased, list):
                tasks = aliased
                break
    if not isinstance(tasks, list):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            tasks = [
                {
                    "task_id": node.get("id"),
                    "description": node.get("label") or node.get("description"),
                    "level": node.get("level", 0),
                }
                for node in nodes
                if isinstance(node, dict)
            ]
    if not isinstance(tasks, list) or not tasks:
        return None
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if isinstance(task, str):
            task = {
                "task_id": f"task-{index + 1}",
                "description": task,
                "level": index,
            }
        if not isinstance(task, dict):
            return None
        task_id = str(task.get("task_id") or task.get("id") or f"task-{index + 1}").strip()
        description = str(
            task.get("description")
            or task.get("label")
            or task.get("title")
            or task.get("name")
            or ""
        ).strip()
        if not task_id or not description:
            return None
        raw_level = task.get("level", 0)
        try:
            level = int(raw_level)
        except (TypeError, ValueError):
            level = 0
        if level < 0:
            level = 0
        normalized_tasks.append(
            {
                "task_id": task_id,
                "description": description,
                "level": level,
            }
        )
    task_ids = {task["task_id"] for task in normalized_tasks}
    if len(task_ids) != len(normalized_tasks):
        return None
    normalized_edges = []
    raw_edges = value.get("edges")
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("from", edge.get("source", ""))).strip()
            target = str(edge.get("to", edge.get("target", ""))).strip()
            if source in task_ids and target in task_ids:
                normalized_edges.append({"from": source, "to": target})
    if len(normalized_tasks) > 1 and not normalized_edges:
        for index in range(len(normalized_tasks) - 1):
            normalized_edges.append(
                {
                    "from": normalized_tasks[index]["task_id"],
                    "to": normalized_tasks[index + 1]["task_id"],
                }
            )
    return {"tasks": normalized_tasks, "edges": normalized_edges}


def _coerce_text_plan_blueprint(text: str) -> dict[str, object] | None:
    lines = [
        _strip_plan_prefix(line.strip())
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    task_lines = [line for line in lines if line]
    if not task_lines:
        return None
    if len(task_lines) == 1 and len(task_lines[0].split()) < 4:
        return None
    tasks = []
    for index, line in enumerate(task_lines):
        tasks.append(
            {
                "task_id": f"task-{index + 1}",
                "description": line,
                "level": index,
            }
        )
    edges = []
    for index in range(len(tasks) - 1):
        edges.append({"from": tasks[index]["task_id"], "to": tasks[index + 1]["task_id"]})
    return {"tasks": tasks, "edges": edges}


def _strip_plan_prefix(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[\).\-\:]\s+)", "", line).strip()
    cleaned = re.sub(r"^\s*(?:step|task)\s+\d+\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _build_plan_result(
    *,
    final_response: str | None,
    blueprint: dict[str, object] | None,
    steps: list[object],
    total_usage: dict[str, int],
    ai_logs: list[str],
    system_logs: list[str],
    metadata: dict[str, object],
    elapsed_ms: int,
    error_message: str | None = None,
) -> dict[str, object]:
    return {
        "final_response": final_response,
        "steps": [step.to_dict() for step in steps],
        "total_usage": dict(total_usage),
        "ai_logs": list(ai_logs),
        "system_logs": list(system_logs),
        "stage_traces": [],
        "verification_results": [],
        "metadata": dict(metadata),
        "state": "PLANNING",
        "blueprint": blueprint,
        "error_message": error_message,
        "elapsed_ms": elapsed_ms,
        "backend_used": metadata.get("backend_used", "opencode"),
        "budget_state": None,
        "usage_sample": {
            "prompt_tokens": int(total_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(total_usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(total_usage.get("total_tokens", 0) or 0),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
