from __future__ import annotations

import json
import logging
import os
import sysconfig
import inspect
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.logging_utils import configure_logging

from .context_builder import ContextBuilderService
from .kernel import DevenvKernel
from .models import PlanningMode, PreparedPromptRequest, PrivacyModeState, RunConfig, ToolReadiness
from .setup import inspect_setup
from .tooling import build_runtime_tools
from .workspace import WorkspaceBrowser

logger = logging.getLogger(__name__)
DEFAULT_WEB_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "moonshotai/kimi-k2-instruct",
)


class AccessPolicy:
    def __init__(self) -> None:
        self.session_access: dict[str, bool] = {"codex": False, "opencode": False}
        self.backend_access: dict[str, bool] = {"opencode": False}

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


def _normalize_replay_error(message: str) -> str:
    cleaned = " ".join(str(message or "").split())
    if not cleaned:
        return "I couldn't complete that replayed answer."
    if "user rejected permission to use this specific tool call" in cleaned.lower():
        return "Permission to use a required tool call was denied."
    if cleaned.endswith("."):
        return cleaned
    return f"{cleaned}."


def _collapse_repeated_blocks(content: str | None) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    if not blocks:
        return raw
    deduped_blocks: list[str] = []
    for block in blocks:
        if not deduped_blocks or deduped_blocks[-1] != block:
            deduped_blocks.append(block)
    collapsed = "\n\n".join(deduped_blocks).strip()
    return collapsed or raw


def _sanitize_replay_text(content: object) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    if not raw.startswith("{") or "\n" not in raw:
        return _collapse_repeated_blocks(raw)

    readable_lines: list[str] = []
    replay_errors: list[str] = []
    tool_failures: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        part = payload.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text_value = str(part.get("text") or "").strip()
            if text_value:
                readable_lines.append(text_value)
            continue

        event_payload = payload.get("payload")
        if isinstance(event_payload, dict) and event_payload.get("type") == "agent_message":
            message = str(event_payload.get("message") or "").strip()
            if message:
                readable_lines.append(message)
            continue

        if payload.get("type") == "error":
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                replay_errors.append(
                    _normalize_replay_error(
                        str((error_payload.get("data") or {}).get("message") or error_payload.get("message") or error_payload.get("name") or "")
                    )
                )
            continue

        if payload.get("type") == "tool_use" and isinstance(part, dict) and part.get("tool") == "invalid":
            state = part.get("state")
            if isinstance(state, dict):
                input_payload = state.get("input")
                if isinstance(input_payload, dict):
                    error_message = str(input_payload.get("error") or "").strip()
                    if error_message:
                        tool_failures.append(error_message)

    unique_lines: list[str] = []
    for line in readable_lines:
        if line and line not in unique_lines:
            unique_lines.append(line)
    if unique_lines:
        return _collapse_repeated_blocks("\n\n".join(unique_lines))

    if replay_errors:
        return _collapse_repeated_blocks(replay_errors[0])
    if tool_failures:
        return "A required tool call was unavailable while replaying that answer."
    return "I couldn't produce a readable answer from that replay."


class DevenvWebApp:
    def __init__(self, config: RunConfig, port: int = 4173, *, memory=None, ai=None) -> None:
        self.config = config
        self.port = port
        self.static_root = _resolve_static_root()
        self.kernel = DevenvKernel(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
            memory=memory,
            ai=ai,
        )
        for tool in build_runtime_tools(self.kernel.memory):
            self.kernel.register_tool(tool)
        self.workspace = WorkspaceBrowser(config.workspace_path)
        self.context_builder = ContextBuilderService(
            config.workspace_path,
            memory=self.kernel.memory,
            provider_configs=config.external_session_configs,
        )
        self.context_builder.set_runtime_allowed_providers(set())
        self.kernel.context_builder = self.context_builder
        self.access_policy = AccessPolicy()

    def create_handler(self):
        return partial(DevenvRequestHandler, app=self)

    def create_server(self) -> ThreadingHTTPServer:
        return ThreadingHTTPServer(("127.0.0.1", self.port), self.create_handler())

    def serve(self) -> None:
        server = self.create_server()
        logger.info("Starting Devenv web server: url=http://127.0.0.1:%s workspace=%s", self.port, self.config.workspace_path)
        print(f"Devenv website running at http://127.0.0.1:{server.server_address[1]}")
        try:
            server.serve_forever()
        finally:
            self.kernel.close()

    def build_health_payload(self) -> dict[str, object]:
        model = getattr(self.kernel.ai, "model", "unknown")
        ai_statuses = getattr(self.kernel.ai, "status", lambda: {})()
        active_backend = getattr(self.kernel.ai, "last_backend_used", "groq")
        active_provider_label = "OpenCode CLI" if active_backend == "opencode" else "Groq"
        setup = inspect_setup(self.config, include_optional=True)
        privacy = PrivacyModeState(no_memory=self.config.no_memory, incognito=self.config.incognito)
        tool_readiness = self._build_tool_readiness()
        return {
            "workspace_path": self.config.workspace_path,
            "port": self.port,
            "tools": sorted(self.kernel.tools),
            "status": "ok",
            "ai_provider": active_provider_label,
            "ai_model": model,
            "available_models": self._available_models(current_model=model),
            "context_builder_enabled": True,
            "context_sources": [source.to_dict() for source in self.context_builder.list_sources()],
            "access_policy": self.access_policy.snapshot(),
            "ai_backends": {name: status.to_dict() for name, status in ai_statuses.items()},
            "active_backend": active_backend,
            "preferred_backend": getattr(self.kernel.ai, "preferred_backend", "auto"),
            "indexing": self.context_builder.indexing_status(),
            "setup": setup.to_dict(),
            "performance_mode": self.config.performance_mode,
            "privacy": privacy.to_dict(),
            "tool_readiness": {name: readiness.to_dict() for name, readiness in tool_readiness.items()},
        }

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
        configured_models = [item.strip() for item in configured.split(",") if item.strip()]
        ordered: list[str] = []
        for model_name in [current_model, *configured_models, *DEFAULT_WEB_MODELS]:
            if model_name and model_name not in ordered:
                ordered.append(model_name)
        return ordered

    def build_files_payload(self, relative_path: str = "") -> dict[str, object]:
        return {
            "path": relative_path,
            "entries": [entry.to_dict() for entry in self.workspace.list_entries(relative_path)],
        }

    def build_file_payload(self, relative_path: str) -> dict[str, object]:
        return {"path": relative_path, **self.workspace.read_file_preview(relative_path)}

    def build_context_sources_payload(self) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.context_builder.list_sources()],
            "access_policy": self.access_policy.snapshot(),
        }

    def build_context_sessions_payload(self, provider_name: str) -> dict[str, object]:
        self._require_provider_access(provider_name)
        return {
            "provider": provider_name,
            "sessions": [session.to_dict() for session in self.context_builder.list_sessions(provider_name)],
        }

    def build_context_session_payload(self, provider_name: str, session_id: str) -> dict[str, object]:
        self._require_provider_access(provider_name)
        detail = self.context_builder.get_session(provider_name, session_id)
        return detail.to_dict()

    def build_prepared_prompt_payload(self, payload: dict[str, object]) -> dict[str, object]:
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Missing required field: task")
        provider = payload.get("provider")
        session_ids_raw = payload.get("session_ids") or []
        if not isinstance(session_ids_raw, list) or any(not isinstance(item, str) for item in session_ids_raw):
            raise ValueError("session_ids must be a list of strings")
        include_workspace_scan = bool(payload.get("include_workspace_scan", True))
        include_prior_context = bool(payload.get("include_prior_context", True))
        output_format = str(payload.get("output_format", "compact"))
        request = PreparedPromptRequest(
            task=task.strip(),
            provider=provider if isinstance(provider, str) and provider.strip() else None,
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
        backend_preference: str = "auto",
        session_budget_tokens: int | None = None,
    ) -> dict[str, object]:
        execute_turn = self.kernel.execute_turn
        kwargs = {
            "max_consecutive_tools": max_consecutive_tools or self.config.max_consecutive_tools,
            "planning_mode": planning_mode,
            "continue_plan": continue_plan,
            "local_only": local_only,
        }
        parameters = inspect.signature(execute_turn).parameters
        if "backend_preference" in parameters:
            kwargs["backend_preference"] = backend_preference
        if "opencode_enabled" in parameters:
            kwargs["opencode_enabled"] = self.access_policy.can_use_backend("opencode")
        if "session_budget_tokens" in parameters:
            kwargs["session_budget_tokens"] = session_budget_tokens
        result = execute_turn(prompt, **kwargs).to_dict()
        result["final_response"] = _sanitize_replay_text(result.get("final_response"))
        result["error_message"] = _sanitize_replay_text(result.get("error_message"))
        metadata = dict(result.get("metadata") or {})
        result["backend_used"] = metadata.get("backend_used", "groq")
        result["budget_state"] = metadata.get("budget_state")
        result["usage_sample"] = {
            "prompt_tokens": int(result.get("total_usage", {}).get("prompt_tokens", 0) or 0),
            "completion_tokens": int(result.get("total_usage", {}).get("completion_tokens", 0) or 0),
            "total_tokens": int(result.get("total_usage", {}).get("total_tokens", 0) or 0),
        }
        return result

    def set_model(self, model: str) -> dict[str, object]:
        cleaned = model.strip()
        if not cleaned:
            raise ValueError("Missing required field: model")
        if hasattr(self.kernel.ai, "set_model"):
            self.kernel.ai.set_model(cleaned)
        else:
            self.kernel.ai.model = cleaned
        return {
            "ai_provider": getattr(self.kernel.ai, "provider_label", "Groq"),
            "ai_model": cleaned,
            "available_models": self._available_models(current_model=cleaned),
        }

    def update_session_access(self, provider: str, allowed: bool) -> dict[str, object]:
        if provider not in {"codex", "opencode"}:
            raise ValueError("provider must be one of: codex, opencode")
        snapshot = self.access_policy.set_session_access(provider, allowed)
        allowed_providers = {name for name, permitted in self.access_policy.session_access.items() if permitted}
        self.context_builder.set_runtime_allowed_providers(allowed_providers)
        return snapshot

    def update_backend_access(self, backend: str, allowed: bool) -> dict[str, object]:
        if backend != "opencode":
            raise ValueError("backend must be: opencode")
        return self.access_policy.set_backend_access(backend, allowed)

    def _require_provider_access(self, provider_name: str) -> None:
        if not self.access_policy.can_access_provider(provider_name):
            raise PermissionError(f"Access to {provider_name} sessions requires explicit user permission.")


class DevenvRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: DevenvWebApp, **kwargs):
        self.app = app
        super().__init__(*args, directory=str(app.static_root), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._write_json(HTTPStatus.OK, self.app.build_health_payload())
                return
            if parsed.path == "/api/files":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(HTTPStatus.OK, self.app.build_files_payload(relative_path))
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(HTTPStatus.OK, self.app.build_file_payload(relative_path))
                return
            if parsed.path == "/api/context-sources":
                self._write_json(HTTPStatus.OK, self.app.build_context_sources_payload())
                return
            if parsed.path.startswith("/api/context-sources/"):
                payload = self._match_context_source_path(parsed.path)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                provider_name, session_id = payload
                if session_id is None:
                    self._write_json(HTTPStatus.OK, self.app.build_context_sessions_payload(provider_name))
                    return
                self._write_json(HTTPStatus.OK, self.app.build_context_session_payload(provider_name, session_id))
                return
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError) as exc:
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
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "provider and allowed are required"})
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
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "backend and allowed are required"})
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
            if not isinstance(model, str):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required field: model"})
                return
            try:
                result = self.app.set_model(model)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, result)
            return
        if parsed.path != "/api/turn":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required field: prompt"})
            return

        max_consecutive_tools = payload.get("max_consecutive_tools")
        if max_consecutive_tools is not None and not isinstance(max_consecutive_tools, int):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "max_consecutive_tools must be an integer"})
            return
        planning_mode_value = payload.get("planning_mode", PlanningMode.AUTO.value)
        if not isinstance(planning_mode_value, str):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "planning_mode must be a string"})
            return
        try:
            planning_mode = PlanningMode(planning_mode_value)
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "planning_mode must be one of: auto, force_plan, force_direct"})
            return
        continue_plan = payload.get("continue_plan", False)
        if not isinstance(continue_plan, bool):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "continue_plan must be a boolean"})
            return
        local_only = payload.get("local_only", False)
        if not isinstance(local_only, bool):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "local_only must be a boolean"})
            return
        backend_preference = payload.get("backend_preference", "auto")
        if not isinstance(backend_preference, str):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "backend_preference must be a string"})
            return
        session_budget_tokens = payload.get("session_budget_tokens")
        if session_budget_tokens is not None and not isinstance(session_budget_tokens, int):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "session_budget_tokens must be an integer"})
            return

        try:
            result = self.app.run_turn(
                prompt=prompt,
                max_consecutive_tools=max_consecutive_tools,
                planning_mode=planning_mode,
                continue_plan=continue_plan,
                local_only=local_only,
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
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "context-sources" and parts[3] == "sessions":
            return parts[2], None
        if len(parts) == 5 and parts[0] == "api" and parts[1] == "context-sources" and parts[3] == "sessions":
            return parts[2], parts[4]
        return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Devenv website runtime.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--max-consecutive-tools", type=int, default=5)
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = RunConfig(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
        max_consecutive_tools=args.max_consecutive_tools,
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

    data_candidate = Path(sysconfig.get_path("data")).resolve() / "share" / "devenv" / "interface" / "website"
    if _is_valid_static_root(data_candidate):
        return data_candidate

    raise FileNotFoundError(
        "Devenv web static assets were not found. "
        "Set DEVENV_STATIC_ROOT or reinstall the package with bundled web assets."
    )


def _is_valid_static_root(path: Path) -> bool:
    return path.is_dir() and (path / "index.html").is_file() and (path / "main.js").is_file() and (path / "styles.css").is_file()


if __name__ == "__main__":
    raise SystemExit(main())
