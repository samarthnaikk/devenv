from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.logging_utils import configure_logging

from .context_builder import ContextBuilderService
from .kernel import DevenvKernel
from .models import DEFAULT_MAX_CONSECUTIVE_TOOLS, RunConfig, RuntimeTurnResult
from .tooling import build_runtime_tools
from .tui_theme import (
    BLUE,
    ERROR as ERROR_COLOR,
    ON_TEAL,
    ROLE_COLORS,
    TEAL,
    TEXT,
    TEXT_MUTED,
    WARN,
    CSS as TUI_CSS,
)
from .web import AccessPolicy, DEFAULT_LLAMACPP_MODELS, DEFAULT_OLLAMA_MODELS, DEFAULT_WEB_MODELS

try:
    from rich.markup import escape as _rich_escape
except Exception:  # pragma: no cover - rich ships with textual, but stay defensive
    def _rich_escape(text: str) -> str:  # type: ignore[misc]
        return text

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Footer,
        Input,
        LoadingIndicator,
        ProgressBar,
        RichLog,
        Static,
    )

    TEXTUAL_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path for environments without textual
    work = None
    App = object
    ComposeResult = object
    Horizontal = object
    Vertical = object
    VerticalScroll = object
    Footer = object
    Input = object
    LoadingIndicator = object
    ProgressBar = object
    RichLog = object
    Static = object
    TEXTUAL_AVAILABLE = False

BACKENDS = ("opencode", "ollama", "llama_cpp", "codex")
SESSION_PROVIDERS = ("codex", "opencode")
LOG_LEVEL_COLORS = {
    logging.DEBUG: TEXT_MUTED,
    logging.INFO: TEAL,
    logging.WARNING: WARN,
    logging.ERROR: ERROR_COLOR,
    logging.CRITICAL: ERROR_COLOR,
}
# Third-party loggers are noisy and not part of devenv's activity story.
QUIET_LOGGERS = (
    "sentence_transformers",
    "transformers",
    "lancedb",
    "urllib3",
    "httpx",
    "httpcore",
    "PIL",
    "filelock",
)


class Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _style(text: str, *codes: str) -> str:
    return "".join(codes) + text + Ansi.RESET


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", str(text or ""))


@dataclass
class TUICommandResult:
    message: str
    should_exit: bool = False


@dataclass(frozen=True)
class PaletteEntry:
    entry_id: str
    label: str
    command: str
    keywords: str


@dataclass(frozen=True)
class RetrievalOutcome:
    """The retrieval engine's output, reshaped for presentation in the TUI."""

    query: str
    context: str = ""
    session_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0


class TUILogBridge(logging.Handler):
    """Forward Python log records into a thread-safe queue for the TUI to drain."""

    def __init__(self, log_queue: "queue.Queue[tuple[float, int, str, str]]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return
        self._queue.put((record.created, record.levelno, record.name, message))


def _format_turn_result_lines(result: RuntimeTurnResult) -> list[str]:
    lines: list[str] = []
    for trace in result.stage_traces:
        checkpoint_label = f" checkpoint={trace.checkpoint_id}" if trace.checkpoint_id is not None else ""
        state = "ok" if trace.success else "failed"
        color = "green" if trace.success else "red"
        lines.append(f"[dim]stage[/] {trace.stage}{checkpoint_label} -> [{color}]{state}[/]. {trace.summary}")
        for log_line in trace.logs:
            cleaned_log = str(log_line or "").strip()
            if cleaned_log:
                lines.append(f"[dim]  {cleaned_log}[/]")
    for log_line in result.system_logs:
        cleaned_log = str(log_line or "").strip()
        if cleaned_log:
            lines.append(f"[dim]system[/] {cleaned_log}")
    for log_line in result.ai_logs:
        cleaned_log = str(log_line or "").strip()
        if cleaned_log:
            lines.append(f"[dim]thinking[/] {cleaned_log}")
    for step in result.steps:
        if step.is_sandboxed_violation:
            lines.append(f"[yellow]sandbox[/] {step.output}")
        else:
            state = "success" if step.success else "failure"
            color = "green" if step.success else "red"
            lines.append(f"[dim]tool[/] {step.tool_name} -> [{color}]{state}[/]")
    if result.error_message:
        lines.append(f"[red]error[/] {result.error_message}")
    if result.final_response:
        lines.append(f"[bold cyan]Assistant[/] {result.final_response}")
    elif not lines:
        lines.append("[yellow]Assistant[/] The runtime completed without producing a visible response.")
    return lines


def _collapse_text(text: str, *, limit: int = 500) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1].rstrip() + "…"
    return collapsed


_CONTEXT_LINE_STYLES: dict[str, tuple[str, str]] = {
    "user asked": ("User asked", ROLE_COLORS["user"]),
    "assistant reported": ("Assistant", ROLE_COLORS["assistant"]),
    "tool output": ("Tool output", ROLE_COLORS["tool"]),
}


def _split_context_line(text: str) -> tuple[str, str, str]:
    prefix, separator, rest = text.partition(":")
    key = prefix.strip().lower()
    if separator and key in _CONTEXT_LINE_STYLES:
        label, color = _CONTEXT_LINE_STYLES[key]
        return label, color, rest.strip()
    if key.startswith("session "):
        return "Session", ROLE_COLORS["session"], text
    return "", "", text


def _context_body_lines(context: str) -> list[str]:
    lines: list[str] = []
    for raw_line in context.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("##"):
            continue
        lines.append(stripped[2:].strip() if stripped.startswith("- ") else stripped)
    return lines


def _retrieval_metrics(outcome: RetrievalOutcome) -> str:
    if outcome.metadata.get("card_context_state") == "reused_prior_cards":
        parts = [f"{outcome.metadata.get('card_context_count', 0)} card(s)", "engine: interaction-memory"]
        if outcome.elapsed_ms:
            parts.append(f"{outcome.elapsed_ms} ms")
        return "  ·  ".join(parts)
    providers = ", ".join(outcome.metadata.get("context_match_providers", []) or []) or "—"
    parts = [
        f"{len(outcome.session_ids)} session(s)",
        f"providers: {providers}",
    ]
    if outcome.elapsed_ms:
        parts.append(f"{outcome.elapsed_ms} ms")
    parts.append(f"index: {'ready' if outcome.metadata.get('index_ready') else 'building'}")
    return "  ·  ".join(parts)


def _format_retrieval_result_lines(outcome: RetrievalOutcome) -> list[str]:
    query = _rich_escape(outcome.query or "")
    lines = [f"[b {TEAL}]Query[/]  [b {TEXT}]{query}[/]"]
    lines.append(f"[{TEXT_MUTED}]{_rich_escape(_retrieval_metrics(outcome))}[/]")
    body_lines = _context_body_lines(outcome.context)
    if not body_lines:
        reason = _rich_escape(
            str(outcome.metadata.get("context_match_reason") or "No prior sessions matched.")
        )
        lines.append("")
        lines.append(f"[{WARN}]no matches[/]  [{TEXT_MUTED}]{reason}[/]")
        if outcome.session_ids:
            lines.append("")
            lines.append(f"[b {BLUE}]Sessions[/]")
            for session_id in outcome.session_ids:
                lines.append(f"  [{TEAL}]◆[/] [{TEXT}]{_rich_escape(session_id)}[/]")
        return lines
    if outcome.session_ids:
        lines.append("")
        lines.append(f"[b {BLUE}]Sessions[/]")
        for session_id in outcome.session_ids:
            lines.append(f"  [{TEAL}]◆[/] [{TEXT}]{_rich_escape(session_id)}[/]")
    lines.append("")
    for text in body_lines:
        label, color, body = _split_context_line(text)
        body = _rich_escape(_collapse_text(body))
        if label == "Session":
            lines.append(f"[{TEXT_MUTED}]·[/] [{TEXT_MUTED}]{body}[/]")
        elif label:
            lines.append(f"[{TEXT_MUTED}]{label:>16}[/] [{TEXT_MUTED}]│[/] [{color}]{body}[/]")
        else:
            lines.append(f"[{TEXT_MUTED}]·[/] [{TEXT}]{body}[/]")
    return lines


def _format_log_line(created: float, levelno: int, name: str, message: str) -> str:
    timestamp = datetime.fromtimestamp(created).strftime("%H:%M:%S")
    color = LOG_LEVEL_COLORS.get(levelno, TEXT_MUTED)
    level = logging.getLevelName(levelno)
    source = name if len(name) <= 24 else name[-23:]
    return (
        f"[{TEXT_MUTED}]{timestamp}[/] [{color}]{level:<7}[/] "
        f"[{BLUE}]{_rich_escape(source)}[/] [{TEXT}]{_rich_escape(message)}[/]"
    )


def _retrieval_plain_text(outcome: RetrievalOutcome) -> str:
    body_lines = _context_body_lines(outcome.context)
    if not body_lines:
        reason = str(outcome.metadata.get("context_match_reason") or "No prior sessions matched.")
        return f'No matches for "{outcome.query}".\n{reason}'
    lines = [f'Retrieved context for "{outcome.query}":']
    if outcome.session_ids:
        lines.append(f"Sessions: {', '.join(outcome.session_ids)}")
    lines.append("")
    for text in body_lines:
        lines.append(f"- {_collapse_text(text)}")
    return "\n".join(lines)


class DevenvTUIController:
    def __init__(
        self,
        config: RunConfig,
        *,
        kernel: DevenvKernel | None = None,
        prompt_input: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        self.kernel = kernel or DevenvKernel(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
        )
        self.prompt_input = prompt_input or input
        self.context_builder = ContextBuilderService(
            config.workspace_path,
            memory=self.kernel.memory,
            provider_configs=config.external_session_configs,
            performance_mode=config.performance_mode,
        )
        self.context_builder.set_runtime_allowed_providers(set())
        self.kernel.context_builder = self.context_builder
        for tool in build_runtime_tools(
            self.kernel.memory,
            context_builder=self.context_builder,
        ):
            self.kernel.register_tool(tool)
        self.access_policy = AccessPolicy()
        self.mode = "retrieve"
        self.preferred_backend = getattr(self.kernel.ai, "preferred_backend", "opencode") or "opencode"
        self._load_persisted_state()
        self._apply_runtime_preferences()

    def close(self) -> None:
        self.kernel.close()

    def _state_file_path(self) -> Path:
        return Path(self.config.workspace_path) / ".devenv" / "tui_state.json"

    def _persisted_backend_models(self) -> dict[str, str]:
        backend_models = getattr(self.kernel.ai, "backend_models", {})
        if not isinstance(backend_models, dict):
            return {}
        persisted: dict[str, str] = {}
        for backend in BACKENDS:
            model_name = str(backend_models.get(backend, "") or "").strip()
            if model_name:
                persisted[backend] = model_name
        return persisted

    def _persist_state(self) -> None:
        payload = {
            "preferred_backend": self.preferred_backend,
            "mode": self.mode,
            "backend_access": dict(self.access_policy.backend_access),
            "session_access": dict(self.access_policy.session_access),
            "backend_models": self._persisted_backend_models(),
        }
        state_path = self._state_file_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load_persisted_state(self) -> None:
        state_path = self._state_file_path()
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        preferred_backend = str(payload.get("preferred_backend", "") or "").strip().lower()
        if preferred_backend in BACKENDS:
            self.preferred_backend = preferred_backend
        persisted_mode = str(payload.get("mode", "") or "").strip().lower()
        if persisted_mode in {"retrieve", "solve"}:
            self.mode = persisted_mode
        backend_access = payload.get("backend_access")
        if isinstance(backend_access, dict):
            for backend in BACKENDS:
                if backend in backend_access:
                    self.access_policy.set_backend_access(backend, bool(backend_access.get(backend)))
        session_access = payload.get("session_access")
        if isinstance(session_access, dict):
            for provider in SESSION_PROVIDERS:
                if provider in session_access:
                    self.access_policy.set_session_access(provider, bool(session_access.get(provider)))
        backend_models = payload.get("backend_models")
        if isinstance(backend_models, dict):
            for backend in BACKENDS:
                model_name = str(backend_models.get(backend, "") or "").strip()
                if not model_name:
                    continue
                if hasattr(self.kernel.ai, "set_backend_model"):
                    self.kernel.ai.set_backend_model(backend, model_name)
                elif backend == self.preferred_backend and hasattr(self.kernel.ai, "set_model"):
                    self.kernel.ai.set_model(model_name)

    def _apply_runtime_preferences(self) -> None:
        allowed_providers = {
            name
            for name, allowed in self.access_policy.session_access.items()
            if allowed
        }
        self.context_builder.set_runtime_allowed_providers(allowed_providers)
        if hasattr(self.kernel.ai, "set_backend_preference"):
            self.kernel.ai.set_backend_preference(
                self.preferred_backend,
                opencode_enabled=self.access_policy.can_use_backend("opencode"),
                ollama_enabled=self.access_policy.can_use_backend("ollama"),
                llama_cpp_enabled=self.access_policy.can_use_backend("llama_cpp"),
                codex_enabled=self.access_policy.can_use_backend("codex"),
            )

    def handle_input(self, raw_text: str) -> TUICommandResult | None:
        prompt = raw_text.strip()
        if not prompt:
            return None
        if prompt.lower() in {"exit", "quit"}:
            return TUICommandResult("Closing Devenv TUI.", should_exit=True)
        if prompt.startswith("/"):
            return self.handle_command(prompt)
        return None

    def handle_command(self, raw_command: str) -> TUICommandResult:
        parts = raw_command.strip().split()
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/help", "/?"}:
            return TUICommandResult(self.help_text())
        if command == "/status":
            return TUICommandResult(self.status_text())
        if command in {"/permission", "/permissions"}:
            return TUICommandResult(self._handle_permission_command(args))
        if command == "/backend":
            return TUICommandResult(self._handle_backend_command(args))
        if command == "/model":
            return TUICommandResult(self._handle_model_command(args))
        if command == "/providers":
            return TUICommandResult(self.providers_text())
        if command in {"/retrieve", "/recall"}:
            if not args:
                return TUICommandResult("Usage: /retrieve <query>")
            return TUICommandResult(_retrieval_plain_text(self.run_retrieval(" ".join(args))))
        if command == "/enable":
            return TUICommandResult(self.enable_all_sources())
        if command == "/sources":
            return TUICommandResult(self.sources_text())
        if command == "/mode":
            if not args:
                return TUICommandResult(f"Current mode: `{self.mode}`.")
            return TUICommandResult(self.set_mode(args[0]))
        if command == "/clear":
            session_id = self.kernel.reset_conversation()
            return TUICommandResult(f"Started a fresh thread. Session id: {session_id}")
        if command in {"/exit", "/quit"}:
            return TUICommandResult("Closing Devenv TUI.", should_exit=True)
        return TUICommandResult(
            f"Unknown command `{command}`.\n\n{self.help_text()}"
        )

    def help_text(self) -> str:
        return "\n".join(
            [
                _style("Commands", Ansi.BOLD, Ansi.CYAN),
                "/status                 Show active backend, model, and permissions",
                "/mode retrieve|solve    Switch TUI mode (solve is still in progress)",
                "/retrieve <query>       Retrieve prior sessions and chunks for a query",
                "/enable                 Enable all session sources (codex + opencode)",
                "/sources                Show session source status",
                "/permissions            Open the permission picker or show command help",
                "/permission backend <name> <on|off>",
                "/permission provider <codex|opencode> <on|off>",
                "/backend                Open the backend picker",
                "/backend <name>         Switch preferred backend",
                "/model                  Open the model picker",
                "/model <name>           Set model for the preferred backend",
                "/model <backend> <name> Set model for a specific backend",
                "/providers              Show session-source health",
                "/clear                  Start a fresh runtime thread",
                "/exit                   Quit the TUI",
            ]
        )

    def status_text(self) -> str:
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        preferred_backend = getattr(self.kernel.ai, "preferred_backend", self.preferred_backend) or self.preferred_backend
        current_model = getattr(self.kernel.ai, "model", "unknown")
        lines = [
            _style("Devenv Status", Ansi.BOLD, Ansi.CYAN),
            f"Workspace: {self.config.workspace_path}",
            f"Performance: {self.config.performance_mode}",
            f"Preferred backend: {preferred_backend}",
            f"Current model: {current_model}",
            "",
            _style("Backend Access", Ansi.BOLD, Ansi.BLUE),
        ]
        for backend in BACKENDS:
            allowed = self.access_policy.can_use_backend(backend)
            label = "on" if allowed else "off"
            color = Ansi.GREEN if allowed else Ansi.DIM
            backend_model = self._backend_model(statuses, backend)
            suffix = f" [{backend_model}]" if backend_model else ""
            lines.append(f"- {backend}: {_style(label, color)}{suffix}")
        lines.extend(["", _style("Session Providers", Ansi.BOLD, Ansi.MAGENTA)])
        for provider in SESSION_PROVIDERS:
            allowed = self.access_policy.can_access_provider(provider)
            label = "on" if allowed else "off"
            color = Ansi.GREEN if allowed else Ansi.DIM
            lines.append(f"- {provider}: {_style(label, color)}")
        return "\n".join(lines)

    def providers_text(self) -> str:
        sources = self.context_builder.list_sources()
        lines = [_style("Session Source Health", Ansi.BOLD, Ansi.CYAN)]
        if not sources:
            lines.append("No external session providers are configured.")
            return "\n".join(lines)
        for source in sources:
            state = "ready" if source.available else "unavailable"
            allowed = "allowed" if self.access_policy.can_access_provider(source.provider) else "blocked"
            lines.append(
                f"- {source.provider}: {state}, {allowed}, sessions={source.session_count}, root={source.root_path}"
            )
        return "\n".join(lines)

    def enabled_sources(self) -> list[str]:
        return [provider for provider in SESSION_PROVIDERS if self.access_policy.can_access_provider(provider)]

    def sources_text(self) -> str:
        lines = [_style("Session Sources", Ansi.BOLD, Ansi.CYAN)]
        for provider in SESSION_PROVIDERS:
            allowed = self.access_policy.can_access_provider(provider)
            label = "on" if allowed else "off"
            color = Ansi.GREEN if allowed else Ansi.DIM
            lines.append(f"- {provider}: {_style(label, color)}")
        lines.append("")
        lines.append(_style(f"Index: {self.index_status_text()}", Ansi.DIM))
        return "\n".join(lines)

    def index_status_text(self) -> str:
        if not self.enabled_sources():
            return "no sources enabled"
        try:
            status = self.context_builder.indexing_status()
        except Exception:
            return "unknown"
        if status.get("active"):
            return f"{status.get('message', 'indexing')} ({status.get('percent', 0)}%)"
        if status.get("completed"):
            return "ready"
        return str(status.get("message") or "idle")

    def set_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"retrieve", "solve"}:
            return "Mode must be `retrieve` or `solve`."
        self.mode = normalized
        self._persist_state()
        if normalized == "solve":
            return "Solve mode selected, but it is still in progress."
        return "Retrieval mode selected."

    def enable_all_sources(self) -> str:
        enabled = list(SESSION_PROVIDERS)
        for provider in enabled:
            self.access_policy.set_session_access(provider, True)
        self._apply_runtime_preferences()
        self._persist_state()
        return f"Enabled all session sources: {', '.join(enabled)}. Building indexes in the background."

    def set_source_enabled(self, provider: str, enabled: bool) -> str:
        if provider not in SESSION_PROVIDERS:
            return f"Unknown session source `{provider}`."
        self.access_policy.set_session_access(provider, enabled)
        self._apply_runtime_preferences()
        self._persist_state()
        state = "enabled" if enabled else "disabled"
        if enabled:
            return f"Session source `{provider}` {state}. Building the index in the background."
        return f"Session source `{provider}` {state}."

    def _handle_permission_command(self, args: list[str]) -> str:
        if not args:
            interactive = self._interactive_permission_picker()
            if interactive is not None:
                return interactive
            return "\n".join(
                [
                    _style("Permission Controls", Ansi.BOLD, Ansi.CYAN),
                    "Grant what the TUI is allowed to use before asking memory-heavy questions.",
                    "Examples:",
                    "/permission backend opencode on",
                    "/permission backend codex on",
                    "/permission provider codex on",
                    "/permission provider opencode on",
                ]
            )
        if len(args) != 3:
            return "Usage: /permission <backend|provider> <name> <on|off>"
        target_type, name, value = args[0].lower(), args[1].lower(), args[2].lower()
        if value not in {"on", "off"}:
            return "Permission values must be `on` or `off`."
        allowed = value == "on"
        if target_type == "backend":
            if name not in BACKENDS:
                return "Backends must be one of: opencode, ollama, llama_cpp, codex."
            self.access_policy.set_backend_access(name, allowed)
            self._apply_runtime_preferences()
            self._persist_state()
            return f"Backend `{name}` permission is now {value}."
        if target_type == "provider":
            if name not in SESSION_PROVIDERS:
                return "Providers must be one of: codex, opencode."
            self.access_policy.set_session_access(name, allowed)
            self._apply_runtime_preferences()
            self._persist_state()
            return f"Provider `{name}` permission is now {value}."
        return "Permission target must be `backend` or `provider`."

    def _handle_backend_command(self, args: list[str]) -> str:
        if not args:
            return self._interactive_backend_picker() or "Backend selection cancelled."
        if len(args) != 1:
            return "Usage: /backend <opencode|ollama|llama_cpp|codex>"
        backend = args[0].lower()
        if backend not in BACKENDS:
            return "Backends must be one of: opencode, ollama, llama_cpp, codex."
        self.preferred_backend = backend
        self._apply_runtime_preferences()
        self._persist_state()
        if not self.access_policy.can_use_backend(backend):
            return (
                f"Preferred backend set to `{backend}`, but it is still blocked. "
                f"Run `/permission backend {backend} on` before chatting."
            )
        return f"Preferred backend set to `{backend}`."

    def _handle_model_command(self, args: list[str]) -> str:
        if not args:
            return self._interactive_model_picker() or "Model selection cancelled."
        backend = self.preferred_backend
        if len(args) == 1:
            model = args[0]
        elif len(args) == 2:
            backend = args[0].lower()
            model = args[1]
            if backend not in BACKENDS:
                return "Backends must be one of: opencode, ollama, llama_cpp, codex."
        else:
            return "Usage: /model <name> or /model <backend> <name>"
        cleaned_model = model.strip()
        if not cleaned_model:
            return "Model name cannot be empty."
        if hasattr(self.kernel.ai, "set_backend_model"):
            self.kernel.ai.set_backend_model(backend, cleaned_model)
        elif hasattr(self.kernel.ai, "set_model"):
            self.kernel.ai.set_model(cleaned_model)
        else:
            self.kernel.ai.model = cleaned_model
        if backend == self.preferred_backend:
            if hasattr(self.kernel.ai, "set_model"):
                self.kernel.ai.set_model(cleaned_model)
            else:
                self.kernel.ai.model = cleaned_model
        self._persist_state()
        return f"Model for `{backend}` set to `{cleaned_model}`."

    def _prompt_choice(self, title: str, options: list[str], *, allow_cancel: bool = True) -> int | None:
        lines = [_style(title, Ansi.BOLD, Ansi.CYAN)]
        for index, option in enumerate(options, start=1):
            lines.append(f"{index}. {option}")
        if allow_cancel:
            lines.append("0. Cancel")
        prompt = "\n".join(lines) + "\n> "
        try:
            raw_value = self.prompt_input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if allow_cancel and raw_value in {"", "0"}:
            return None
        if not raw_value.isdigit():
            return None
        selected = int(raw_value) - 1
        if 0 <= selected < len(options):
            return selected
        return None

    def _prompt_text(self, prompt: str) -> str | None:
        try:
            value = self.prompt_input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return value or None

    def _interactive_permission_picker(self) -> str | None:
        target_index = self._prompt_choice("Permission Picker", ["Backend access", "Session provider access"])
        if target_index is None:
            return None
        if target_index == 0:
            backend_options = [
                f"{backend} [{'on' if self.access_policy.can_use_backend(backend) else 'off'}]"
                for backend in BACKENDS
            ]
            backend_index = self._prompt_choice("Select Backend", backend_options)
            if backend_index is None:
                return None
            value_index = self._prompt_choice("Set Backend Access", ["on", "off"])
            if value_index is None:
                return None
            return self._handle_permission_command(["backend", BACKENDS[backend_index], ("on", "off")[value_index]])
        provider_options = [
            f"{provider} [{'on' if self.access_policy.can_access_provider(provider) else 'off'}]"
            for provider in SESSION_PROVIDERS
        ]
        provider_index = self._prompt_choice("Select Session Provider", provider_options)
        if provider_index is None:
            return None
        value_index = self._prompt_choice("Set Provider Access", ["on", "off"])
        if value_index is None:
            return None
        return self._handle_permission_command(["provider", SESSION_PROVIDERS[provider_index], ("on", "off")[value_index]])

    def _interactive_backend_picker(self) -> str | None:
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        options = []
        for backend in BACKENDS:
            model = self._backend_model(statuses, backend)
            state = "on" if self.access_policy.can_use_backend(backend) else "off"
            suffix = f" [{model}]" if model else ""
            options.append(f"{backend} ({state}){suffix}")
        backend_index = self._prompt_choice("Backend Picker", options)
        if backend_index is None:
            return None
        return self._handle_backend_command([BACKENDS[backend_index]])

    def _interactive_model_picker(self) -> str | None:
        backend_options = []
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        for backend in BACKENDS:
            current = self._backend_model(statuses, backend) or "unset"
            backend_options.append(f"{backend} [{current}]")
        backend_index = self._prompt_choice("Choose Backend For Model", backend_options)
        if backend_index is None:
            return None
        backend = BACKENDS[backend_index]
        known_models: list[str] = []
        backend_models = getattr(self.kernel.ai, "backend_models", {})
        if isinstance(backend_models, dict):
            current = str(backend_models.get(backend, "") or "").strip()
            if current:
                known_models.append(current)
        defaults = {
            "opencode": ("opencode/claude-sonnet-4", "opencode/gpt-5-codex"),
            "ollama": ("qwen2.5:3b", "qwen2.5-coder:7b"),
            "llama_cpp": ("qwen2.5-coder.gguf", "deepseek-coder.gguf"),
            "codex": ("gpt-5-codex", "gpt-5-codex-high"),
        }
        for model in defaults.get(backend, ()):
            if model not in known_models:
                known_models.append(model)
        options = [*known_models, "Enter custom model…"]
        model_index = self._prompt_choice("Model Picker", options)
        if model_index is None:
            return None
        if model_index == len(options) - 1:
            custom_model = self._prompt_text("Custom model name\n> ")
            if custom_model is None:
                return None
            return self._handle_model_command([backend, custom_model])
        return self._handle_model_command([backend, options[model_index]])

    def _backend_model(self, statuses: dict[str, Any], backend: str) -> str:
        status = statuses.get(backend) if isinstance(statuses, dict) else None
        model = getattr(status, "model", "")
        if model:
            return str(model)
        backend_models = getattr(self.kernel.ai, "backend_models", {})
        if isinstance(backend_models, dict):
            return str(backend_models.get(backend, "") or "")
        if backend == self.preferred_backend:
            return str(getattr(self.kernel.ai, "model", "") or "")
        return ""

    def _available_opencode_models(self, current_model: str) -> list[str]:
        configured = os.getenv("DEVENV_AVAILABLE_MODELS", "")
        configured_models = [item.strip() for item in configured.split(",") if item.strip()]
        ordered: list[str] = []
        for model_name in [current_model, *configured_models, *DEFAULT_WEB_MODELS]:
            if model_name and model_name not in ordered:
                ordered.append(model_name)
        return ordered

    def _model_catalog(self) -> dict[str, list[str]]:
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        current_backend = getattr(self.kernel.ai, "preferred_backend", self.preferred_backend) or self.preferred_backend
        current_model = str(getattr(self.kernel.ai, "model", "") or "")
        catalog: dict[str, list[str]] = {
            "opencode": self._available_opencode_models(
                current_model if current_backend == "opencode" else self._backend_model(statuses, "opencode")
            ),
            "ollama": list(DEFAULT_OLLAMA_MODELS),
            "llama_cpp": list(DEFAULT_LLAMACPP_MODELS),
            "codex": [],
        }
        if isinstance(statuses, dict):
            for backend in BACKENDS:
                status = statuses.get(backend)
                metadata = dict(getattr(status, "metadata", {}) or {})
                reported = [str(item).strip() for item in metadata.get("models", []) or [] if str(item).strip()]
                ordered = list(catalog.get(backend, []))
                selected = self._backend_model(statuses, backend).strip()
                for model_name in [selected, *reported]:
                    if model_name and model_name not in ordered:
                        ordered.append(model_name)
                catalog[backend] = ordered
        return catalog

    def model_options_for_backend(self, backend: str) -> list[str]:
        return list(self._model_catalog().get(backend, []))

    def palette_entries(self, query: str = "") -> list[PaletteEntry]:
        statuses = getattr(self.kernel.ai, "status", lambda: {})()
        entries: list[PaletteEntry] = [
            PaletteEntry("status", "Show status", "/status", "status summary permissions backend model"),
            PaletteEntry("providers", "Show session providers", "/providers", "providers sessions health codex opencode"),
            PaletteEntry("clear", "Start fresh thread", "/clear", "clear reset thread conversation"),
            PaletteEntry("quit", "Quit TUI", "/quit", "quit exit close"),
        ]
        for backend in BACKENDS:
            state = "on" if self.access_policy.can_use_backend(backend) else "off"
            entries.append(
                PaletteEntry(
                    f"toggle_backend:{backend}",
                    f"Toggle backend {backend} [{state}]",
                    f"/permission backend {backend} {'off' if state == 'on' else 'on'}",
                    f"permission backend toggle {backend} {state}",
                )
            )
            model = self._backend_model(statuses, backend) or "unset"
            entries.append(
                PaletteEntry(
                    f"select_backend:{backend}",
                    f"Use backend {backend} [{model}]",
                    f"/backend {backend}",
                    f"backend preferred select {backend} model {model}",
                )
            )
            for model_name in self.model_options_for_backend(backend):
                entries.append(
                    PaletteEntry(
                        f"model:{backend}:{model_name}",
                        f"Set {backend} model to {model_name}",
                        f"/model {backend} {model_name}",
                        f"model {backend} {model_name}",
                    )
                )
        for provider in SESSION_PROVIDERS:
            state = "on" if self.access_policy.can_access_provider(provider) else "off"
            entries.append(
                PaletteEntry(
                    f"toggle_provider:{provider}",
                    f"Toggle session provider {provider} [{state}]",
                    f"/permission provider {provider} {'off' if state == 'on' else 'on'}",
                    f"permission provider session toggle {provider} {state}",
                )
            )
        normalized_query = query.strip().lower()
        if not normalized_query:
            return entries
        tokens = [token for token in normalized_query.split() if token]
        filtered: list[PaletteEntry] = []
        for entry in entries:
            haystack = f"{entry.label.lower()} {entry.command.lower()} {entry.keywords.lower()}"
            if all(token in haystack for token in tokens):
                filtered.append(entry)
        return filtered or entries

    def run_retrieval(self, query: str, *, max_lines: int = 12) -> RetrievalOutcome:
        self._apply_runtime_preferences()
        started = time.perf_counter()
        card_context, card_metadata = self._retrieve_card_context(query)
        if card_context:
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            return RetrievalOutcome(
                query=query,
                context=card_context,
                session_ids=(),
                metadata=card_metadata,
                elapsed_ms=elapsed_ms,
            )
        context, session_ids, metadata = self.context_builder.build_runtime_memory_context(
            query,
            max_lines=max_lines,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        return RetrievalOutcome(
            query=query,
            context=context,
            session_ids=tuple(session_ids),
            metadata=dict(metadata),
            elapsed_ms=elapsed_ms,
        )

    def _retrieve_card_context(self, query: str) -> tuple[str, dict[str, Any]]:
        memory = getattr(self.kernel, "memory", None)
        if memory is None or not hasattr(memory, "retrieve_cards"):
            return "", {}
        try:
            from core.memory.card_retrieval import DEFAULT_MIN_SCORE
            from core.memory.query_plan import build_query_plan
        except Exception:
            return "", {}
        lanes = build_query_plan(query)
        try:
            matches = memory.retrieve_cards(query, lanes=lanes, top_k=8)
        except Exception:
            return "", {}
        if not matches or max(match.score for match in matches) < DEFAULT_MIN_SCORE:
            return "", {}
        lines = ["## Interaction Memory"]
        for match in matches:
            card = match.card
            intent = " ".join(card.intent_text.split())[:200]
            answer = " ".join(card.answer_text.split())[:300]
            lines.append(
                f"- [{card.project or 'unknown'}] {intent} — {answer} (source: {card.provider}:{card.session_id[:8]})"
            )
        metadata = {
            "card_context_state": "reused_prior_cards",
            "card_context_count": len(matches),
            "card_context_sources": [f"{match.card.provider}:{match.card.session_id[:8]}" for match in matches],
            "index_ready": True,
        }
        return "\n".join(lines), metadata

    def run_prompt(self, prompt: str) -> RuntimeTurnResult:
        self._apply_runtime_preferences()
        return self.kernel.execute_turn(
            prompt,
            max_consecutive_tools=self.config.max_consecutive_tools,
            backend_preference=self.preferred_backend,
            opencode_enabled=self.access_policy.can_use_backend("opencode"),
            ollama_enabled=self.access_policy.can_use_backend("ollama"),
            llama_cpp_enabled=self.access_policy.can_use_backend("llama_cpp"),
            codex_enabled=self.access_policy.can_use_backend("codex"),
            no_memory=self.config.no_memory,
            incognito=self.config.incognito,
        )


def render_banner(config: RunConfig) -> None:
    line = _style("━" * 78, Ansi.DIM)
    print(line)
    print(
        _style("DEVENV CORE TUI", Ansi.BOLD, Ansi.CYAN)
        + f"  {_style('workspace', Ansi.DIM)} {config.workspace_path}"
    )
    print(
        f"{_style('mode', Ansi.GREEN)} retrieve   "
        f"{_style('sources', Ansi.BLUE)} /sources   "
        f"{_style('commands', Ansi.MAGENTA)} /retrieve /mode /help"
    )
    print(line)


def render_turn_result(result: RuntimeTurnResult) -> None:
    for line in _format_turn_result_lines(result):
        rendered = (
            line.replace("[bold cyan]", f"{Ansi.BOLD}{Ansi.CYAN}")
            .replace("[yellow]", Ansi.YELLOW)
            .replace("[red]", Ansi.RED)
            .replace("[green]", Ansi.GREEN)
            .replace("[dim]", Ansi.DIM)
            .replace("[/]", Ansi.RESET)
            .replace("[/yellow]", Ansi.RESET)
            .replace("[/red]", Ansi.RESET)
            .replace("[/green]", Ansi.RESET)
            .replace("[/dim]", Ansi.RESET)
        )
        print(rendered + (Ansi.RESET if not rendered.endswith(Ansi.RESET) else ""))


if TEXTUAL_AVAILABLE:
    class DevenvTextualApp(App[None]):
        CSS = TUI_CSS
        MAX_RESULT_CARDS = 12

        BINDINGS = [
            ("f1", "mode_retrieve", "Retrieve"),
            ("f2", "mode_solve", "Solve (WIP)"),
            ("f3", "toggle_codex", "Codex"),
            ("f4", "toggle_opencode", "OpenCode"),
            ("f5", "toggle_logs", "Logs"),
            ("ctrl+y", "copy_result", "Copy"),
            ("ctrl+e", "export_result", "Export"),
            ("ctrl+l", "clear_results", "Clear"),
            ("ctrl+q", "quit_app", "Quit"),
        ]

        def __init__(self, controller: DevenvTUIController) -> None:
            super().__init__()
            self.controller = controller
            self._busy = False
            self._last_outcome: RetrievalOutcome | None = None
            self._result_cards: list[Static] = []
            self._log_queue: "queue.Queue[tuple[float, int, str, str]]" = queue.Queue()
            self._bridge = TUILogBridge(self._log_queue)
            self._saved_handlers: list[logging.Handler] = []

        def compose(self) -> ComposeResult:
            yield Static("", id="header")
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Static("", id="mode-pills")
                    yield Static("SOURCES", classes="section-title")
                    yield Static("", id="sources-list")
                    yield Static("INDEX", classes="section-title")
                    yield ProgressBar(total=100, show_percentage=True, show_eta=False, id="index-bar")
                    yield Static("", id="index-info")
                with Vertical(id="results-pane"):
                    with Horizontal(id="result-bar"):
                        yield Static("Results", id="result-title")
                        yield LoadingIndicator(id="spinner")
                    yield VerticalScroll(id="results-list")
            with Vertical(id="log-panel"):
                yield Static("Activity  ·  F5 to hide", id="log-title")
                yield RichLog(id="log", markup=True, wrap=True)
            yield Input(placeholder="Ask a retrieval question and press Enter…", id="composer")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "DEVENV"
            self.sub_title = self.controller.config.workspace_path
            self._install_logging()
            self.query_one("#spinner", LoadingIndicator).display = False
            self._activity(f"TUI ready · workspace {self.controller.config.workspace_path}")
            if not self.controller.enabled_sources():
                self._activity(
                    "No session sources enabled. Run /enable or press F3/F4.", logging.WARNING
                )
            self._refresh_header()
            self._refresh_sidebar()
            self._refresh_sources()
            self._refresh_index()
            self.query_one("#composer", Input).focus()
            self.set_interval(0.25, self._drain_logs)
            self.set_interval(0.75, self._refresh_index)

        def on_unmount(self) -> None:
            self._restore_logging()

        # ------------------------------------------------------------------ logging
        def _install_logging(self) -> None:
            root = logging.getLogger()
            self._saved_handlers = [
                handler for handler in list(root.handlers) if not isinstance(handler, TUILogBridge)
            ]
            for handler in self._saved_handlers:
                root.removeHandler(handler)
            self._bridge.setLevel(logging.NOTSET)
            root.addHandler(self._bridge)
            for name in QUIET_LOGGERS:
                logging.getLogger(name).setLevel(logging.WARNING)

        def _restore_logging(self) -> None:
            root = logging.getLogger()
            if self._bridge in root.handlers:
                root.removeHandler(self._bridge)
            for handler in self._saved_handlers:
                root.addHandler(handler)
            self._saved_handlers = []

        def _activity(self, message: str, levelno: int = logging.INFO, name: str = "devenv") -> None:
            self._log_queue.put((time.time(), levelno, name, message))

        def _drain_logs(self) -> None:
            try:
                log = self.query_one("#log", RichLog)
            except Exception:  # pragma: no cover - widget may be gone during shutdown
                return
            written = 0
            while written < 300:
                try:
                    created, levelno, name, message = self._log_queue.get_nowait()
                except queue.Empty:
                    break
                log.write(_format_log_line(created, levelno, name, message))
                written += 1

        # ------------------------------------------------------------------ actions
        def action_mode_retrieve(self) -> None:
            self._set_mode("retrieve")

        def action_mode_solve(self) -> None:
            self._set_mode("solve")

        def action_toggle_codex(self) -> None:
            self._toggle_source("codex")

        def action_toggle_opencode(self) -> None:
            self._toggle_source("opencode")

        def action_toggle_logs(self) -> None:
            self.query_one("#log-panel", Vertical).toggle_class("hidden")

        def action_clear_results(self) -> None:
            for card in self._result_cards:
                card.remove()
            self._result_cards.clear()
            self._activity("Cleared results.")

        def action_copy_result(self) -> None:
            self._copy_last()

        def action_export_result(self) -> None:
            self._export_last()

        def action_quit_app(self) -> None:
            self.exit()

        def _set_mode(self, mode: str) -> None:
            message = self.controller.set_mode(mode)
            self._activity(f"mode: {message}")
            self._refresh_header()
            self._refresh_sidebar()

        def _toggle_source(self, provider: str) -> None:
            enabled = not self.controller.access_policy.can_access_provider(provider)
            message = self.controller.set_source_enabled(provider, enabled)
            self._activity(f"source: {message}", logging.INFO if enabled else logging.WARNING)
            self.notify(
                message,
                title="Source enabled" if enabled else "Source disabled",
                severity="information" if enabled else "warning",
            )
            self._refresh_sources()
            self._refresh_index()

        def _copy_last(self) -> None:
            if self._last_outcome is None:
                self.notify("Nothing to copy yet.", severity="warning")
                return
            self.copy_to_clipboard(_retrieval_plain_text(self._last_outcome))
            self.notify("Copied last retrieval result to the clipboard.")
            self._activity("Copied last result to clipboard.")

        def _export_path(self) -> Path:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return (
                Path(self.controller.config.workspace_path)
                / ".devenv"
                / "retrieval_exports"
                / f"retrieval-{stamp}.md"
            )

        def _export_last(self) -> None:
            if self._last_outcome is None:
                self.notify("Nothing to export yet.", severity="warning")
                return
            path = self._export_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_retrieval_plain_text(self._last_outcome) + "\n", encoding="utf-8")
            except OSError as exc:
                self.notify(f"Export failed: {exc}", severity="error")
                self._activity(f"export failed: {exc}", logging.ERROR)
                return
            self.notify(f"Exported to {path}")
            self._activity(f"Exported last result to {path}")

        # ------------------------------------------------------------------ input
        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id != "composer":
                return
            value = event.value.strip()
            event.input.value = ""
            if not value:
                return
            if value.startswith("/"):
                command_name = value.split()[0].lower()
                result = self.controller.handle_command(value)
                self._activity(f"command: {value}")
                if result.message:
                    self._mount_command_card(value, result.message)
                if command_name in {"/enable", "/permission", "/permissions"}:
                    self._refresh_sources()
                    self._refresh_index()
                if command_name == "/mode":
                    self._refresh_header()
                    self._refresh_sidebar()
                if result.should_exit:
                    self.exit()
                return
            if self._busy:
                self.notify("A retrieval is already running.", severity="warning")
                return
            if self.controller.mode == "solve":
                self._activity("Solve mode is still in progress.", logging.WARNING)
                self.notify("Solve mode is still in progress. Press F1 for retrieval.", severity="warning")
                return
            self._set_busy(True)
            self._activity(f"retrieving: {value}")
            self._run_retrieval(value)

        @work(thread=True)
        def _run_retrieval(self, query: str) -> None:
            try:
                outcome = self.controller.run_retrieval(query, max_lines=20)
            except Exception as exc:  # pragma: no cover - defensive UI path
                self.call_from_thread(self._render_failure, str(exc))
                return
            self.call_from_thread(self._render_result, outcome)

        def _render_result(self, outcome: RetrievalOutcome) -> None:
            self._last_outcome = outcome
            self._mount_card(_format_retrieval_result_lines(outcome))
            self._set_busy(False)
            providers = ", ".join(outcome.metadata.get("context_match_providers", []) or []) or "—"
            self._activity(
                f"retrieval: {len(outcome.session_ids)} session(s) in {outcome.elapsed_ms} ms "
                f"(providers: {providers})",
                logging.INFO if outcome.session_ids else logging.WARNING,
            )
            self._refresh_header()
            self.query_one("#composer", Input).focus()

        def _render_failure(self, error_message: str) -> None:
            self._set_busy(False)
            self._mount_card(
                [
                    f"[b {ERROR_COLOR}]Retrieval failed[/]",
                    f"[{TEXT_MUTED}]{_rich_escape(error_message)}[/]",
                ]
            )
            self._activity(f"retrieval failed: {error_message}", logging.ERROR)
            self.notify(f"Retrieval failed: {error_message}", severity="error")
            self.query_one("#composer", Input).focus()

        # ------------------------------------------------------------------ rendering
        def _mount_card(self, lines: list[str]) -> None:
            if not lines:
                lines = [f"[{TEXT_MUTED}](empty)[/]"]
            container = self.query_one("#results-list", VerticalScroll)
            card = Static("\n".join(lines), markup=True, classes="result-card")
            container.mount(card, before=0)
            self._result_cards.insert(0, card)
            while len(self._result_cards) > self.MAX_RESULT_CARDS:
                oldest = self._result_cards.pop()
                oldest.remove()

        def _mount_command_card(self, command: str, message: str) -> None:
            lines = [f"[b {TEAL}]Command[/]  [{TEXT}]{_rich_escape(command)}[/]"]
            for raw in _strip_ansi(message).splitlines():
                lines.append(f"[{TEXT}]{_rich_escape(raw)}[/]")
            if not message.strip():
                lines.append(f"[{TEXT_MUTED}](no output)[/]")
            self._mount_card(lines)

        def _set_busy(self, busy: bool) -> None:
            self._busy = busy
            try:
                self.query_one("#spinner", LoadingIndicator).display = busy
                self.query_one("#composer", Input).disabled = busy
                self.query_one("#result-title", Static).update("Retrieving…" if busy else "Results")
            except Exception:  # pragma: no cover - widget may be gone during shutdown
                pass

        def _refresh_header(self) -> None:
            if self.controller.mode == "retrieve":
                pill = f"[{ON_TEAL} on {TEAL}] RETRIEVE [/]"
            else:
                pill = f"[#2e3036 on {WARN}] SOLVE (WIP) [/]"
            summary = ""
            if self._last_outcome is not None:
                summary = (
                    f"   [{TEXT_MUTED}]last:[/] [{TEXT}]"
                    f"{len(self._last_outcome.session_ids)} sessions · "
                    f"{self._last_outcome.elapsed_ms} ms[/]"
                )
            workspace = _rich_escape(self.controller.config.workspace_path)
            text = f"[b {TEAL}]DEVENV[/]  [{TEXT_MUTED}]{workspace}[/]   {pill}{summary}"
            try:
                self.query_one("#header", Static).update(text)
            except Exception:  # pragma: no cover
                pass

        def _refresh_sidebar(self) -> None:
            if self.controller.mode == "retrieve":
                pills = f"[{ON_TEAL} on {TEAL}] RETRIEVE [/]  [{TEXT_MUTED}] SOLVE (WIP) [/]"
            else:
                pills = f"[{TEXT_MUTED}] RETRIEVE [/]  [#2e3036 on {WARN}] SOLVE (WIP) [/]"
            try:
                self.query_one("#mode-pills", Static).update(f"[b {TEXT_MUTED}]MODE[/]\n{pills}")
            except Exception:  # pragma: no cover
                pass

        def _refresh_sources(self) -> None:
            try:
                sources = self.controller.context_builder.list_sources()
            except Exception:  # pragma: no cover - defensive
                sources = []
            health = {source.provider: source for source in sources}
            lines: list[str] = []
            for provider in SESSION_PROVIDERS:
                allowed = self.controller.access_policy.can_access_provider(provider)
                color = TEAL if allowed else TEXT_MUTED
                pip = "●" if allowed else "○"
                if allowed:
                    count = getattr(health.get(provider), "session_count", 0) or 0
                    suffix = f"[{TEXT_MUTED}]{count}[/]"
                else:
                    suffix = f"[{TEXT_MUTED}]off[/]"
                lines.append(f"[{color}]{pip}[/] [{TEXT}]{provider}[/]  {suffix}")
            try:
                self.query_one("#sources-list", Static).update("\n".join(lines))
            except Exception:  # pragma: no cover
                pass

        def _refresh_index(self) -> None:
            try:
                bar = self.query_one("#index-bar", ProgressBar)
                info = self.query_one("#index-info", Static)
            except Exception:  # pragma: no cover - widget may be gone during shutdown
                return
            if not self.controller.enabled_sources():
                bar.update(total=100, progress=0)
                info.update(f"[{TEXT_MUTED}]enable a source (/enable)[/]")
                return
            try:
                status = self.controller.context_builder.indexing_status()
            except Exception:
                info.update(f"[{TEXT_MUTED}]unknown[/]")
                return
            if status.get("active"):
                total = float(status.get("total_sessions") or 0)
                done = float(status.get("processed_sessions") or 0)
                bar.update(total=max(total, 1.0), progress=done)
                eta = status.get("eta_seconds")
                eta_text = f"  ·  eta {eta}s" if eta is not None else ""
                info.update(f"[{TEAL}]{int(done)}/{int(total)}[/][{TEXT_MUTED}]{eta_text}[/]")
            elif status.get("completed"):
                total = int(status.get("total_sessions") or 0)
                bar.update(total=1, progress=1)
                info.update(f"[{TEAL}]ready[/]  [{TEXT_MUTED}]· {total} sessions[/]")
            else:
                bar.update(total=100, progress=0)
                message = _rich_escape(str(status.get("message") or "idle"))
                info.update(f"[{TEXT_MUTED}]{message}[/]")


def run_tui(config: RunConfig) -> int:
    controller = DevenvTUIController(config)
    if TEXTUAL_AVAILABLE:
        try:
            app = DevenvTextualApp(controller)
            app.run()
            controller.close()
            return 0
        except KeyboardInterrupt:
            controller.close()
            return 0
    render_banner(config)

    while True:
        try:
            prompt = input(_style("devenv", Ansi.BOLD, Ansi.CYAN) + _style(" › ", Ansi.DIM)).strip()
        except EOFError:
            print()
            controller.close()
            return 0
        except KeyboardInterrupt:
            print()
            controller.close()
            return 0

        command_result = controller.handle_input(prompt)
        if command_result is not None:
            print(command_result.message)
            print()
            if command_result.should_exit:
                controller.close()
                return 0
            continue

        if controller.mode == "retrieve":
            print(_style("retrieving prior sessions…", Ansi.DIM))
            result = controller.run_retrieval(prompt)
            print(_retrieval_plain_text(result))
            print()
            continue

        print(_style("solve mode is still in progress.", Ansi.YELLOW))
        print(_style("Use /mode retrieve to return to retrieval mode.", Ansi.DIM))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Devenv runtime TUI.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument(
        "--max-consecutive-tools",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_TOOLS,
    )
    parser.add_argument("--performance-mode", default="medium", choices=("low", "medium", "high"))
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
    return run_tui(config)


if __name__ == "__main__":
    raise SystemExit(main())
