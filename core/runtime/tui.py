from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.logging_utils import configure_logging

from .context_builder import ContextBuilderService
from .kernel import DevenvKernel
from .models import DEFAULT_MAX_CONSECUTIVE_TOOLS, RunConfig, RuntimeTurnResult
from .tooling import build_runtime_tools
from .web import AccessPolicy, DEFAULT_LLAMACPP_MODELS, DEFAULT_OLLAMA_MODELS, DEFAULT_WEB_MODELS

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Container, Vertical
    from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
    from textual.widgets.option_list import Option

    TEXTUAL_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path for environments without textual
    work = None
    App = object
    ComposeResult = object
    Container = object
    Vertical = object
    Footer = object
    Header = object
    Input = object
    OptionList = object
    RichLog = object
    Static = object
    Option = object
    TEXTUAL_AVAILABLE = False

BACKENDS = ("opencode", "ollama", "llama_cpp", "codex")
SESSION_PROVIDERS = ("codex", "opencode")


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


def _style(text: str, *codes: str) -> str:
    return "".join(codes) + text + Ansi.RESET


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
        self.preferred_backend = getattr(self.kernel.ai, "preferred_backend", "opencode") or "opencode"
        self._apply_runtime_preferences()

    def close(self) -> None:
        self.kernel.close()

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
                _style("Command Palette", Ansi.BOLD, Ansi.CYAN),
                "/status                 Show active backend, model, and permissions",
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
            return f"Backend `{name}` permission is now {value}."
        if target_type == "provider":
            if name not in SESSION_PROVIDERS:
                return "Providers must be one of: codex, opencode."
            self.access_policy.set_session_access(name, allowed)
            self._apply_runtime_preferences()
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
        f"{_style('memory', Ansi.GREEN)} online   "
        f"{_style('performance', Ansi.BLUE)} {config.performance_mode}   "
        f"{_style('commands', Ansi.MAGENTA)} /help /status /permissions"
    )
    print(line)


def render_turn_result(result: RuntimeTurnResult) -> None:
    for trace in result.stage_traces:
        checkpoint_label = f" checkpoint={trace.checkpoint_id}" if trace.checkpoint_id is not None else ""
        status = _style("ok", Ansi.GREEN) if trace.success else _style("failed", Ansi.RED)
        print(f"{_style('stage', Ansi.DIM)} {trace.stage}{checkpoint_label} -> {status}. {trace.summary}")
    for step in result.steps:
        if step.is_sandboxed_violation:
            print(f"{_style('sandbox', Ansi.YELLOW)} {step.output}")
            continue
        status = _style("success", Ansi.GREEN) if step.success else _style("failure", Ansi.RED)
        print(f"{_style('tool', Ansi.DIM)} {step.tool_name} -> {status}")

    if result.final_response:
        print()
        print(_style("assistant", Ansi.BOLD, Ansi.CYAN))
        print(result.final_response)


if TEXTUAL_AVAILABLE:
    class DevenvTextualApp(App[None]):
        CSS = """
        Screen {
            layout: vertical;
            background: #0b1220;
            color: #f7f4ea;
        }

        #shell {
            height: 1fr;
        }

        #log {
            height: 1fr;
            border: round #2f6fed;
            background: #11192b;
            padding: 0 1;
        }

        #composer {
            margin: 1 0 0 0;
            border: round #f08a24;
            background: #0f1727;
        }

        #hint {
            color: #8ca3c7;
            margin: 1 0 0 0;
        }

        #palette {
            layer: overlay;
            align: center middle;
            width: 84;
            height: 28;
            display: none;
        }

        #palette.visible {
            display: block;
        }

        #palette-panel {
            border: round #f08a24;
            background: #101826;
            padding: 1 2;
        }

        #palette-title {
            color: #f7f4ea;
            text-style: bold;
            margin: 0 0 1 0;
        }

        #palette-query {
            margin: 0 0 1 0;
            border: round #2f6fed;
            background: #0f1727;
        }

        #palette-options {
            height: 1fr;
            border: round #24324c;
            background: #0c1320;
        }
        """

        BINDINGS = [
            ("/", "open_palette", "Command Palette"),
            ("escape", "close_palette", "Close Palette"),
        ]

        def __init__(self, controller: DevenvTUIController) -> None:
            super().__init__()
            self.controller = controller
            self._palette_entries: list[PaletteEntry] = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Container(id="shell"):
                yield RichLog(id="log", markup=True, wrap=True)
                yield Static("Type a prompt to chat. Type `/` for the command palette.", id="hint")
                yield Input(placeholder="Ask devenv anything…", id="composer")
            with Container(id="palette"):
                with Vertical(id="palette-panel"):
                    yield Static("Command Palette", id="palette-title")
                    yield Input(placeholder="Search commands, permissions, backends, models…", id="palette-query")
                    yield OptionList(id="palette-options")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "DEVENV CORE TUI"
            self.sub_title = self.controller.config.workspace_path
            self._write_shell_line(f"[bold cyan]Workspace[/] {self.controller.config.workspace_path}")
            self._write_shell_line("[dim]Use / to search commands, toggle permissions, switch backends, and set models.[/]")
            self.query_one("#composer", Input).focus()

        def action_open_palette(self) -> None:
            palette = self.query_one("#palette", Container)
            palette.add_class("visible")
            query = self.query_one("#palette-query", Input)
            query.value = ""
            self._refresh_palette()
            query.focus()

        def action_close_palette(self) -> None:
            palette = self.query_one("#palette", Container)
            palette.remove_class("visible")
            self.query_one("#composer", Input).focus()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "palette-query":
                self._refresh_palette(event.value)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "composer":
                value = event.value.strip()
                event.input.value = ""
                if not value:
                    return
                if value.startswith("/"):
                    self.action_open_palette()
                    query = self.query_one("#palette-query", Input)
                    query.value = value[1:].strip()
                    self._refresh_palette(query.value)
                    return
                self._write_shell_line(f"[bold cyan]You[/] {value}")
                self._write_shell_line("[dim]Retrieving memory context…[/]")
                self._write_shell_line("[dim]Reasoning…[/]")
                self._run_prompt(value)
                return
            if event.input.id == "palette-query":
                self._activate_highlighted_palette_entry()

        def on_key(self, event) -> None:
            palette = self.query_one("#palette", Container)
            if not palette.has_class("visible"):
                return
            options = self.query_one("#palette-options", OptionList)
            if event.key == "down":
                options.action_cursor_down()
                event.prevent_default()
            elif event.key == "up":
                options.action_cursor_up()
                event.prevent_default()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            if event.option_list.id != "palette-options":
                return
            self._execute_palette_selection(event.option.id or "")

        def _refresh_palette(self, query: str = "") -> None:
            options = self.query_one("#palette-options", OptionList)
            self._palette_entries = self.controller.palette_entries(query)
            options.clear_options()
            rendered = [
                Option(f"{entry.label}\n[dim]{entry.command}[/]", id=entry.entry_id)
                for entry in self._palette_entries
            ]
            options.add_options(rendered)
            if rendered:
                options.highlighted = 0

        def _activate_highlighted_palette_entry(self) -> None:
            options = self.query_one("#palette-options", OptionList)
            if options.option_count <= 0:
                return
            options.action_select()

        def _execute_palette_selection(self, entry_id: str) -> None:
            selected = next((entry for entry in self._palette_entries if entry.entry_id == entry_id), None)
            if selected is None:
                return
            result = self.controller.handle_command(selected.command)
            self.action_close_palette()
            self._write_shell_line(f"[bold magenta]Command[/] {selected.label}")
            self._write_shell_line(result.message)
            if result.should_exit:
                self.exit()

        def _write_shell_line(self, message: str) -> None:
            self.query_one("#log", RichLog).write(message)

        @work(thread=True)
        def _run_prompt(self, prompt: str) -> None:
            result = self.controller.run_prompt(prompt)
            self.call_from_thread(self._render_prompt_result, result)

        def _render_prompt_result(self, result: RuntimeTurnResult) -> None:
            for trace in result.stage_traces:
                checkpoint_label = f" checkpoint={trace.checkpoint_id}" if trace.checkpoint_id is not None else ""
                state = "ok" if trace.success else "failed"
                color = "green" if trace.success else "red"
                self._write_shell_line(f"[dim]stage[/] {trace.stage}{checkpoint_label} -> [{color}]{state}[/]. {trace.summary}")
            for step in result.steps:
                if step.is_sandboxed_violation:
                    self._write_shell_line(f"[yellow]sandbox[/] {step.output}")
                else:
                    state = "success" if step.success else "failure"
                    color = "green" if step.success else "red"
                    self._write_shell_line(f"[dim]tool[/] {step.tool_name} -> [{color}]{state}[/]")
            if result.final_response:
                self._write_shell_line(f"[bold cyan]Assistant[/] {result.final_response}")
            self.query_one("#composer", Input).focus()


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

        print(_style("retrieving memory context…", Ansi.DIM))
        print(_style("reasoning…", Ansi.DIM))
        result = controller.run_prompt(prompt)
        render_turn_result(result)
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
