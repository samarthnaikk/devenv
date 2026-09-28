"""Reusable Textual widgets for the Devenv TUI.

This module holds presentational widgets that are shared by the runtime app and
the modal screens. It must stay free of retrieval/AI logic: widgets only render
state that the controller already exposes.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from .tui_theme import BLUE, PANEL, TEAL, TEXT, TEXT_MUTED, WARN

try:  # pragma: no cover - rich ships with textual
    from rich.markup import escape as _escape
except Exception:  # pragma: no cover

    def _escape(text: str) -> str:
        return str(text)


__all__ = ["StatusBar", "HelpOverlay"]


class StatusBar(Static):
    """One-line status bar: workspace, mode, backend, model, permissions, index."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: #16191e;
        color: #e2e2e6;
        padding: 0 1;
        text-overflow: ellipsis;
    }
    """

    def set_state(self, **state: Any) -> None:
        mode = str(state.get("mode") or "retrieve")
        mode_color = TEAL if mode == "retrieve" else WARN
        mode_label = "RETRIEVE" if mode == "retrieve" else "SOLVE (WIP)"
        pieces = [f"[b {TEAL}]DEVENV[/]"]
        workspace = state.get("workspace")
        if workspace:
            pieces.append(f"[{TEXT_MUTED}]{_escape(str(workspace))}[/]")
        pieces.append(f"[{mode_color} on {PANEL}]{mode_label}[/]")
        tab = state.get("tab")
        if tab:
            pieces.append(f"[{BLUE}]{_escape(str(tab))}[/]")
        for label, key in (("backend", "backend"), ("model", "model"), ("perm", "permission")):
            value = state.get(key)
            if value:
                pieces.append(f"[{TEXT_MUTED}]{label}[/] [{TEXT}]{_escape(str(value))}[/]")
        local_only = bool(state.get("local_only", True))
        pieces.append(f"[{TEAL}]● LOCAL[/]" if local_only else f"[{WARN}]● REMOTE[/]")
        index = state.get("index")
        if index:
            pieces.append(f"[{TEXT_MUTED}]{_escape(str(index))}[/]")
        last = state.get("last_summary")
        if last:
            pieces.append(f"[{TEXT_MUTED}]last[/] [{TEXT}]{_escape(str(last))}[/]")
        self.update("  ".join(pieces))


class HelpOverlay(ModalScreen[None]):
    """Keyboard and command reference, generated from live bindings + registry."""

    BINDINGS = [
        ("escape", "close_overlay", "Close"),
        ("question_mark", "close_overlay", "Close"),
    ]

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
    }

    #help-box {
        width: 92;
        height: auto;
        max-height: 85%;
        background: #16191e;
        border: round #2d333b;
        padding: 1 2;
    }

    #help-title {
        text-style: bold;
        color: #4fdbc8;
        margin-bottom: 1;
    }

    .help-heading {
        text-style: bold;
        color: #adc6ff;
        margin-top: 1;
    }

    .help-body {
        color: #e2e2e6;
    }

    .help-hint {
        color: #859490;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        commands: Iterable[Any],
        bindings: Sequence[tuple[str, str, str]],
    ) -> None:
        super().__init__()
        self._commands = list(commands)
        self._bindings = list(bindings)

    def compose(self) -> ComposeResult:
        from .tui_commands import group_by_category

        with VerticalScroll(id="help-box"):
            yield Static("Keyboard & Commands", id="help-title")
            yield Static("Press ? or Esc to close.", classes="help-hint")
            if self._bindings:
                yield Static("Keys", classes="help-heading")
                key_lines = "   ".join(
                    f"[b {TEAL}]{_escape(key)}[/] {_escape(label or action)}"
                    for key, action, label in self._bindings
                )
                yield Static(key_lines, classes="help-body", markup=True)
            for category, specs in group_by_category(self._commands):
                yield Static(category, classes="help-heading")
                body = "\n".join(
                    f"  [{TEAL}]{_escape(spec.command):<34}[/] {_escape(spec.title)}"
                    for spec in specs
                )
                yield Static(body, classes="help-body", markup=True)

    def action_close_overlay(self) -> None:
        self.dismiss(None)
