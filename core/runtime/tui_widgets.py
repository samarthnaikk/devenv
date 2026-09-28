"""Reusable Textual widgets for the Devenv TUI.

This module holds presentational widgets that are shared by the runtime app and
the modal screens. It must stay free of retrieval/AI logic: widgets only render
state that the controller already exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterable, Sequence, TypeVar

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from .tui_theme import BLUE, PANEL, TEAL, TEXT, TEXT_MUTED, WARN

try:  # pragma: no cover - rich ships with textual
    from rich.markup import escape as _escape
except Exception:  # pragma: no cover

    def _escape(text: str) -> str:
        return str(text)

try:  # rapidfuzz is a declared dependency, but stay defensive
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover

    _fuzz = None


__all__ = ["StatusBar", "HelpOverlay", "Choice", "SelectionScreen"]

T = TypeVar("T")


@dataclass(frozen=True)
class Choice(Generic[T]):
    """A single selectable option for :class:`SelectionScreen`."""

    value: T
    label: str
    detail: str = ""
    disabled: bool = False

    def haystack(self) -> str:
        return f"{self.label} {self.detail} {self.value}".lower()


class _SelectionInput(Input):
    """Filter input that forwards 1-9 to the picker when it is empty."""

    async def _on_key(self, event: Any) -> None:  # type: ignore[override]
        key = event.key
        if not self.value and len(key) == 1 and key in "123456789":
            screen = self.screen
            if isinstance(screen, SelectionScreen) and screen.activate_digit(int(key)):
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)


class SelectionScreen(ModalScreen[Any]):
    """One reusable, searchable picker used by every option-selecting flow.

    Replaces the previous per-domain modal screens (model, agent, permission)
    with a single widget so the interaction model stays consistent.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("up", "cursor_up", ""),
        ("down", "cursor_down", ""),
        ("pageup", "page_up", ""),
        ("pagedown", "page_down", ""),
        ("ctrl+n", "cursor_down", ""),
        ("ctrl+p", "cursor_up", ""),
    ]

    DEFAULT_CSS = f"""
    SelectionScreen {{
        align: center middle;
    }}

    #selection-box {{
        width: 84;
        height: auto;
        max-height: 85%;
        background: #16191e;
        border: round #2d333b;
        padding: 1 2;
    }}

    #selection-title {{
        text-style: bold;
        color: {TEAL};
        margin-bottom: 1;
    }}

    #selection-hint {{
        color: {TEXT_MUTED};
        margin-bottom: 1;
    }}

    #selection-filter {{
        background: #0a0c0e;
        border: round #2d333b;
        color: {TEXT};
    }}

    #selection-filter:focus {{
        border: round {TEAL};
    }}

    #selection-list {{
        height: auto;
        max-height: 22;
        margin-top: 1;
        background: transparent;
    }}

    #selection-list > .option-list--option {{
        padding: 0 1;
        color: {TEXT};
    }}

    #selection-list > .option-list--option-highlighted {{
        background: {TEAL};
        color: #003731;
        text-style: bold;
    }}
    """

    def __init__(
        self,
        choices: Sequence[Choice[Any]],
        *,
        title: str = "Select",
        hint: str = "",
        current: Any = None,
        allow_custom: bool = False,
        custom_placeholder: str = "Type a value and press Enter",
    ) -> None:
        super().__init__()
        self._all: list[Choice[Any]] = list(choices)
        self._title = title
        self._hint = hint
        self._current = current
        self._allow_custom = allow_custom
        self._custom_placeholder = custom_placeholder
        self._visible: list[Choice[Any]] = list(self._all)
        self._custom_mode = False

    # ------------------------------------------------------------------ compose
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="selection-box"):
            yield Static(self._title, id="selection-title")
            yield Static("", id="selection-hint")
            yield _SelectionInput(placeholder="Filter…", id="selection-filter")
            yield OptionList(id="selection-list")

    def on_mount(self) -> None:
        self._refresh_hint()
        self._rebuild()
        self.query_one("#selection-filter", Input).focus()

    def _refresh_hint(self) -> None:
        total = len(self._all)
        current = f" · current: {self._current}" if self._current else ""
        extra = f" · {self._hint}" if self._hint else ""
        self.query_one("#selection-hint", Static).update(f"{total} option(s){current}{extra}")

    # ------------------------------------------------------------------ filtering
    def _filtered(self, query: str) -> list[Choice[Any]]:
        query = query.strip()
        if not query:
            return list(self._all)
        scored: list[tuple[float, Choice[Any]]] = []
        for choice in self._all:
            if _fuzz is not None:
                score = float(_fuzz.WRatio(query, choice.haystack()))
            else:
                score = 100.0 if query.lower() in choice.haystack() else 0.0
            if score >= 45.0:
                scored.append((score, choice))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [choice for _, choice in scored]

    def _rebuild(self) -> None:
        option_list = self.query_one("#selection-list", OptionList)
        option_list.clear_options()
        options: list[Option] = []
        for index, choice in enumerate(self._visible):
            marker = "● " if self._current is not None and str(choice.value) == str(self._current) else "  "
            accelerator = str(index + 1) if index < 9 else " "
            text = f"{accelerator} {marker}{choice.label}"
            if choice.detail:
                text += f"  ·  {choice.detail}"
            if choice.disabled:
                text += "  (unavailable)"
            options.append(Option(text, id=str(index), disabled=choice.disabled))
        if self._allow_custom:
            options.append(Option("  Enter custom value…", id="custom"))
        option_list.add_options(options)
        if options:
            option_list.highlighted = 0

    # ------------------------------------------------------------------ events
    @on(Input.Changed, "#selection-filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        if self._custom_mode:
            return
        self._visible = self._filtered(event.value)
        self._rebuild()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "selection-filter":
            return
        if self._custom_mode:
            value = event.value.strip()
            if value:
                self.dismiss(value)
            return
        self._choose_highlighted()

    @on(OptionList.OptionSelected, "#selection-list")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._select_option_id(event.option.id)

    # ------------------------------------------------------------------ actions
    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        self._move(1)

    def action_cursor_up(self) -> None:
        self._move(-1)

    def action_page_down(self) -> None:
        self._move(5)

    def action_page_up(self) -> None:
        self._move(-5)

    def activate_digit(self, digit: int) -> bool:
        """Select the visible choice at ``digit`` (1-based). Returns True if handled."""
        if self._custom_mode:
            return False
        position = digit - 1
        if 0 <= position < len(self._visible):
            choice = self._visible[position]
            if not choice.disabled:
                self.dismiss(choice.value)
                return True
        return False

    # ------------------------------------------------------------------ helpers
    def _move(self, delta: int) -> None:
        option_list = self.query_one("#selection-list", OptionList)
        count = option_list.option_count
        if count <= 0:
            return
        current = option_list.highlighted or 0
        option_list.highlighted = (current + delta) % count

    def _choose_highlighted(self) -> None:
        option_list = self.query_one("#selection-list", OptionList)
        if len(self._visible) == 1 and option_list.option_count == 1:
            choice = self._visible[0]
            if not choice.disabled:
                self.dismiss(choice.value)
            return
        index = option_list.highlighted
        if index is None:
            return
        if 0 <= index < len(self._visible):
            choice = self._visible[index]
            if not choice.disabled:
                self.dismiss(choice.value)
        elif self._allow_custom and index == len(self._visible):
            self._enter_custom()

    def _select_option_id(self, option_id: str | None) -> None:
        if option_id == "custom":
            self._enter_custom()
            return
        if option_id is None:
            return
        try:
            index = int(option_id)
        except ValueError:
            return
        if 0 <= index < len(self._visible):
            choice = self._visible[index]
            if not choice.disabled:
                self.dismiss(choice.value)

    def _enter_custom(self) -> None:
        self._custom_mode = True
        filter_input = self.query_one("#selection-filter", Input)
        filter_input.value = ""
        filter_input.placeholder = self._custom_placeholder
        filter_input.focus()
        self.query_one("#selection-hint", Static).update(
            "Manual entry · Enter to accept · Esc to cancel"
        )


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
