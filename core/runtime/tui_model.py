"""Textual screen for selecting an OpenCode model.

Imported lazily by the TUI so the runtime keeps working when Textual is
unavailable. The screen lists every model discovered from OpenCode with a
search filter, optional cost metadata, and a manual-entry escape hatch.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from core.ai.model_catalog import OpenCodeModelInfo


def format_cost(model: OpenCodeModelInfo) -> str:
    if model.cost_input is None and model.cost_output is None:
        return ""
    cost_in = "" if model.cost_input is None else _format_price(model.cost_input)
    cost_out = "" if model.cost_output is None else _format_price(model.cost_output)
    return f"${cost_in}/${cost_out} per 1M"


def _format_price(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def format_model_label(model: OpenCodeModelInfo) -> str:
    parts = [model.full_id]
    if model.name and model.name.lower() != model.model_id.lower():
        parts.append(model.name)
    cost = format_cost(model)
    if cost:
        parts.append(cost)
    return "  ·  ".join(parts)


def filter_models(
    models: Sequence[OpenCodeModelInfo],
    query: str,
) -> list[OpenCodeModelInfo]:
    tokens = [token for token in query.strip().lower().split() if token]
    if not tokens:
        return list(models)
    matched: list[OpenCodeModelInfo] = []
    for model in models:
        haystack = " ".join(
            [model.full_id, model.name, model.family, model.provider_id, model.model_id]
        ).lower()
        if all(token in haystack for token in tokens):
            matched.append(model)
    return matched


class ModelPickerScreen(ModalScreen[str | None]):
    """Dropdown of OpenCode models; dismisses with the chosen model id."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }

    #model-picker-box {
        width: 80;
        height: auto;
        max-height: 80%;
        background: #16191e;
        border: round #2d333b;
        padding: 1 2;
    }

    #model-picker-title {
        text-style: bold;
        color: #4fdbc8;
        margin-bottom: 1;
    }

    #model-picker-hint {
        color: #8b949e;
        margin-bottom: 1;
    }

    #model-picker-list {
        height: auto;
        max-height: 24;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem.--highlight {
        background: #4fdbc8;
        color: #003731;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        models: Sequence[OpenCodeModelInfo],
        *,
        current: str = "",
        title: str = "Select a model",
    ) -> None:
        super().__init__()
        self._models = list(models)
        self._current = current.strip()
        self._title = title
        self._visible: list[OpenCodeModelInfo] = list(self._models)
        self._custom_mode = False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-box"):
            yield Static(self._title, id="model-picker-title")
            yield Static("", id="model-picker-hint")
            yield Input(placeholder="Filter models…", id="model-picker-filter")
            yield ListView(id="model-picker-list")

    def on_mount(self) -> None:
        self._refresh_hint()
        self._visible = filter_models(self._models, "")
        self.run_worker(self._rebuild_list(), group="model-filter", exclusive=True)
        self.query_one("#model-picker-filter", Input).focus()

    def _refresh_hint(self) -> None:
        hint = self.query_one("#model-picker-hint", Static)
        total = len(self._models)
        if self._current:
            hint.update(f"{total} models · current: {self._current}")
        else:
            hint.update(f"{total} models")

    async def _rebuild_list(self) -> None:
        list_view = self.query_one("#model-picker-list", ListView)
        await list_view.remove_children()
        items: list[ListItem] = []
        for model in self._visible:
            marker = "● " if model.full_id == self._current else "  "
            items.append(ListItem(Label(f"{marker}{format_model_label(model)}")))
        if not self._visible:
            items.append(ListItem(Label("  (no models match — type a custom id below)")))
        items.append(ListItem(Label("  Enter custom model…"), id="model-picker-custom"))
        await list_view.mount(*items)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._custom_mode:
            return
        self._visible = filter_models(self._models, event.value)
        self.run_worker(self._rebuild_list(), group="model-filter", exclusive=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._custom_mode:
            value = event.value.strip()
            if value:
                self.dismiss(value)
            return
        if len(self._visible) == 1:
            self.dismiss(self._visible[0].full_id)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.index < len(self._visible):
            self.dismiss(self._visible[event.index].full_id)
            return
        self._enter_custom_mode()

    def _enter_custom_mode(self) -> None:
        self._custom_mode = True
        filter_input = self.query_one("#model-picker-filter", Input)
        filter_input.value = ""
        filter_input.placeholder = "Type a model id (provider/model) and press Enter"
        filter_input.focus()
        self.query_one("#model-picker-hint", Static).update(
            "Manual entry · ESC to cancel"
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
