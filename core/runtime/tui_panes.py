"""Secondary workspace panes for the Devenv TUI.

These panes are read-only views over data the runtime already exposes: indexed
external sessions from the context builder and the last retrieval trace from the
memory engine. They contain no retrieval or model logic.
"""

from __future__ import annotations

from typing import Any, Sequence

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Markdown, Static


class SessionsPane(Vertical):
    """Browsable table of indexed Codex/OpenCode sessions."""

    DEFAULT_CSS = """
    SessionsPane {
        height: 1fr;
        background: $background;
        padding: 0 1;
    }

    SessionsPane .pane-hint {
        color: $text-muted;
        height: auto;
    }

    SessionsPane DataTable {
        height: 1fr;
        background: $background;
    }

    SessionsPane .pane-detail {
        color: $foreground;
        height: auto;
        max-height: 8;
        border-top: solid $border;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._previews: list[str] = []
        self._detail = Static(
            "Select a session row to preview it.",
            classes="pane-detail",
            markup=False,
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "Indexed sessions across enabled sources. Toggle sources with /enable.",
            classes="pane-hint",
        )
        yield DataTable(id="sessions-table", cursor_type="row", zebra_stripes=True)
        yield self._detail

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Provider", "Title", "Msgs", "Updated")

    def set_sessions(self, rows: Sequence[Sequence[Any]]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self._previews = []
        for row in rows:
            values = list(row)
            preview = str(values[4]) if len(values) > 4 and values[4] is not None else ""
            self._previews.append(preview)
            table.add_row(*[("" if value is None else str(value)) for value in values[:4]])
        if not rows:
            self._detail.update(
                "No indexed sessions yet. Enable a source with /enable (F3/F4)."
            )
        else:
            self._detail.update(f"{len(rows)} session(s).")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        index = event.cursor_row
        if 0 <= index < len(self._previews):
            preview = self._previews[index] or "(no preview)"
            self._detail.update(preview)


class MemoryPane(Vertical):
    """Renders the last retrieval trace captured by the memory engine."""

    DEFAULT_CSS = """
    MemoryPane {
        height: 1fr;
        background: $background;
        padding: 0 1;
    }

    MemoryPane .pane-hint {
        color: $text-muted;
        height: auto;
    }

    MemoryPane Markdown {
        height: 1fr;
        background: transparent;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Retrieval context and provenance for the most recent query.",
            classes="pane-hint",
        )
        yield Markdown(
            "_No retrieval yet. Ask a question in the Retrieve tab._",
            id="memory-body",
        )

    def set_trace(self, trace: Any) -> None:
        markdown = self.query_one("#memory-body", Markdown)
        provenance: list[str] = []
        for label, attr in (
            ("Matched nodes", "matched_nodes"),
            ("Expanded candidates", "expanded_candidates"),
            ("Selected nodes", "selected_nodes"),
        ):
            value = getattr(trace, attr, None)
            if value is None:
                continue
            try:
                provenance.append(f"- {label}: {len(value)}")
            except TypeError:
                continue
        body = str(getattr(trace, "markdown_context", "") or "")
        sections: list[str] = []
        if provenance:
            sections.append("## Retrieval provenance\n" + "\n".join(provenance))
        if body.strip():
            sections.append(body)
        if not sections:
            markdown.update("_No retrieval context captured yet._")
            return
        markdown.update("\n\n".join(sections))


__all__ = ["SessionsPane", "MemoryPane"]
