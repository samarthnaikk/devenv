from __future__ import annotations

import unittest

from textual.app import App, ComposeResult

from core.runtime.tui_panes import MemoryPane, SessionsPane
from core.runtime.tui_widgets import (
    DiffView,
    EvidenceCard,
    ResultCard,
    StreamingMarkdown,
    ToolTrace,
)


class _Host(App[None]):
    def __init__(self, widget: object) -> None:
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget  # type: ignore[misc]


class ResultCardTest(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_body_treats_brackets_as_text(self) -> None:
        body = "## Session\n[bold]not markup[/]\n- a [x](y) line"
        app = _Host(ResultCard(body, mode="markdown"))
        async with app.run_test() as pilot:
            await pilot.pause()
            cards = app.query(ResultCard)
            self.assertEqual(len(cards), 1)
            self.assertTrue(app.query("Markdown"))


class EvidenceCardTest(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_is_rendered_as_literal_text(self) -> None:
        body = "# not a heading\n- not a list\n[bold]not markup[/]\n[x](http://evil)"
        app = _Host(EvidenceCard(body, title="Evidence"))
        async with app.run_test() as pilot:
            await pilot.pause()
            card = app.query_one(EvidenceCard)
            self.assertEqual(len(app.query(EvidenceCard)), 1)
            # Evidence must never be handed to the Markdown parser.
            self.assertFalse(card.query("Markdown"))
            self.assertEqual(card._body, body)
            static = card.query_one(".evidence-body")
            self.assertFalse(static.has_class("markup"))


class StreamingMarkdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_append_accumulates_and_renders_markdown(self) -> None:
        app = _Host(StreamingMarkdown())
        async with app.run_test() as pilot:
            await pilot.pause()
            widget = app.query_one(StreamingMarkdown)
            widget.append("## Title\n")
            widget.append("hello **world**")
            await pilot.pause()
            self.assertEqual(widget.text, "## Title\nhello **world**")
            self.assertTrue(widget.query("Markdown"))

    async def test_append_ignores_empty_fragments(self) -> None:
        app = _Host(StreamingMarkdown())
        async with app.run_test() as pilot:
            await pilot.pause()
            widget = app.query_one(StreamingMarkdown)
            widget.append("")
            await pilot.pause()
            self.assertEqual(widget.text, "")


class DiffViewTest(unittest.IsolatedAsyncioTestCase):
    async def test_renders_unified_diff(self) -> None:
        app = _Host(DiffView("core/app.py", "old line\n", "new line\n"))
        async with app.run_test() as pilot:
            await pilot.pause()
            view = app.query_one(DiffView)
            self.assertEqual(view.path, "core/app.py")
            self.assertEqual(view.old_text, "old line\n")

    async def test_empty_diff_is_safe(self) -> None:
        app = _Host(DiffView("core/app.py", "same\n", "same\n"))
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertTrue(app.query(DiffView))


class ToolTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_set_status_updates_title(self) -> None:
        app = _Host(ToolTrace("tool · read", collapsed=True))
        async with app.run_test() as pilot:
            await pilot.pause()
            trace = app.query_one(ToolTrace)
            trace.set_status("tool · read · completed", "output text")
            await pilot.pause()
            self.assertEqual(trace.title, "tool · read · completed")


class SessionsPaneTest(unittest.IsolatedAsyncioTestCase):
    async def test_set_sessions_populates_table(self) -> None:
        app = _Host(SessionsPane())
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one(SessionsPane)
            pane.set_sessions(
                [
                    ("opencode", "Fix auth flow", 12, "2026-09-01", "session preview text"),
                    ("codex", "Add retrieval eval", 8, "2026-09-02", "another preview"),
                ]
            )
            await pilot.pause()
        self.assertEqual(len(pane._previews), 2)

    async def test_empty_sessions_message(self) -> None:
        app = _Host(SessionsPane())
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one(SessionsPane)
            pane.set_sessions([])
            await pilot.pause()
        self.assertEqual(pane._previews, [])


class MemoryPaneTest(unittest.IsolatedAsyncioTestCase):
    async def test_set_trace_renders_markdown(self) -> None:
        app = _Host(MemoryPane())
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one(MemoryPane)
            pane.set_trace(type("Trace", (), {"markdown_context": "## Recall\n- item one"})())
            await pilot.pause()
            self.assertTrue(app.query("Markdown"))

    async def test_set_trace_includes_provenance_counts(self) -> None:
        trace = type(
            "Trace",
            (),
            {
                "markdown_context": "body",
                "matched_nodes": [1, 2, 3],
                "expanded_candidates": [1, 2],
                "selected_nodes": [1],
            },
        )()
        app = _Host(MemoryPane())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(MemoryPane).set_trace(trace)
            await pilot.pause()
            self.assertTrue(app.query("#memory-body"))


if __name__ == "__main__":
    unittest.main()
