"""Visual regression snapshots for the TUI, run with pytest.

These tests use ``pytest-textual-snapshot`` and are intentionally separate from
the unittest suite. Update baselines with ``pytest --snapshot-update``.
"""

from __future__ import annotations

from textual.app import App

from core.runtime.tui_widgets import Choice, SelectionScreen


class _SelectionHarness(App[None]):
    def on_mount(self) -> None:
        self.push_screen(
            SelectionScreen(
                [
                    Choice("opencode/claude-sonnet-4", "Claude Sonnet 4", detail="$3/$15 per 1M"),
                    Choice("opencode/gpt-5-codex", "GPT-5 Codex", detail="$1/$4 per 1M"),
                    Choice("ollama/qwen2.5:3b", "Qwen 2.5 3B", detail="local"),
                ],
                title="Select the answer model",
                current="opencode/gpt-5-codex",
            )
        )


def test_selection_screen_snapshot(snap_compare) -> None:
    assert snap_compare(_SelectionHarness())
