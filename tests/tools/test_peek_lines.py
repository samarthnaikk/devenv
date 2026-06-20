from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.peek_lines import PeekLinesTool


FIXTURE_FILE = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures" / "README.md"


class PeekLinesToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = PeekLinesTool()

    def test_range_mode_returns_selected_lines(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="range", start=1, end=3)

        self.assertTrue(result.success)
        self.assertEqual(result.data["line_start"], 1)
        self.assertEqual(result.data["line_end"], 3)
        self.assertIn("Tool Fixture Workspace", result.data["content"])

    def test_head_mode_uses_end_as_count(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="head", end=2)

        self.assertTrue(result.success)
        self.assertEqual(result.data["line_end"], 2)
        self.assertEqual(len(result.data["lines"]), 2)

    def test_tail_mode_returns_last_rows(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="tail", end=3)

        self.assertTrue(result.success)
        self.assertEqual(len(result.data["lines"]), 3)
        self.assertIn("project planning notes", result.data["content"])

    def test_range_mode_requires_bounds(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="range")

        self.assertFalse(result.success)
        self.assertIn("start and end", result.output)
