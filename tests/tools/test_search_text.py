from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.search_text import SearchTextTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"


class SearchTextToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = SearchTextTool()

    def test_literal_mode_finds_matching_lines(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), query="calendar", mode="literal")

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.data["count"], 2)

    def test_regex_mode_finds_python_class(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), query=r"class\s+CalendarService", mode="regex", ext_filter=".py")

        self.assertTrue(result.success)
        self.assertEqual(result.data["matches"][0]["relative_path"], "app/services/calendar_service.py")

    def test_semantic_mode_returns_ranked_files(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), query="job scheduling backend", mode="semantic")

        self.assertTrue(result.success)
        self.assertTrue(result.data["matches"])
        self.assertIn("relative_path", result.data["matches"][0])
