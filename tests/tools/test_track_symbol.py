from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.track_symbol import TrackSymbolTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"


class TrackSymbolToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TrackSymbolTool()

    def test_references_mode_finds_symbol_mentions(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), symbol="CalendarService", mode="references")

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.data["count"], 2)

    def test_definitions_mode_finds_class_definition(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), symbol="CalendarService", mode="definitions")

        self.assertTrue(result.success)
        class_matches = [match for match in result.data["matches"] if match["kind"] == "class"]
        self.assertEqual(class_matches[0]["relative_path"], "app/services/calendar_service.py")
