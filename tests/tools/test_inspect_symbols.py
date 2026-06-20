from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.inspect_symbols import InspectSymbolsTool


FIXTURE_FILE = (
    Path(__file__).resolve().parents[2]
    / "sample-test"
    / "tool-fixtures"
    / "app"
    / "services"
    / "calendar_service.py"
)


class InspectSymbolsToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = InspectSymbolsTool()

    def test_outline_mode_lists_class_and_methods(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="outline")

        self.assertTrue(result.success)
        self.assertEqual(result.data["symbols"][0]["name"], "CalendarService")
        self.assertIn("describe_backend", result.data["symbols"][0]["methods"])

    def test_signatures_mode_extracts_annotations(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="signatures")

        self.assertTrue(result.success)
        self.assertEqual(result.data["signatures"][0]["name"], "describe_backend")
        self.assertEqual(result.data["signatures"][0]["returns"], "str")

    def test_documentation_mode_returns_docstrings(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_FILE), mode="documentation")

        self.assertTrue(result.success)
        self.assertEqual(result.data["documentation"][0]["scope"], "module")
        self.assertIn("Business logic", result.data["documentation"][0]["docstring"])
