from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.locate_files import LocateFilesTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"


class LocateFilesToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = LocateFilesTool()

    def test_exact_mode_finds_single_file(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), pattern="config.json", mode="exact")

        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["matches"][0]["relative_path"], "data/config.json")

    def test_glob_mode_finds_python_files(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), pattern="*.py", mode="glob")

        self.assertTrue(result.success)
        relpaths = [match["relative_path"] for match in result.data["matches"]]
        self.assertIn("app/main.py", relpaths)
        self.assertIn("app/services/calendar_service.py", relpaths)

    def test_missing_pattern_is_rejected(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), mode="exact")

        self.assertFalse(result.success)
        self.assertIn("pattern", result.output)
