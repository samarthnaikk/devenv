from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.list_directory import ListDirectoryTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"


class ListDirectoryToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ListDirectoryTool()

    def test_flat_mode_lists_immediate_children(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), mode="flat")

        self.assertTrue(result.success)
        names = [entry["name"] for entry in result.data["entries"]]
        self.assertEqual(names, ["app", "data", "docs", "notes", "README.md"])

    def test_recursive_mode_honors_depth(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), mode="recursive", max_depth=2)

        self.assertTrue(result.success)
        relpaths = [entry["relative_path"] for entry in result.data["entries"]]
        self.assertIn("app/main.py", relpaths)
        self.assertIn("app/services", relpaths)
        self.assertNotIn("app/services/calendar_service.py", relpaths)

    def test_topology_mode_omits_files_and_summarizes_shape(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT), mode="topology", max_depth=3)

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "topology")
        self.assertEqual(result.data["topology"][0]["relative_path"], "app")
        self.assertTrue(all(entry["is_dir"] for entry in result.data["topology"]))

    def test_missing_mode_is_rejected(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT))

        self.assertFalse(result.success)
        self.assertIn("mode", result.output)
