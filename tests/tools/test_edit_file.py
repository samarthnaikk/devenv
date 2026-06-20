from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.tools.edit_file import EditFileTool


SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class EditFileToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = EditFileTool()
        self.tempdir = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.addCleanup(self.tempdir.cleanup)
        self.target = Path(self.tempdir.name) / "editable.txt"
        self.target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    def test_patch_mode_replaces_matching_block(self) -> None:
        result = self.tool.execute(
            path=str(self.target),
            mode="patch",
            search_block="beta",
            replace_block="BETA",
        )

        self.assertTrue(result.success)
        self.assertIn("BETA", self.target.read_text(encoding="utf-8"))

    def test_undo_mode_restores_previous_contents(self) -> None:
        self.tool.execute(path=str(self.target), mode="patch", search_block="beta", replace_block="BETA")
        result = self.tool.execute(path=str(self.target), mode="undo")

        self.assertTrue(result.success)
        self.assertEqual(self.target.read_text(encoding="utf-8"), "alpha\nbeta\ngamma\n")

    def test_patch_mode_rejects_missing_block(self) -> None:
        result = self.tool.execute(
            path=str(self.target),
            mode="patch",
            search_block="missing",
            replace_block="noop",
        )

        self.assertFalse(result.success)
        self.assertIn("search_block not found", result.output)
