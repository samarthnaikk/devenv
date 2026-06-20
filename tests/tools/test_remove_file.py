from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.tools.remove_file import RemoveFileTool


SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class RemoveFileToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = RemoveFileTool()
        self.tempdir = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.addCleanup(self.tempdir.cleanup)

    def test_soft_mode_truncates_file(self) -> None:
        target = Path(self.tempdir.name) / "soft.txt"
        target.write_text("hello", encoding="utf-8")
        result = self.tool.execute(path=str(target), mode="soft")

        self.assertTrue(result.success)
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "")

    def test_permanent_mode_deletes_file(self) -> None:
        target = Path(self.tempdir.name) / "gone.txt"
        target.write_text("bye", encoding="utf-8")
        result = self.tool.execute(path=str(target), mode="permanent")

        self.assertTrue(result.success)
        self.assertFalse(target.exists())
