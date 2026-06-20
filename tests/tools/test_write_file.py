from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.tools.write_file import WriteFileTool


SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class WriteFileToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = WriteFileTool()
        self.tempdir = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_fresh_mode_creates_new_file(self) -> None:
        target = self.root / "new_note.txt"
        result = self.tool.execute(path=str(target), content="hello", mode="fresh")

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")

    def test_overwrite_mode_replaces_existing_content(self) -> None:
        target = self.root / "overwrite.txt"
        target.write_text("before", encoding="utf-8")
        result = self.tool.execute(path=str(target), content="after", mode="overwrite")

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "after")

    def test_append_mode_extends_existing_file(self) -> None:
        target = self.root / "append.txt"
        target.write_text("line1\n", encoding="utf-8")
        result = self.tool.execute(path=str(target), content="line2\n", mode="append")

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "line1\nline2\n")
