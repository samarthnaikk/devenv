from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from core.tools.audit_changes import AuditChangesTool


SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class AuditChangesToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = AuditChangesTool()
        self.tempdir = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        subprocess.run(["git", "init", str(self.root)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "tool@test.local"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Tool Test"], check=True, capture_output=True, text=True)
        (self.root / "tracked.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "tracked.txt"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-m", "init"], check=True, capture_output=True, text=True)
        (self.root / "tracked.txt").write_text("hello\nworld\n", encoding="utf-8")

    def test_status_mode_reports_changed_file(self) -> None:
        result = self.tool.execute(mode="status", path=str(self.root))

        self.assertTrue(result.success)
        self.assertEqual(result.data["entries"][0]["path"], "tracked.txt")

    def test_diff_mode_returns_stat_output(self) -> None:
        result = self.tool.execute(mode="diff", path=str(self.root))

        self.assertTrue(result.success)
        self.assertIn("tracked.txt", result.data["stat"])
