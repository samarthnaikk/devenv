from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.tools.run_diagnostics import RunDiagnosticsTool


SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class RunDiagnosticsToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = RunDiagnosticsTool()
        self.tempdir = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_tests_mode_runs_unittest_discovery(self) -> None:
        (self.root / "test_demo.py").write_text(
            "import unittest\n\nclass DemoTest(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        result = self.tool.execute(mode="tests", target_path=str(self.root))

        self.assertTrue(result.success)
        self.assertEqual(result.data["tests_run"], 1)

    def test_lint_mode_compiles_python_file(self) -> None:
        target = self.root / "lint_demo.py"
        target.write_text("def ok():\n    return 1\n", encoding="utf-8")
        result = self.tool.execute(mode="lint", target_path=str(target))

        self.assertTrue(result.success)
        self.assertTrue(result.data["passed"])

    def test_types_mode_reports_missing_annotations(self) -> None:
        target = self.root / "types_demo.py"
        target.write_text("def bad(name):\n    return name\n", encoding="utf-8")
        result = self.tool.execute(mode="types", target_path=str(target))

        self.assertFalse(result.success)
        self.assertEqual(result.data["issues"][0]["function"], "bad")
