from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.runtime.models import RunConfig
from core.runtime.setup import inspect_setup


class SetupInspectionTest(unittest.TestCase):
    @patch("core.runtime.setup._find_missing_dependencies", return_value=[])
    @patch("core.runtime.setup._check_opencode", return_value=(True, "OpenCode CLI available: 1.0.0."))
    def test_apply_changes_initializes_workspace_state(self, _mock_opencode, _mock_deps) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = RunConfig(workspace_path=tempdir)

            result = inspect_setup(config, apply_changes=True, include_optional=False)

            self.assertTrue(result.ready)
            self.assertTrue((Path(tempdir) / "memory.db").is_file())
            self.assertTrue((Path(tempdir) / "vectors").is_dir())
            self.assertEqual(result.required_checks[-1].name, "workspace_state")
            self.assertEqual(result.required_checks[-1].status, "ready")

    @patch("core.runtime.setup._find_missing_dependencies", return_value=["sentence_transformers"])
    @patch("core.runtime.setup._check_opencode", return_value=(False, "OpenCode CLI was not found on PATH."))
    def test_inspection_reports_missing_required_dependencies(self, _mock_opencode, _mock_deps) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = inspect_setup(RunConfig(workspace_path=tempdir), apply_changes=False, include_optional=False)

        self.assertFalse(result.ready)
        self.assertEqual(result.required_checks[1].status, "failed")
        self.assertIn("sentence_transformers", result.required_checks[1].detail)
        self.assertEqual(result.required_checks[2].status, "failed")

    @patch("core.runtime.setup._find_missing_dependencies", return_value=[])
    @patch("core.runtime.setup._check_opencode", return_value=(True, "OpenCode CLI available: 1.0.0."))
    @patch("core.runtime.setup._check_sentence_transformer_cache", return_value=("ready", "cache ready"))
    @patch("core.runtime.setup._check_web_search_prerequisites", return_value=("ready", "web ready"))
    @patch("core.runtime.setup._check_latex_pdf_toolchain", return_value=("pending", "latex pending"))
    def test_optional_checks_report_capability_statuses(
        self,
        _mock_latex,
        _mock_web,
        _mock_cache,
        _mock_opencode,
        _mock_deps,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = inspect_setup(RunConfig(workspace_path=tempdir), apply_changes=True, include_optional=True)

        self.assertEqual(result.optional_checks[0].status, "ready")
        self.assertEqual(result.optional_checks[1].detail, "web ready")
        self.assertEqual(result.optional_checks[2].status, "pending")


if __name__ == "__main__":
    unittest.main()
