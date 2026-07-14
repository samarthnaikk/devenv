from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.runtime.models import RunConfig
from core.runtime.setup import _check_codex_backend, _check_ollama_backend, inspect_setup


class SetupInspectionTest(unittest.TestCase):
    @patch("core.runtime.setup._find_missing_dependencies", return_value=[])
    @patch("core.runtime.setup._check_opencode", return_value=(True, "OpenCode CLI available: 1.0.0."))
    @patch("core.runtime.setup._check_opencode_server", return_value=("pending", "OpenCode server is unavailable."))
    def test_apply_changes_initializes_workspace_state(self, _mock_server, _mock_opencode, _mock_deps) -> None:
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
    @patch("core.runtime.setup._check_opencode_server", return_value=("failed", "OpenCode server is unavailable."))
    def test_inspection_reports_missing_required_dependencies(self, _mock_server, _mock_opencode, _mock_deps) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = inspect_setup(RunConfig(workspace_path=tempdir), apply_changes=False, include_optional=False)

        self.assertFalse(result.ready)
        self.assertEqual(result.required_checks[1].status, "failed")
        self.assertIn("sentence_transformers", result.required_checks[1].detail)
        self.assertEqual(result.required_checks[2].status, "failed")

    @patch("core.runtime.setup._find_missing_dependencies", return_value=[])
    @patch("core.runtime.setup._check_opencode", return_value=(True, "OpenCode CLI available: 1.0.0."))
    @patch("core.runtime.setup._check_opencode_server", return_value=("ready", "OpenCode server reachable at http://127.0.0.1:4096 (1.0.0)."))
    @patch("core.runtime.setup._check_ollama_backend", return_value=("ready", "Ollama reachable at http://127.0.0.1:11434 with models: qwen2.5:3b."))
    @patch("core.runtime.setup._check_codex_backend", return_value=("ready", "Codex backend configured for model gpt-5-codex."))
    @patch("core.runtime.setup._check_sentence_transformer_cache", return_value=("ready", "cache ready"))
    @patch("core.runtime.setup._check_web_search_prerequisites", return_value=("ready", "web ready"))
    @patch("core.runtime.setup._check_latex_pdf_toolchain", return_value=("pending", "latex pending"))
    def test_optional_checks_report_capability_statuses(
        self,
        _mock_latex,
        _mock_web,
        _mock_cache,
        _mock_codex,
        _mock_ollama,
        _mock_server,
        _mock_opencode,
        _mock_deps,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = inspect_setup(RunConfig(workspace_path=tempdir), apply_changes=True, include_optional=True)

        self.assertEqual(result.optional_checks[0].name, "opencode_server")
        self.assertEqual(result.optional_checks[0].status, "ready")
        self.assertEqual(result.optional_checks[2].name, "ollama_backend")
        self.assertEqual(result.optional_checks[2].status, "ready")
        self.assertEqual(result.optional_checks[3].name, "codex_backend")
        self.assertEqual(result.optional_checks[3].status, "ready")
        self.assertEqual(result.optional_checks[4].detail, "cache ready")
        self.assertEqual(result.optional_checks[5].detail, "web ready")
        self.assertEqual(result.optional_checks[6].status, "pending")

    @patch("core.runtime.setup.importlib.util.find_spec", return_value=None)
    def test_codex_check_distinguishes_missing_sdk_after_credentials(self, _mock_find_spec) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "DEVENV_CODEX_MODEL": "gpt-5-codex",
            },
            clear=False,
        ):
            status, detail = _check_codex_backend()

        self.assertEqual(status, "failed")
        self.assertIn("agents sdk", detail.lower())

    @patch("core.runtime.setup.urllib.request.urlopen", side_effect=OSError("connection refused"))
    def test_ollama_check_reports_pending_when_backend_is_not_running(self, _mock_urlopen) -> None:
        status, detail = _check_ollama_backend()

        self.assertEqual(status, "pending")
        self.assertIn("Ollama is not running", detail)


if __name__ == "__main__":
    unittest.main()
