from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.runtime import smoke


class SmokeCliTest(unittest.TestCase):
    def test_smoke_main_parses_backend_enable_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")

            captured: dict[str, object] = {}

            def fake_execute_turn(self, prompt, **kwargs):
                captured["prompt"] = prompt
                captured.update(kwargs)

                class Result:
                    final_response = "ok"
                    steps = []
                    total_usage = {}
                    metadata = {}
                    ai_logs = []
                    system_logs = []
                    elapsed_ms = 0

                return Result()

            argv = [
                "smoke.py",
                str(workspace),
                "hello",
                "--backend-preference",
                "ollama",
                "--enable-ollama-backend",
                "--enable-codex-backend",
            ]
            stdout = io.StringIO()
            with mock.patch("sys.argv", argv), mock.patch("sys.stdout", stdout), mock.patch(
                "core.runtime.kernel.DevenvKernel.execute_turn", new=fake_execute_turn
            ):
                exit_code = smoke.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["prompt"], "hello")
            self.assertEqual(captured["backend_preference"], "ollama")
            self.assertTrue(captured["ollama_enabled"])
            self.assertTrue(captured["codex_enabled"])
            self.assertFalse(captured["opencode_enabled"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["final_response"], "ok")

    def test_smoke_main_passes_planning_continue_and_selected_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")

            captured: dict[str, object] = {}

            def fake_execute_turn(self, prompt, **kwargs):
                captured["prompt"] = prompt
                captured.update(kwargs)

                class Result:
                    final_response = "ok"
                    steps = []
                    total_usage = {}
                    metadata = {"backend_used": "local"}
                    ai_logs = []
                    system_logs = []
                    elapsed_ms = 0

                return Result()

            argv = [
                "smoke.py",
                str(workspace),
                "hello",
                "--planning-mode",
                "force_plan",
                "--continue-plan",
                "--selected-tool",
                "read_file",
                "--selected-tool",
                "inspect_trace",
                "--local-only",
            ]
            stdout = io.StringIO()
            with mock.patch("sys.argv", argv), mock.patch("sys.stdout", stdout), mock.patch(
                "core.runtime.kernel.DevenvKernel.execute_turn", new=fake_execute_turn
            ):
                exit_code = smoke.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["prompt"], "hello")
            self.assertEqual(captured["planning_mode"].value, "force_plan")
            self.assertTrue(captured["continue_plan"])
            self.assertTrue(captured["local_only"])
            self.assertEqual(captured["selected_tools"], ["read_file", "inspect_trace"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["metadata"]["backend_used"], "local")

    def test_smoke_main_persists_plan_state_across_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)

            first_argv = [
                "smoke.py",
                str(workspace),
                "Create a tiny static HTML notes app in folder notesapp with index.html, styles.css, and script.js.",
                "--planning-mode",
                "force_plan",
                "--local-only",
            ]
            first_stdout = io.StringIO()
            with mock.patch("sys.argv", first_argv), mock.patch("sys.stdout", first_stdout):
                first_exit_code = smoke.main()

            self.assertEqual(first_exit_code, 0)
            first_payload = json.loads(first_stdout.getvalue())
            self.assertEqual(first_payload["metadata"]["target_path_hint"], "notesapp")

            second_argv = [
                "smoke.py",
                str(workspace),
                "continue",
                "--continue-plan",
                "--local-only",
            ]
            second_stdout = io.StringIO()
            with mock.patch("sys.argv", second_argv), mock.patch("sys.stdout", second_stdout):
                second_exit_code = smoke.main()

            self.assertEqual(second_exit_code, 0)
            second_payload = json.loads(second_stdout.getvalue())
            self.assertEqual(second_payload["final_response"], "Nothing left to execute.")
            self.assertEqual(second_payload["metadata"]["original_objective"], first_payload["metadata"]["original_objective"])
