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
