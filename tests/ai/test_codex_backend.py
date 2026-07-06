from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from core.ai.codex_backend import CodexAICore, CodexRunResult
from core.ai.models import AIExecutedToolStep


class CodexBackendTest(unittest.TestCase):
    def test_codex_core_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = CodexAICore(
                workspace_path=tempdir,
                model="gpt-5-codex",
                runner=_FakeCodexRunner(),
                mcp_server_manager=_FakeMCPManager(),
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError) as ctx:
                    core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertIn("openai_api_key", str(ctx.exception).lower())

    def test_codex_core_requires_model(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = CodexAICore(
                workspace_path=tempdir,
                model="",
                runner=_FakeCodexRunner(),
                mcp_server_manager=_FakeMCPManager(),
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                with self.assertRaises(RuntimeError) as ctx:
                    core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertIn("devenv_codex_model", str(ctx.exception).lower())

    def test_codex_core_returns_executed_steps_from_runner(self) -> None:
        runner = _FakeCodexRunner(
            result=CodexRunResult(
                content="Codex answered",
                usage={"total_tokens": 11},
                run_id="run_123",
                metadata={"provider": "openai"},
                executed_steps=(
                    AIExecutedToolStep(
                        step_id="step_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                        output='{"success": true, "output": "hello", "data": {}}',
                        success=True,
                    ),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as tempdir:
            core = CodexAICore(
                workspace_path=tempdir,
                model="gpt-5-codex",
                runner=runner,
                mcp_server_manager=_FakeMCPManager(),
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                response = core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=["read_file"])

        self.assertEqual(response.backend, "codex")
        self.assertEqual(response.content, "Codex answered")
        self.assertEqual(response.usage["total_tokens"], 11)
        self.assertEqual(len(response.executed_steps), 1)
        self.assertEqual(response.executed_steps[0].tool_name, "read_file")
        self.assertEqual(runner.calls[0]["allowed_tools"], ["read_file"])
        self.assertEqual(runner.calls[0]["mcp_server_url"], "http://127.0.0.1:8765/mcp")

    def test_codex_core_aborts_active_run(self) -> None:
        runner = _FakeCodexRunner(
            result=CodexRunResult(
                content="Codex answered",
                usage={"total_tokens": 11},
                run_id="run_123",
            )
        )
        with tempfile.TemporaryDirectory() as tempdir:
            core = CodexAICore(
                workspace_path=tempdir,
                model="gpt-5-codex",
                runner=runner,
                mcp_server_manager=_FakeMCPManager(),
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])
            aborted = core.abort()

        self.assertTrue(aborted)
        self.assertEqual(runner.aborted_run_ids, ["run_123"])


class _FakeMCPStatus:
    base_url = "http://127.0.0.1:8765/mcp"


class _FakeMCPManager:
    def __init__(self) -> None:
        self.config = type("Config", (), {"auth_token": "token-1"})()

    def ensure_server(self):
        return _FakeMCPStatus()

    def inspect(self):
        return type("Status", (), {"to_metadata": lambda self: {"reachable": True, "base_url": _FakeMCPStatus.base_url}})()


class _FakeCodexRunner:
    def __init__(self, result: CodexRunResult | None = None) -> None:
        self.result = result or CodexRunResult(content="ok", run_id="run_default")
        self.calls: list[dict[str, object]] = []
        self.aborted_run_ids: list[str] = []

    def run_turn(self, **kwargs) -> CodexRunResult:
        self.calls.append(kwargs)
        return self.result

    def abort_run(self, run_id: str) -> bool:
        self.aborted_run_ids.append(run_id)
        return True


if __name__ == "__main__":
    unittest.main()
