from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.ai.ollama_backend import OllamaAICore
from core.tools.base import BaseTool, ToolResult


class FakeTool(BaseTool):
    name = "read_file"
    description = "Read a file"

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=str(kwargs), data={})


class OllamaBackendTest(unittest.TestCase):
    def test_list_models_reads_gguf_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_dir = Path(tempdir) / "models"
            model_dir.mkdir()
            (model_dir / "tiny.gguf").write_text("x", encoding="utf-8")
            with patch.dict("os.environ", {"DEVENV_LLAMA_CPP_MODELS_DIR": str(model_dir)}):
                core = OllamaAICore(workspace_path=tempdir)
                models = core.list_models()

        self.assertIn("tiny.gguf", models)

    def test_chat_runs_llama_cpp_subprocess_and_returns_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir) / "tiny.gguf"
            model_path.write_text("x", encoding="utf-8")
            core = OllamaAICore(workspace_path=tempdir, model=str(model_path))
            with patch("core.ai.ollama_backend.shutil.which", return_value="/usr/local/bin/llama-cli"):
                with patch(
                    "core.ai.ollama_backend.subprocess.run",
                    return_value=_completed(stdout="Hello world\n"),
                ):
                    response = core.chat(messages=[{"role": "user", "content": "Say hello"}])

        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.backend, "ollama")
        self.assertEqual(response.metadata["runtime"], "llama.cpp")

    def test_chat_parses_tool_call_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir) / "tiny.gguf"
            model_path.write_text("x", encoding="utf-8")
            core = OllamaAICore(workspace_path=tempdir, model=str(model_path))
            core.register_tool(FakeTool())
            with patch("core.ai.ollama_backend.shutil.which", return_value="/usr/local/bin/llama-cli"):
                with patch(
                    "core.ai.ollama_backend.subprocess.run",
                    return_value=_completed(
                        stdout=json.dumps(
                            {
                                "type": "tool_call",
                                "tool_name": "read_file",
                                "arguments": {"path": "README.md"},
                            }
                        )
                    ),
                ):
                    response = core.chat(
                        messages=[{"role": "user", "content": "Open the readme"}],
                        tool_names=["read_file"],
                    )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")

    def test_chat_uses_json_schema_for_planner_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir) / "tiny.gguf"
            model_path.write_text("x", encoding="utf-8")
            core = OllamaAICore(workspace_path=tempdir, model=str(model_path))
            core.register_tool(FakeTool())
            captured: dict[str, object] = {}

            def fake_run(command, **kwargs):
                captured["command"] = command
                return _completed(
                    stdout=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "task-1", "description": "Inspect runtime", "level": 0}
                            ],
                            "edges": [],
                        }
                    )
                )

            with patch("core.ai.ollama_backend.shutil.which", return_value="/usr/local/bin/llama-cli"):
                with patch("core.ai.ollama_backend.subprocess.run", side_effect=fake_run):
                    response = core.chat(
                        messages=[
                            {"role": "system", "content": "PLANNER_OUTPUT_MODE: blueprint_json"},
                            {"role": "user", "content": "Plan the work"},
                        ],
                        tool_names=["read_file"],
                        temperature=0.0,
                    )

        self.assertEqual(response.finish_reason, "stop")
        command = captured["command"]
        self.assertIn("--json-schema", command)

    def test_status_reports_cli_missing_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir)
            with patch("core.ai.ollama_backend.shutil.which", return_value=None):
                status = core.status()

        self.assertFalse(status.available)
        self.assertIn("llama.cpp CLI", status.detail)

    def test_low_performance_mode_uses_single_thread_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir) / "tiny.gguf"
            model_path.write_text("x", encoding="utf-8")
            core = OllamaAICore(workspace_path=tempdir, model=str(model_path))
            core.set_performance_mode("low")
            captured: dict[str, object] = {}

            def fake_run(command, **kwargs):
                captured["command"] = command
                return _completed(stdout="done")

            with patch("core.ai.ollama_backend.shutil.which", return_value="/usr/local/bin/llama-cli"):
                with patch("core.ai.ollama_backend.subprocess.run", side_effect=fake_run):
                    core.chat(messages=[{"role": "user", "content": "Hi"}])

        command = captured["command"]
        thread_index = command.index("--threads")
        self.assertEqual(command[thread_index + 1], "1")


def _completed(*, stdout: str, stderr: str = "", returncode: int = 0):
    return type(
        "Completed",
        (),
        {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
        },
    )()


if __name__ == "__main__":
    unittest.main()
