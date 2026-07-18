from __future__ import annotations

import json
import tempfile
import unittest
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
    def test_list_models_reads_ollama_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir)
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                return_value=_response({"models": [{"name": "qwen2.5-coder:3b"}]}),
            ):
                models = core.list_models()

        self.assertEqual(models, ["qwen2.5-coder:3b"])

    def test_chat_posts_to_ollama_and_returns_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5-coder:3b")
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                return_value=_response(
                    {
                        "message": {"role": "assistant", "content": "Hello world"},
                        "prompt_eval_count": 12,
                        "eval_count": 7,
                    }
                ),
            ) as mock_urlopen:
                response = core.chat(messages=[{"role": "user", "content": "Say hello"}])

        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.backend, "ollama")
        self.assertEqual(response.metadata["runtime"], "ollama")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen2.5-coder:3b")
        self.assertFalse(payload["stream"])

    def test_chat_parses_tool_call_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5-coder:3b")
            core.register_tool(FakeTool())
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                return_value=_response(
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "type": "tool_call",
                                    "tool_name": "read_file",
                                    "arguments": {"path": "README.md"},
                                }
                            ),
                        }
                    }
                ),
            ):
                response = core.chat(
                    messages=[{"role": "user", "content": "Open the readme"}],
                    tool_names=["read_file"],
                )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")

    def test_chat_uses_schema_format_for_planner_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5-coder:3b")
            core.register_tool(FakeTool())
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                return_value=_response(
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "tasks": [
                                        {"task_id": "task-1", "description": "Inspect runtime", "level": 0}
                                    ],
                                    "edges": [],
                                }
                            ),
                        }
                    }
                ),
            ) as mock_urlopen:
                response = core.chat(
                    messages=[
                        {"role": "system", "content": "PLANNER_OUTPUT_MODE: blueprint_json"},
                        {"role": "user", "content": "Plan the work"},
                    ],
                    tool_names=["read_file"],
                    temperature=0.0,
                )

        self.assertEqual(response.finish_reason, "stop")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("format", payload)
        self.assertEqual(payload["options"]["temperature"], 0.0)

    def test_status_reports_ollama_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir)
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                side_effect=OSError("connection refused"),
            ):
                status = core.status()

        self.assertFalse(status.available)
        self.assertIn("Ollama is not running", status.detail)

    def test_low_performance_mode_uses_single_thread_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5-coder:3b")
            core.set_performance_mode("low")
            with patch(
                "core.ai.ollama_backend.urllib.request.urlopen",
                return_value=_response({"message": {"role": "assistant", "content": "done"}}),
            ) as mock_urlopen:
                core.chat(messages=[{"role": "user", "content": "Hi"}])

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["options"]["num_thread"], 1)


def _response(payload: dict[str, object]):
    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    return _FakeResponse(json.dumps(payload).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
