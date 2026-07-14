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
    def test_list_models_reads_tags_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir)
            with patch("core.ai.ollama_backend.request.urlopen", return_value=_FakeHTTPResponse(json.dumps({"models": [{"name": "qwen2.5:3b"}]}))):
                models = core.list_models()

        self.assertIn("qwen2.5:3b", models)

    def test_chat_streams_final_text_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5:3b")
            stream = "\n".join(
                [
                    json.dumps({"message": {"content": "Hello "}, "done": False}),
                    json.dumps({"message": {"content": "world"}, "done": True, "prompt_eval_count": 12, "eval_count": 4}),
                ]
            )
            with patch("core.ai.ollama_backend.request.urlopen", return_value=_FakeHTTPResponse(stream)):
                response = core.chat(messages=[{"role": "user", "content": "Say hello"}])

        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.usage["total_tokens"], 16)
        self.assertEqual(response.backend, "ollama")

    def test_chat_parses_tool_call_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir, model="qwen2.5:3b")
            core.register_tool(FakeTool())
            stream = json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "type": "tool_call",
                                "tool_name": "read_file",
                                "arguments": {"path": "README.md"},
                            }
                        )
                    },
                    "done": True,
                }
            )
            with patch("core.ai.ollama_backend.request.urlopen", return_value=_FakeHTTPResponse(stream)):
                response = core.chat(
                    messages=[{"role": "user", "content": "Open the readme"}],
                    tool_names=["read_file"],
                )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments["path"], "README.md")

    def test_status_reports_not_running_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            core = OllamaAICore(workspace_path=tempdir)
            with patch("core.ai.ollama_backend.request.urlopen", side_effect=OSError("boom")):
                with patch.object(core, "_request_json", side_effect=RuntimeError("Ollama is not running at http://127.0.0.1:11434. Start Ollama and try again. (boom)")):
                    status = core.status()

        self.assertFalse(status.available)
        self.assertIn("Ollama is not running", status.detail)


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._lines = [line.encode("utf-8") + b"\n" for line in body.splitlines()]
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b"".join(self._lines)

    def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line


if __name__ == "__main__":
    unittest.main()
