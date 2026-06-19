from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib import error

from core.ai import AICore
from core.tools.read_file import ReadFileTool


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class AICoreTest(unittest.TestCase):
    def test_missing_api_key_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                with patch.dict("os.environ", {}, clear=True):
                    with self.assertRaises(ValueError):
                        AICore()
            finally:
                os.chdir(original_cwd)

    def test_model_precedence_prefers_constructor_then_env_then_default(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key", "GROQ_MODEL": "env-model"}, clear=True):
            self.assertEqual(AICore().model, "env-model")
            self.assertEqual(AICore(model="ctor-model").model, "ctor-model")

        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            self.assertEqual(AICore().model, "llama-3.3-70b-versatile")

    def test_loads_groq_configuration_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = os.path.join(tempdir, ".env")
            with open(env_path, "w", encoding="utf-8") as handle:
                handle.write("GROQ_API_KEY=dotenv-key\n")
                handle.write("GROQ_MODEL=dotenv-model\n")

            original_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                with patch.dict("os.environ", {}, clear=True):
                    core = AICore()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(core.api_key, "dotenv-key")
        self.assertEqual(core.model, "dotenv-model")

    def test_build_tool_definitions_matches_read_file_schema(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore(tools=[ReadFileTool()])

        definitions = core._build_tool_definitions()

        self.assertEqual(len(definitions), 1)
        function = definitions[0]["function"]
        parameters = function["parameters"]
        self.assertEqual(function["name"], "read_file")
        self.assertEqual(parameters["required"], ["path"])
        self.assertEqual(parameters["properties"]["features"]["enum"], ["content", "metadata", "extension", "all"])

    def test_compile_system_frame_preserves_section_order(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore(tools=[ReadFileTool()], system_instructions="Static rules")

        frame = core._compile_system_frame("## Retrieved Memory\n- remember this")

        self.assertLess(frame.index("## System Core Instructions"), frame.index("## Reconciled Tool Declarations"))
        self.assertLess(frame.index("## Reconciled Tool Declarations"), frame.index("## Cognitive Memory Context"))
        self.assertIn("Static rules", frame)
        self.assertIn('"name": "read_file"', frame)

    def test_compile_system_frame_omits_blank_memory_block(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore(tools=[ReadFileTool()])

        frame = core._compile_system_frame("   ")

        self.assertNotIn("## Cognitive Memory Context", frame)

    def test_chat_posts_payload_and_parses_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md","features":"content"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

        captured_request: dict[str, object] = {}

        def fake_urlopen(req):
            captured_request["url"] = req.full_url
            captured_request["body"] = json.loads(req.data.decode("utf-8"))
            captured_request["authorization"] = req.headers["Authorization"]
            return FakeHTTPResponse(payload)

        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore(tools=[ReadFileTool()])

        with patch("core.ai.engine.request.urlopen", side_effect=fake_urlopen):
            response = core.chat(
                messages=[{"role": "user", "content": "Open the readme"}],
                memory_context="## Retrieved Memory\n- Repo uses local memory",
            )

        self.assertEqual(captured_request["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(captured_request["authorization"], "Bearer env-key")
        body = captured_request["body"]
        self.assertEqual(body["model"], "llama-3.3-70b-versatile")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["content"], "Open the readme")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].call_id, "call_123")
        self.assertEqual(response.tool_calls[0].arguments["path"], "README.md")
        self.assertEqual(response.usage["total_tokens"], 18)

    def test_post_chat_completion_network_failure_raises_runtime_error(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore()

        with patch("core.ai.engine.request.urlopen", side_effect=error.URLError("offline")):
            with self.assertRaises(RuntimeError):
                core.chat(messages=[{"role": "user", "content": "hello"}])

    def test_parse_response_rejects_malformed_tool_call_arguments_json(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {"name": "read_file", "arguments": '{"path":'},
                            }
                        ],
                    },
                }
            ]
        }

        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=True):
            core = AICore()

        with self.assertRaises(ValueError):
            core._parse_response(payload)


if __name__ == "__main__":
    unittest.main()
