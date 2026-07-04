from __future__ import annotations

import os
import stat
import tempfile
import textwrap
import unittest
from pathlib import Path

from core.ai.models import AIResponse
from core.ai.routing import OpenCodeAICore, RoutingAICore
from core.tools.base import BaseTool, ToolResult


class FakeGroqAI:
    def __init__(self) -> None:
        self.model = "groq-model"
        self.api_key = "test-key"
        self.calls: list[dict[str, object]] = []

    def register_tool(self, tool) -> None:
        return None

    def chat(self, messages, memory_context=None, temperature=0.2, tool_names=None):
        self.calls.append(
            {
                "messages": messages,
                "memory_context": memory_context,
                "temperature": temperature,
                "tool_names": tuple(tool_names or ()),
            }
        )
        return AIResponse(content="Groq fallback answer", tool_calls=(), finish_reason="stop", usage={"total_tokens": 7})


class FakeTool(BaseTool):
    name = "read_file"
    description = "Read a file from disk."

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=str(kwargs), data={})


class OpenCodeRoutingTest(unittest.TestCase):
    def test_tool_prompt_includes_web_and_large_file_policy(self) -> None:
        core = OpenCodeAICore(workspace_path=".")
        core.register_tool(FakeTool())

        prompt = core._compile_tool_prompt(
            messages=[{"role": "user", "content": "who is the president of india?"}],
            memory_context=None,
            tool_names=["read_file"],
        )

        self.assertIn("Use web_search for current or time-sensitive facts", prompt)
        self.assertIn("AGENTS.md", prompt)

    def test_opencode_core_parses_json_line_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text(
                "#!/bin/sh\nprintf '%s\\n' '{\"content\":\"OpenCode answered\",\"usage\":{\"total_tokens\":13}}'\n",
                encoding="utf-8",
            )
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            core = OpenCodeAICore(workspace_path=tempdir, executable=str(script_path), model="openrouter/test")
            response = core.chat(messages=[{"role": "user", "content": "hello"}], memory_context="## Memory")

        self.assertEqual(response.content, "OpenCode answered")
        self.assertEqual(response.usage["total_tokens"], 13)

    def test_routing_core_prefers_opencode_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text(
                "#!/bin/sh\nprintf '%s\\n' '{\"content\":\"OpenCode primary path\",\"usage\":{\"total_tokens\":5}}'\n",
                encoding="utf-8",
            )
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            groq = FakeGroqAI()
            router = RoutingAICore(
                workspace_path=tempdir,
                groq_ai=groq,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable=str(script_path), model="openrouter/test"),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            response = router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertEqual(response.content, "OpenCode primary path")
        self.assertEqual(groq.calls, [])
        self.assertEqual(router.last_backend_used, "opencode")

    def test_opencode_core_emits_tool_call_from_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text(
                "#!/bin/sh\nprintf '%s\\n' '{\"type\":\"tool_call\",\"tool_name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"},\"usage\":{\"total_tokens\":9}}'\n",
                encoding="utf-8",
            )
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            core = OpenCodeAICore(workspace_path=tempdir, executable=str(script_path))
            core.register_tool(FakeTool())
            response = core.chat(messages=[{"role": "user", "content": "open the readme"}], tool_names=["read_file"])

        self.assertEqual(response.content, "")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments["path"], "README.md")
        self.assertEqual(response.usage["total_tokens"], 9)

    def test_routing_core_falls_back_to_groq_when_opencode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text("#!/bin/sh\nprintf '%s\\n' 'boom' >&2\nexit 1\n", encoding="utf-8")
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            groq = FakeGroqAI()
            router = RoutingAICore(
                workspace_path=tempdir,
                groq_ai=groq,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable=str(script_path)),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            response = router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=["read_file"])

        self.assertEqual(response.content, "Groq fallback answer")
        self.assertEqual(router.last_backend_used, "groq")
        self.assertIn("failed", router.last_backend_fallback.lower())


if __name__ == "__main__":
    unittest.main()
