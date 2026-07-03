from __future__ import annotations

import os
import stat
import tempfile
import textwrap
import unittest
from pathlib import Path

from core.ai.models import AIResponse
from core.ai.routing import OpenCodeAICore, RoutingAICore


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


class OpenCodeRoutingTest(unittest.TestCase):
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

    def test_routing_core_falls_back_to_groq_for_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
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
        self.assertIn("does not support", router.last_backend_fallback)


if __name__ == "__main__":
    unittest.main()
