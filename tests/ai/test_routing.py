from __future__ import annotations

import os
import stat
import tempfile
import textwrap
import unittest
from pathlib import Path

from core.ai.routing import OpenCodeAICore, RoutingAICore
from core.tools.base import BaseTool, ToolResult


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

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable=str(script_path), model="openrouter/test"),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            response = router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertEqual(response.content, "OpenCode primary path")
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

    def test_routing_core_raises_when_opencode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text("#!/bin/sh\nprintf '%s\\n' 'boom' >&2\nexit 1\n", encoding="utf-8")
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable=str(script_path)),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            with self.assertRaises(RuntimeError):
                router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=["read_file"])

        self.assertEqual(router.last_backend_used, "opencode")

    def test_routing_core_requires_opencode_access(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            script_path = Path(tempdir) / "opencode"
            script_path.write_text(
                "#!/bin/sh\nprintf '%s\\n' '{\"content\":\"OpenCode primary path\",\"usage\":{\"total_tokens\":5}}'\n",
                encoding="utf-8",
            )
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable=str(script_path), model="openrouter/test"),
            )
            router.set_backend_preference("opencode", opencode_enabled=False)
            with self.assertRaises(RuntimeError):
                router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])


if __name__ == "__main__":
    unittest.main()
