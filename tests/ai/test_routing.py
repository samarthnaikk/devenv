from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from core.ai.opencode_client import OpenCodeClientError
from core.ai.routing import OpenCodeAICore, RoutingAICore, _opencode_output_format
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

    def test_opencode_core_uses_server_session_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    _fake_message("msg_1", structured_output={"type": "final", "content": "OpenCode answered"}, usage={"total_tokens": 13}),
                    _fake_message("msg_2", structured_output={"type": "final", "content": "OpenCode again"}, usage={"total_tokens": 5}),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(
                workspace_path=tempdir,
                executable="opencode",
                model="openrouter/test",
                client=client,
                server_manager=manager,
            )
            response_one = core.chat(messages=[{"role": "user", "content": "hello"}], memory_context="## Memory")
            response_two = core.chat(messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "OpenCode answered"}, {"role": "user", "content": "again"}])

        self.assertEqual(response_one.content, "OpenCode answered")
        self.assertEqual(response_one.usage["total_tokens"], 13)
        self.assertEqual(response_two.content, "OpenCode again")
        self.assertEqual(client.created_titles, [f"Devenv: {Path(tempdir).name}"])
        self.assertEqual(client.sent_messages[0]["session_id"], "ses_123")
        self.assertEqual(client.sent_messages[1]["session_id"], "ses_123")

    def test_routing_core_prefers_opencode_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[_fake_message("msg_1", structured_output={"type": "final", "content": "OpenCode primary path"}, usage={"total_tokens": 5})]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(
                    workspace_path=tempdir,
                    executable="opencode",
                    model="openrouter/test",
                    client=client,
                    server_manager=manager,
                ),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            response = router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertEqual(response.content, "OpenCode primary path")
        self.assertEqual(router.last_backend_used, "opencode")

    def test_opencode_core_emits_tool_call_from_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    _fake_message(
                        "msg_1",
                        structured_output={"type": "tool_call", "tool_name": "read_file", "arguments": {"path": "README.md"}},
                        usage={"total_tokens": 9},
                    )
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
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
            client = _FakeOpenCodeClient(error=OpenCodeClientError("boom"))
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager),
            )
            router.set_backend_preference("opencode", opencode_enabled=True)
            with self.assertRaises(RuntimeError):
                router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=["read_file"])

        self.assertEqual(router.last_backend_used, "opencode")

    def test_routing_core_requires_opencode_access(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[_fake_message("msg_1", structured_output={"type": "final", "content": "OpenCode primary path"}, usage={"total_tokens": 5})]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            router = RoutingAICore(
                workspace_path=tempdir,
                opencode_ai=OpenCodeAICore(
                    workspace_path=tempdir,
                    executable="opencode",
                    model="openrouter/test",
                    client=client,
                    server_manager=manager,
                ),
            )
            router.set_backend_preference("opencode", opencode_enabled=False)
            with self.assertRaises(RuntimeError):
                router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

    def test_routing_core_forwards_session_lifecycle_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[_fake_message("msg_1", structured_output={"type": "final", "content": "hello"}, usage={"total_tokens": 1})]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()
            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            router = RoutingAICore(workspace_path=tempdir, opencode_ai=core)
            router.set_backend_preference("opencode", opencode_enabled=True)

            router.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])
            aborted = router.abort()
            router.reset_session()

        self.assertTrue(aborted)
        self.assertEqual(client.aborted_session_ids, ["ses_123"])

    def test_legacy_cli_mode_still_parses_json_line_output(self) -> None:
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

    def test_output_format_restricts_tool_calls_to_scoped_tools(self) -> None:
        payload = _opencode_output_format(["read_file"])

        self.assertEqual(payload["type"], "json_schema")
        self.assertEqual(payload["schema"]["properties"]["tool_name"]["enum"], ["read_file"])


class _FakeOpenCodeClient:
    def __init__(self, *, responses: list[Any] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.created_titles: list[str] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.aborted_session_ids: list[str] = []

    def create_session(self, *, title: str | None = None, parent_id: str | None = None):
        self.created_titles.append(title or "")
        return type("Session", (), {"session_id": "ses_123"})()

    def send_message(self, session_id: str, **kwargs):
        self.sent_messages.append({"session_id": session_id, **kwargs})
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)

    def abort_session(self, session_id: str) -> bool:
        self.aborted_session_ids.append(session_id)
        return True


def _fake_message(message_id: str, *, structured_output: dict[str, Any], usage: dict[str, int]):
    return type(
        "Message",
        (),
        {
            "raw": {"info": {"id": message_id, "structured_output": structured_output, "usage": usage}, "parts": [{"type": "text", "text": structured_output.get("content", "")}]},
            "parts": ({"type": "text", "text": structured_output.get("content", "")},),
            "structured_output": structured_output,
        },
    )()


def _fake_server_status():
    return type(
        "Status",
        (),
        {
            "to_metadata": lambda self: {
                "reachable": True,
                "healthy": True,
                "version": "1.3.3",
                "detail": "OpenCode server reachable: 1.3.3",
                "base_url": "http://127.0.0.1:4096",
                "started_by_manager": False,
            }
        },
    )()


if __name__ == "__main__":
    unittest.main()
