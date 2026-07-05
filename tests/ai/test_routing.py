from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from core.ai.models import AIResponse, ToolCallRequest
from core.ai.opencode_client import OpenCodeClientError
from core.ai.routing import DEFAULT_OPENCODE_MODEL, OpenCodeAICore, RoutingAICore, _opencode_output_format
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
    def test_opencode_core_uses_stable_default_model_when_env_is_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            core = OpenCodeAICore(workspace_path=".")

        self.assertEqual(core.model, DEFAULT_OPENCODE_MODEL)

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
        self.assertNotIn("tools", client.sent_messages[0])
        self.assertIn("Available Tools", client.sent_messages[0]["parts"][0]["text"])

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

            previous = os.environ.get("DEVENV_OPENCODE_USE_LEGACY_CLI")
            os.environ["DEVENV_OPENCODE_USE_LEGACY_CLI"] = "1"
            try:
                core = OpenCodeAICore(workspace_path=tempdir, executable=str(script_path), model="openrouter/test")
                response = core.chat(messages=[{"role": "user", "content": "hello"}], memory_context="## Memory")
            finally:
                if previous is None:
                    os.environ.pop("DEVENV_OPENCODE_USE_LEGACY_CLI", None)
                else:
                    os.environ["DEVENV_OPENCODE_USE_LEGACY_CLI"] = previous

        self.assertEqual(response.content, "OpenCode answered")
        self.assertEqual(response.usage["total_tokens"], 13)

    def test_output_format_restricts_tool_calls_to_scoped_tools(self) -> None:
        payload = _opencode_output_format(["read_file"])

        self.assertEqual(payload["type"], "json_schema")
        self.assertEqual(payload["schema"]["properties"]["tool_name"]["enum"], ["read_file"])

    def test_opencode_core_recovers_from_stale_server_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("Unknown session", status_code=404),
                    _fake_message("msg_2", structured_output={"type": "final", "content": "Recovered answer"}, usage={"total_tokens": 7}),
                ],
                session_ids=["ses_old", "ses_new"],
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            response = core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])
            status = core.status()

        self.assertEqual(response.content, "Recovered answer")
        self.assertEqual(client.created_titles, [f"Devenv: {Path(tempdir).name}", f"Devenv: {Path(tempdir).name}"])
        self.assertEqual(status.metadata["session_id"], "ses_new")
        self.assertEqual(status.metadata["transport"], "server")

    def test_opencode_core_retries_without_structured_output_after_400(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("Model does not support structured output", status_code=400),
                    _fake_message("msg_2", structured_output=None, usage={"total_tokens": 7}, text="Recovered plain text"),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            response = core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        self.assertEqual(response.content, "Recovered plain text")
        self.assertEqual(len(client.sent_messages), 2)
        self.assertEqual(client.sent_messages[0]["output_format"]["type"], "json_schema")
        self.assertIsNone(client.sent_messages[1]["output_format"])
        self.assertIn("retried without output schema", core.last_backend_fallback.lower())

    def test_opencode_core_retries_tool_turn_without_structured_output_after_400(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("response_format unsupported", status_code=400),
                    _fake_message(
                        "msg_2",
                        structured_output=None,
                        usage={"total_tokens": 9},
                        text='{"type":"tool_call","tool_name":"read_file","arguments":{"path":"README.md"}}',
                    ),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            core.register_tool(FakeTool())
            response = core.chat(messages=[{"role": "user", "content": "open the readme"}], tool_names=["read_file"])

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        self.assertIsNone(client.sent_messages[1]["output_format"])

    def test_opencode_core_disables_structured_output_after_first_400_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("response_format unsupported", status_code=400),
                    _fake_message("msg_1", structured_output=None, usage={"total_tokens": 5}, text="Recovered plain text"),
                    _fake_message("msg_2", structured_output=None, usage={"total_tokens": 4}, text="Second plain text"),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            first = core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])
            second = core.chat(messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": first.content}, {"role": "user", "content": "again"}], tool_names=[])

        self.assertEqual(first.content, "Recovered plain text")
        self.assertEqual(second.content, "Second plain text")
        self.assertEqual(len(client.sent_messages), 3)
        self.assertEqual(client.sent_messages[0]["output_format"]["type"], "json_schema")
        self.assertIsNone(client.sent_messages[1]["output_format"])
        self.assertIsNone(client.sent_messages[2]["output_format"])
        self.assertFalse(core.status().metadata["structured_output_supported"])

    def test_opencode_core_caches_transport_failure_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[OpenCodeClientError("Unable to reach OpenCode server: operation not permitted")]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            with self.assertRaises(RuntimeError):
                core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])
            with self.assertRaises(RuntimeError) as retry_ctx:
                core.chat(messages=[{"role": "user", "content": "hello again"}], tool_names=[])

        self.assertEqual(len(client.sent_messages), 1)
        self.assertIn("recent transport failure cached", str(retry_ctx.exception).lower())

    def test_opencode_core_raises_when_server_400_persists_without_cli_fallback_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("bad request", status_code=400),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            with patch.object(
                core,
                "_legacy_cli_chat",
                return_value=AIResponse(content="CLI rescue", tool_calls=(), finish_reason="stop", usage={"total_tokens": 3}),
            ) as legacy_chat:
                with self.assertRaises(RuntimeError):
                    core.chat(messages=[{"role": "user", "content": "hello"}], tool_names=[])

        legacy_chat.assert_not_called()
        self.assertIn("opencode server failed", core.last_error.lower())

    def test_opencode_core_allows_explicit_cli_fallback_for_degraded_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            client = _FakeOpenCodeClient(
                responses=[
                    OpenCodeClientError("unprocessable entity", status_code=422),
                ]
            )
            manager = Mock()
            manager.ensure_server.return_value = Mock()
            manager.inspect.return_value = _fake_server_status()

            core = OpenCodeAICore(workspace_path=tempdir, executable="opencode", client=client, server_manager=manager)
            core.register_tool(FakeTool())
            cli_tool_response = AIResponse(
                content="",
                tool_calls=(ToolCallRequest(call_id="call_1", tool_name="read_file", arguments={"path": "README.md"}),),
                finish_reason="tool_calls",
                usage={"total_tokens": 4},
            )
            with patch.dict(os.environ, {"DEVENV_OPENCODE_ALLOW_CLI_FALLBACK": "1"}, clear=False):
                with patch.object(core, "_legacy_cli_chat", return_value=cli_tool_response) as legacy_chat:
                    response = core.chat(messages=[{"role": "user", "content": "open the readme"}], tool_names=["read_file"])

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        legacy_chat.assert_called_once()
        self.assertIn("fell back to cli transport", core.last_backend_fallback.lower())


class _FakeOpenCodeClient:
    def __init__(self, *, responses: list[Any] | None = None, error: Exception | None = None, session_ids: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.session_ids = list(session_ids or ["ses_123"])
        self.created_titles: list[str] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.aborted_session_ids: list[str] = []

    def create_session(self, *, title: str | None = None, parent_id: str | None = None):
        self.created_titles.append(title or "")
        return type("Session", (), {"session_id": self.session_ids.pop(0) if self.session_ids else "ses_123"})()

    def send_message(self, session_id: str, **kwargs):
        self.sent_messages.append({"session_id": session_id, **kwargs})
        if self.error is not None:
            raise self.error
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def abort_session(self, session_id: str) -> bool:
        self.aborted_session_ids.append(session_id)
        return True


def _fake_message(message_id: str, *, structured_output: dict[str, Any] | None, usage: dict[str, int], text: str | None = None):
    rendered_text = text if text is not None else (structured_output or {}).get("content", "")
    return type(
        "Message",
        (),
        {
            "raw": {"info": {"id": message_id, "structured_output": structured_output, "usage": usage}, "parts": [{"type": "text", "text": rendered_text}]},
            "parts": ({"type": "text", "text": rendered_text},),
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
