from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)

from core.ai.acp_agent import ACPAgentError, ACPAgentSession
from core.ai.agents import AgentSpec

_FAKE_AGENT = r"""
import sys, json

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1, "agentCapabilities": {}, "agentInfo": {"name": "fake", "title": "Fake Agent", "version": "0.0.1"}, "authMethods": []}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "sess-1"}})
    elif method == "session/prompt":
        sid = msg["params"]["sessionId"]
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "hello from fake"}}}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})
"""

_FAKE_PERMISSION_AGENT = r"""
import sys, json

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

pending = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": []}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "sess-2"}})
    elif method == "session/prompt":
        pending = mid
        send({"jsonrpc": "2.0", "id": 555, "method": "session/request_permission", "params": {"sessionId": msg["params"]["sessionId"], "toolCall": {"toolCallId": "t1", "title": "Run tests", "kind": "execute"}, "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"}, {"optionId": "reject", "name": "Reject", "kind": "reject_once"}]}})
    elif method is None and mid == 555:
        with open("perm-result.json", "w") as fh:
            fh.write(json.dumps(msg.get("result")))
        send({"jsonrpc": "2.0", "id": pending, "result": {"stopReason": "end_turn"}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})
"""


def _fake_spec(script: str) -> AgentSpec:
    return AgentSpec(
        name="fake",
        title="Fake Agent",
        command=sys.executable,
        args=("-c", script),
    )


class ACPAgentSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_prompt_and_close(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            session = ACPAgentSession(_fake_spec(_FAKE_AGENT), tempdir)
            info = await session.start()

            self.assertEqual(info.session_id, "sess-1")
            self.assertEqual(info.agent_title, "Fake Agent")

            stop_reason = await session.send_prompt("hi")
            self.assertEqual(stop_reason, "end_turn")

            updates = []
            while not session.events.empty():
                updates.append(session.events.get_nowait())

            await session.close()

        self.assertTrue(any(isinstance(update, AgentMessageChunk) for update in updates))

    async def test_permission_handler_receives_tool_call(self) -> None:
        seen: dict[str, object] = {}

        async def handler(tool_call: ToolCallUpdate, options: list[PermissionOption]):
            seen["tool_call"] = tool_call
            seen["options"] = options
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id="allow", outcome="selected")
            )

        with tempfile.TemporaryDirectory() as tempdir:
            session = ACPAgentSession(
                _fake_spec(_FAKE_PERMISSION_AGENT),
                tempdir,
                permission_handler=handler,
            )
            await session.start()
            await session.send_prompt("run the tests")
            await session.close()

            result_path = Path(tempdir) / "perm-result.json"
            recorded = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(seen["tool_call"].tool_call_id, "t1")
        self.assertEqual([option.option_id for option in seen["options"]], ["allow", "reject"])
        self.assertIn("allow", json.dumps(recorded))

    async def test_read_and_write_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            session = ACPAgentSession(_fake_spec(_FAKE_AGENT), tempdir)
            client = session._client

            target = Path(tempdir) / "note.txt"
            target.write_text("hello", encoding="utf-8")
            read_response = await client.read_text_file("s", str(target))
            self.assertEqual(read_response.content, "hello")

            written = Path(tempdir) / "nested" / "out.txt"
            await client.write_text_file("s", str(written), "world")
            self.assertEqual(written.read_text(encoding="utf-8"), "world")
            await session.close()

    async def test_start_fails_when_command_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            spec = AgentSpec(
                name="missing",
                title="Missing",
                command="definitely-not-a-real-agent-binary",
                args=(),
            )
            session = ACPAgentSession(spec, tempdir)
            with self.assertRaises(ACPAgentError):
                await session.start()


if __name__ == "__main__":
    unittest.main()
