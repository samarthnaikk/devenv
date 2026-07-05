from __future__ import annotations

import io
import json
import unittest
from urllib import error
from unittest.mock import MagicMock, patch

from core.ai.opencode_client import (
    OpenCodeClient,
    OpenCodeClientError,
    OpenCodeModelRef,
    OpenCodeServerConfig,
    OpenCodeServerManager,
    OpenCodeToolSpec,
)


class _FakeHTTPResponse:
    def __init__(self, payload: str, *, status: int = 200, content_type: str = "application/json") -> None:
        self.status = status
        self._buffer = io.BytesIO(payload.encode("utf-8"))
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._buffer.read()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        line = self._buffer.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class OpenCodeClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenCodeClient(OpenCodeServerConfig(base_url="http://127.0.0.1:4096", timeout_seconds=2.0))

    def test_health_reads_server_payload(self) -> None:
        with patch("core.ai.opencode_client.request.urlopen", return_value=_FakeHTTPResponse('{"healthy": true, "version": "1.3.3"}')):
            payload = self.client.health()

        self.assertTrue(payload.healthy)
        self.assertEqual(payload.version, "1.3.3")

    def test_create_session_and_send_message_use_structured_json_requests(self) -> None:
        captured_requests = []

        def fake_urlopen(req, timeout=0):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            captured_requests.append({"method": req.get_method(), "url": req.full_url, "body": body})
            if req.full_url.endswith("/session"):
                return _FakeHTTPResponse('{"id": "ses_123", "title": "Demo"}')
            return _FakeHTTPResponse('{"info":{"id":"msg_1","structured_output":{"type":"final","content":"done"}},"parts":[{"type":"text","text":"done"}]}')

        with patch("core.ai.opencode_client.request.urlopen", side_effect=fake_urlopen):
            session = self.client.create_session(title="Demo")
            message = self.client.send_message(
                session.session_id,
                parts=[{"type": "text", "text": "hello"}],
                model=OpenCodeModelRef(provider_id="openrouter", model_id="test-model"),
                system="Be brief",
                tools=[OpenCodeToolSpec(name="read_file", description="Read a file", parameters={"type": "object"})],
                output_format={"type": "json_schema", "schema": {"type": "object"}},
            )

        self.assertEqual(session.session_id, "ses_123")
        self.assertEqual(message.message_id, "msg_1")
        self.assertEqual(message.structured_output["type"], "final")
        self.assertEqual(captured_requests[0]["body"], {"title": "Demo"})
        sent_body = captured_requests[1]["body"]
        self.assertEqual(sent_body["model"]["providerID"], "openrouter")
        self.assertEqual(sent_body["system"], "Be brief")
        self.assertEqual(sent_body["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(sent_body["outputFormat"]["type"], "json_schema")

    def test_list_messages_and_abort_session(self) -> None:
        def fake_urlopen(req, timeout=0):
            if req.full_url.endswith("/message?limit=5"):
                return _FakeHTTPResponse('[{"info":{"id":"msg_1"},"parts":[{"type":"text","text":"one"}]}]')
            return _FakeHTTPResponse("true")

        with patch("core.ai.opencode_client.request.urlopen", side_effect=fake_urlopen):
            messages = self.client.list_messages("ses_123", limit=5)
            aborted = self.client.abort_session("ses_123")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_id, "msg_1")
        self.assertTrue(aborted)

    def test_stream_events_yields_sse_payloads(self) -> None:
        raw = 'event: status\ndata: {"phase":"ready"}\n\n'
        with patch("core.ai.opencode_client.request.urlopen", return_value=_FakeHTTPResponse(raw, content_type="text/event-stream")):
            events = list(self.client.stream_events())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "status")
        self.assertEqual(events[0].data["phase"], "ready")

    def test_http_errors_raise_typed_client_error(self) -> None:
        http_error = error.HTTPError(
            url="http://127.0.0.1:4096/global/health",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"bad request"}'),
        )

        with patch("core.ai.opencode_client.request.urlopen", side_effect=http_error):
            with self.assertRaises(OpenCodeClientError) as ctx:
                self.client.health()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(str(ctx.exception), "bad request")

    def test_server_manager_reports_reachable_health(self) -> None:
        manager = OpenCodeServerManager(config=self.client.config)
        with patch.object(OpenCodeClient, "health", return_value=type("Health", (), {"healthy": True, "version": "1.3.3", "detail": ""})()):
            status = manager.inspect()

        self.assertTrue(status.reachable)
        self.assertTrue(status.healthy)
        self.assertEqual(status.version, "1.3.3")

    def test_server_manager_starts_process_when_needed(self) -> None:
        manager = OpenCodeServerManager(config=self.client.config, executable="opencode", startup_timeout_seconds=0.2)
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(OpenCodeServerManager, "inspect", side_effect=[
            type("Status", (), {"reachable": False, "healthy": False, "detail": "down", "base_url": self.client.config.base_url})(),
            type("Status", (), {"reachable": True, "healthy": True, "detail": "up", "base_url": self.client.config.base_url})(),
        ]), patch("core.ai.opencode_client.shutil.which", return_value="/opt/homebrew/bin/opencode"), patch(
            "core.ai.opencode_client.subprocess.Popen",
            return_value=process,
        ):
            status = manager.ensure_server()

        self.assertTrue(status.reachable)
        self.assertTrue(status.healthy)


if __name__ == "__main__":
    unittest.main()
