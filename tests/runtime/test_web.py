from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse
from core.runtime.models import RunConfig
from core.runtime.web import DevenvWebApp


@dataclass(frozen=True)
class FakeRetrievalResult:
    markdown_context: str


class FakeMemory:
    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        return None

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context="")

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "log-1"


class FakeAI:
    def __init__(self) -> None:
        self.responses = [
            AIResponse(content="Website response", tool_calls=(), finish_reason="stop", usage={"prompt_tokens": 3})
        ]
        self.registered_tools: list[str] = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def chat(self, messages: list[dict[str, Any]], memory_context: str | None = None, temperature: float = 0.2) -> AIResponse:
        return self.responses.pop(0)


class DevenvWebAppTest(unittest.TestCase):
    def test_payload_helpers_expose_workspace_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("hello", encoding="utf-8")
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            health = app.build_health_payload()
            files = app.build_files_payload()
            file_payload = app.build_file_payload("README.md")

        self.assertEqual(health["status"], "ok")
        self.assertEqual(files["entries"][0]["name"], "README.md")
        self.assertEqual(file_payload["content"], "hello")

    def test_http_endpoints_serve_health_files_and_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("hello", encoding="utf-8")
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                port=0,
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            server = app.create_server()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            self.addCleanup(thread.join, 1.0)

            port = server.server_address[1]
            connection = http.client.HTTPConnection("127.0.0.1", port)

            connection.request("GET", "/api/health")
            health = json.loads(connection.getresponse().read().decode("utf-8"))

            connection.request("GET", "/api/files")
            files = json.loads(connection.getresponse().read().decode("utf-8"))

            connection.request("GET", "/api/file?path=../secrets.txt")
            invalid_file_response = connection.getresponse()
            invalid_file = json.loads(invalid_file_response.read().decode("utf-8"))

            connection.request("POST", "/api/turn", body=json.dumps({"prompt": "hello"}), headers={"Content-Type": "application/json"})
            turn = json.loads(connection.getresponse().read().decode("utf-8"))

            connection.close()

        self.assertEqual(health["status"], "ok")
        self.assertEqual(files["entries"][0]["name"], "README.md")
        self.assertEqual(invalid_file_response.status, 400)
        self.assertIn("escapes workspace", invalid_file["error"])
        self.assertEqual(turn["final_response"], "Website response")


if __name__ == "__main__":
    unittest.main()
