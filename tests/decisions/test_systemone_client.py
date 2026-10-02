from __future__ import annotations

import json
import unittest
import urllib.error

from core.decisions.models import DecisionError, DecisionTimeout
from core.decisions.systemone import SystemOneClient


def _ok_body(answers: dict | None = None) -> bytes:
    return json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": answers or {"x": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    ).encode("utf-8")


class SystemOneClientTest(unittest.TestCase):
    def _client(self, transport, **kwargs) -> SystemOneClient:
        kwargs.setdefault("sleep", lambda _seconds: None)
        return SystemOneClient(
            base_url="https://api.example.test",
            transport=transport,
            **kwargs,
        )

    def test_successful_evaluation(self) -> None:
        captured = {}

        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization")
            return 200, _ok_body()

        client = self._client(transport, api_key="abc123")
        result = client.evaluate("state", {"x": {"type": "noul", "instructions": "is it?"}})

        self.assertEqual(result.answers["x"]["noul"], 0.9)
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(captured["url"], "https://api.example.test/v1/systemone")
        self.assertEqual(captured["auth"], "Bearer abc123")

    def test_retries_on_rate_limit_then_succeeds(self) -> None:
        calls = {"n": 0}

        def transport(request, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return 429, b'{"error":"slow down"}'
            return 200, _ok_body()

        client = self._client(transport, retries=1)
        result = client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})
        self.assertEqual(result.answers["x"]["noul"], 0.9)
        self.assertEqual(calls["n"], 2)

    def test_auth_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def transport(request, timeout):
            calls["n"] += 1
            return 401, b'{"error":"unauthorized"}'

        client = self._client(transport, retries=3)
        with self.assertRaises(DecisionError):
            client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})
        self.assertEqual(calls["n"], 1)

    def test_timeout_raises_decision_timeout(self) -> None:
        def transport(request, timeout):
            raise TimeoutError("timed out")

        client = self._client(transport, retries=0)
        with self.assertRaises(DecisionTimeout):
            client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})

    def test_url_error_raises_decision_error(self) -> None:
        def transport(request, timeout):
            raise urllib.error.URLError("connection refused")

        client = self._client(transport, retries=0)
        with self.assertRaises(DecisionError):
            client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})

    def test_malformed_json_raises(self) -> None:
        client = self._client(lambda request, timeout: (200, b"not json"))
        with self.assertRaises(DecisionError):
            client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})

    def test_missing_answers_raises(self) -> None:
        client = self._client(lambda request, timeout: (200, b'{"model":"jev-1.13.0"}'))
        with self.assertRaises(DecisionError):
            client.evaluate("state", {"x": {"type": "noul", "instructions": "?"}})

    def test_empty_questions_rejected(self) -> None:
        client = self._client(lambda request, timeout: (200, _ok_body()))
        with self.assertRaises(DecisionError):
            client.evaluate("state", {})

    def test_missing_base_url_rejected(self) -> None:
        with self.assertRaises(DecisionError):
            SystemOneClient(base_url="")


if __name__ == "__main__":
    unittest.main()
