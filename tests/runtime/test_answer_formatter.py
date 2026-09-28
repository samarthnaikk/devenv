from __future__ import annotations

import json
import unittest
from unittest import mock

from core.runtime.answer_formatter import (
    AnswerFormatter,
    build_answer_formatter,
    build_formatter_messages,
    formatter_enabled,
)


def _evidence() -> dict:
    return {
        "source": "selector",
        "sessions": [
            {
                "session_id": "ses_1cb77a",
                "provider": "opencode",
                "title": "Add Design and Management evaluation tracks",
                "workspace_path": "/Users/x/hirex-frontend",
                "updated_at": "2026-05-17T00:00:00",
            }
        ],
        "lines": [
            "Assistant reported: Added `qualifly_track` column to `jobs` table (engineering, design, management)",
            "Tool output: backend-1 | AttributeError: module 'sqlalchemy' has no attribute 'JSONB'",
            "Assistant reported: up to 6 external resource links per application",
        ],
    }


class FormatterMessagesTest(unittest.TestCase):
    def test_includes_question_and_raw_lines(self) -> None:
        messages = build_formatter_messages("What are the track values?", _evidence())
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertIn("You are a formatter", system)
        self.assertIn("Never mention or restate chat roles", system)
        self.assertIn("QUESTION:", user)
        self.assertIn("engineering, design, management", user)
        self.assertIn("ses_1cb77a", user)

    def test_handles_empty_evidence(self) -> None:
        messages = build_formatter_messages("q", {"lines": [], "sessions": []})
        self.assertIn("(no evidence retrieved)", messages[1]["content"])


class AnswerFormatterTest(unittest.TestCase):
    def test_returns_clean_content(self) -> None:
        def chat(_messages):
            return "- **Tracks:** engineering, design, management\n- **Migration:** 0011"

        formatter = AnswerFormatter(chat)
        result = formatter.format("q", _evidence())
        self.assertIn("**Tracks:**", result)
        self.assertNotIn("Assistant reported", result)

    def test_returns_none_without_evidence(self) -> None:
        formatter = AnswerFormatter(lambda _m: "should not be called")
        self.assertIsNone(formatter.format("q", {"lines": [], "sessions": []}))

    def test_returns_none_on_chat_error(self) -> None:
        def chat(_messages):
            raise RuntimeError("model down")

        formatter = AnswerFormatter(chat)
        self.assertIsNone(formatter.format("q", _evidence()))

    def test_build_answer_formatter_uses_response_content(self) -> None:
        class FakeAI:
            def __init__(self) -> None:
                self.calls = []

            def chat(self, messages, **_kwargs):
                self.calls.append(messages)
                return type("Response", (), {"content": "formatted answer"})()

        ai = FakeAI()
        formatter = build_answer_formatter(ai)
        self.assertEqual(formatter.format("q", _evidence()), "formatted answer")
        self.assertEqual(len(ai.calls), 1)


class FormatterFlagTest(unittest.TestCase):
    def test_enabled_by_default(self) -> None:
        with mock.patch.dict(__import__("os").environ, {}, clear=True):
            self.assertTrue(formatter_enabled())

    def test_can_disable(self) -> None:
        with mock.patch.dict(
            __import__("os").environ, {"DEVENV_ANSWER_FORMATTER": "0"}, clear=True
        ):
            self.assertFalse(formatter_enabled())


class FormatterIntegrationTest(unittest.TestCase):
    def test_raw_evidence_is_not_leaked_into_output(self) -> None:
        """The formatter receives raw evidence but must only emit the answer."""
        raw_context = "\n".join(
            [
                "- Assistant reported: Added `qualifly_track` column to `jobs` table (engineering, design, management)",
                "- Tool output: backend-1 | AttributeError: module 'sqlalchemy' has no attribute 'JSONB'",
                "- Assistant reported: up to 6 external resource links per application",
            ]
        )
        bundle = {
            "source": "selector",
            "sessions": [{"session_id": "ses_x", "title": "", "workspace_path": "/w/hirex"}],
            "lines": [
                ln[2:].strip() if ln.startswith("- ") else ln
                for ln in raw_context.splitlines()
                if ln.strip()
            ],
        }
        captured: dict = {}

        def chat(messages):
            captured["user"] = messages[1]["content"]
            return (
                "- **Tracks:** engineering, design, management\n"
                "- **External links:** up to 6 per application"
            )

        formatter = AnswerFormatter(chat)
        out = formatter.format("What are the qualifly tracks?", bundle)

        # Evidence is passed raw (formatter's job to strip), so the prompt contains noise...
        self.assertIn("Assistant reported", captured["user"])
        # ...but the output must be clean.
        self.assertNotIn("Assistant reported", out)
        self.assertNotIn("Tool output", out)
        self.assertIn("engineering, design, management", out)


if __name__ == "__main__":
    unittest.main()
