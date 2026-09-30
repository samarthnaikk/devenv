from __future__ import annotations

import unittest
from unittest import mock

from core.runtime.answer_synthesis import (
    MultiSessionAnswer,
    group_evidence_by_session,
    max_sessions,
    multisession_enabled,
)


def _evidence() -> dict:
    return {
        "source": "selector",
        "sessions": [
            {"session_id": "s1", "title": "Alpha", "workspace_path": "/ws/a"},
            {"session_id": "s2", "title": "Beta", "workspace_path": "/ws/b"},
        ],
        "lines": ["l1", "l2", "l3"],
        "by_session": {"s1": ["a1", "a2"], "s2": ["b1"]},
    }


class GroupingTest(unittest.TestCase):
    def test_groups_by_session_when_present(self) -> None:
        grouped = group_evidence_by_session(_evidence())
        self.assertEqual(set(grouped), {"s1", "s2"})
        self.assertEqual(grouped["s1"]["lines"], ["a1", "a2"])
        self.assertEqual(grouped["s1"]["session"]["title"], "Alpha")

    def test_falls_back_to_first_session(self) -> None:
        grouped = group_evidence_by_session(
            {"sessions": [{"session_id": "solo"}], "lines": ["x", "y"]}
        )
        self.assertEqual(grouped["solo"]["lines"], ["x", "y"])

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(group_evidence_by_session({}), {})


class MultiSessionAnswerTest(unittest.TestCase):
    def test_synthesizes_per_session_then_reconciles(self) -> None:
        calls: list[list[dict[str, str]]] = []

        def chat(messages: list[dict[str, str]]) -> str:
            calls.append(messages)
            system = messages[0]["content"]
            if "MERGING" in system.upper() or "merge" in system.lower():
                return "**Fused answer** (source: s1)"
            # Per-session response keyed by the session header in the prompt.
            if "s1" in messages[1]["content"]:
                return "Answer from s1"
            return "Answer from s2"

        answerer = MultiSessionAnswer(chat, max_sessions=3)
        result = answerer.synthesize("what happened?", _evidence())

        self.assertIsNotNone(result)
        self.assertIn("Fused answer", result or "")
        # 2 per-session + 1 reconcile
        self.assertEqual(len(calls), 3)

    def test_single_session_gets_source_tag(self) -> None:
        def chat(messages: list[dict[str, str]]) -> str:
            return "Solo answer"

        answerer = MultiSessionAnswer(chat, max_sessions=3)
        result = answerer.synthesize(
            "q",
            {"sessions": [{"session_id": "only"}], "by_session": {"only": ["x"]}, "lines": ["x"]},
        )
        self.assertIn("Solo answer", result or "")
        self.assertIn("only", result or "")

    def test_no_relevant_evidence_returns_none(self) -> None:
        def chat(messages: list[dict[str, str]]) -> str:
            return "NO_RELEVANT_EVIDENCE"

        answerer = MultiSessionAnswer(chat, max_sessions=3)
        self.assertIsNone(answerer.synthesize("q", _evidence()))

    def test_chat_failure_returns_none(self) -> None:
        def chat(messages: list[dict[str, str]]) -> str:
            raise RuntimeError("down")

        answerer = MultiSessionAnswer(chat, max_sessions=3)
        self.assertIsNone(answerer.synthesize("q", _evidence()))

    def test_respects_max_sessions(self) -> None:
        seen: list[str] = []

        def chat(messages: list[dict[str, str]]) -> str:
            body = messages[1]["content"]
            seen.append(body)
            return "answer"

        evidence = {
            "sessions": [{"session_id": f"s{i}"} for i in range(5)],
            "by_session": {f"s{i}": ["line"] * (5 - i) for i in range(5)},
            "lines": ["line"],
        }
        MultiSessionAnswer(chat, max_sessions=2).synthesize("q", evidence)
        # 2 per-session + 1 reconcile
        self.assertEqual(len(seen), 3)


class EnvTest(unittest.TestCase):
    def test_enabled_by_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("DEVENV_MULTISESSION_ANSWERS", None)
            self.assertTrue(multisession_enabled())

    def test_disabled(self) -> None:
        with mock.patch.dict("os.environ", {"DEVENV_MULTISESSION_ANSWERS": "0"}):
            self.assertFalse(multisession_enabled())

    def test_max_sessions_default_and_override(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("DEVENV_MULTISESSION_MAX_SESSIONS", None)
            self.assertEqual(max_sessions(), 3)
        with mock.patch.dict("os.environ", {"DEVENV_MULTISESSION_MAX_SESSIONS": "5"}):
            self.assertEqual(max_sessions(), 5)


if __name__ == "__main__":
    unittest.main()
