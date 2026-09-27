from __future__ import annotations

import json
import unittest

from core.runtime.session_selection import SessionCandidate
from core.runtime.session_selector_llm import (
    LLMSessionSelector,
    build_selector_messages,
    parse_selection,
    project_name,
)


def _candidate(session_id: str, *, workspace: str | None = "/ws/devenv") -> SessionCandidate:
    return SessionCandidate(
        session_id=session_id,
        provider="codex",
        title=f"title {session_id}",
        workspace_path=workspace,
        score=5,
        updated_at="2026-01-01T00:00:00",
        snippet=f"snippet {session_id}",
    )


class PromptTest(unittest.TestCase):
    def test_project_name(self) -> None:
        self.assertEqual(project_name("/Users/x/devenv"), "devenv")
        self.assertEqual(project_name(None), "unknown")
        self.assertEqual(project_name(""), "unknown")

    def test_messages_include_project_and_candidates(self) -> None:
        messages = build_selector_messages(
            "how did we invoke opencode?",
            [_candidate("c1"), _candidate("c2", workspace="/ws/facepred")],
            workspace_path="/ws/devenv",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CURRENT PROJECT", messages[0]["content"])
        user = messages[1]["content"]
        self.assertIn("CURRENT PROJECT: devenv", user)
        self.assertIn("USER QUERY: how did we invoke opencode?", user)
        self.assertIn("id=c1 | project=devenv", user)
        self.assertIn("id=c2 | project=facepred", user)
        self.assertIn("snippet c1", user)


class ParseSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [_candidate("c1"), _candidate("c2")]

    def test_parses_valid_json(self) -> None:
        raw = json.dumps(
            {
                "ordered": ["c1", "c2"],
                "selected": ["c1"],
                "confidence": 0.75,
                "need_more": True,
                "refined_query": "opencode server config",
                "evidence": {"c1": ["line one", "line two"]},
                "reasons": {"c1": "same project"},
            }
        )
        result = parse_selection(raw, self.candidates)
        self.assertEqual(result.session_ids, ("c1",))
        self.assertEqual(result.confidence, 0.75)
        self.assertTrue(result.need_more)
        self.assertEqual(result.refined_query, "opencode server config")
        self.assertEqual(result.evidence["c1"], ["line one", "line two"])
        self.assertIn("same project", result.reason)
        self.assertFalse(result.degraded)

    def test_strips_code_fences(self) -> None:
        raw = '```json\n{"selected": ["c2"]}\n```'
        result = parse_selection(raw, self.candidates)
        self.assertEqual(result.session_ids, ("c2",))

    def test_missing_selected_uses_ordered(self) -> None:
        result = parse_selection(json.dumps({"ordered": ["c2", "c1"]}), self.candidates)
        self.assertEqual(result.session_ids, ("c2", "c1"))

    def test_explicit_empty_selected_abstains(self) -> None:
        result = parse_selection(
            json.dumps({"ordered": ["c1"], "selected": []}), self.candidates
        )
        self.assertEqual(result.session_ids, ())

    def test_unknown_ids_filtered(self) -> None:
        result = parse_selection(
            json.dumps({"selected": ["c1", "nope", "c2"]}), self.candidates
        )
        self.assertEqual(result.session_ids, ("c1", "c2"))

    def test_invalid_json_degrades(self) -> None:
        result = parse_selection("not json at all", self.candidates)
        self.assertTrue(result.degraded)
        self.assertEqual(result.session_ids, ())

    def test_confidence_clamped(self) -> None:
        result = parse_selection(json.dumps({"selected": ["c1"], "confidence": 4}), self.candidates)
        self.assertEqual(result.confidence, 1.0)


class LLMSessionSelectorTest(unittest.TestCase):
    def test_select_uses_chat_output(self) -> None:
        def chat(_messages):
            return json.dumps({"selected": ["c2"], "confidence": 0.5})

        selector = LLMSessionSelector(chat, model="opencode/claude-haiku-4-5")
        result = selector.select(
            "query", [_candidate("c1"), _candidate("c2")], workspace_path="/ws/devenv"
        )
        self.assertEqual(result.session_ids, ("c2",))
        self.assertFalse(result.degraded)

    def test_select_degrades_on_chat_error(self) -> None:
        def chat(_messages):
            raise RuntimeError("model down")

        selector = LLMSessionSelector(chat)
        result = selector.select("query", [_candidate("c1")], workspace_path="/ws/devenv")
        self.assertTrue(result.degraded)
        self.assertEqual(result.session_ids, ())


if __name__ == "__main__":
    unittest.main()
