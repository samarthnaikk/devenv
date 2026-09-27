from __future__ import annotations

import json
import random
import unittest

from core.runtime.session_selection import SessionCandidate
from core.runtime.session_selector_llm import (
    LLMSessionSelector,
    SELECTOR_SCHEMA,
    _chat_accepts_schema,
    _extract_json,
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

    def test_parses_coverage_and_filters_unknown_ids(self) -> None:
        raw = json.dumps(
            {
                "selected": ["c1"],
                "coverage": [
                    {"subquestion": "part a", "session_id": "c1", "evidence": ["line a"]},
                    {"subquestion": "part b", "session_id": "ghost", "evidence": ["x"]},
                ],
            }
        )
        result = parse_selection(raw, self.candidates)
        self.assertEqual(len(result.coverage), 2)
        self.assertEqual(result.coverage[0]["session_id"], "c1")
        self.assertEqual(result.coverage[0]["evidence"], ["line a"])
        self.assertEqual(result.coverage[1]["session_id"], "")

    def test_selector_schema_has_coverage(self) -> None:
        self.assertIn("coverage", SELECTOR_SCHEMA["properties"])


class ExtractJsonTest(unittest.TestCase):
    def test_ignores_stray_braces_in_prose(self) -> None:
        raw = (
            "Let me think. The format {a: b} is odd, but {not json}.\n"
            'Final: {"selected": ["s1"], "confidence": 0.5}'
        )
        payload = _extract_json(raw)
        self.assertEqual(payload, {"selected": ["s1"], "confidence": 0.5})

    def test_prefers_fenced_block(self) -> None:
        raw = (
            "Analysis with a brace { here.\n"
            "```json\n"
            '{"selected": ["s2"], "ordered": ["s2"]}\n'
            "```\n"
            "trailing notes"
        )
        payload = _extract_json(raw)
        self.assertEqual(payload, {"selected": ["s2"], "ordered": ["s2"]})

    def test_unwraps_final_content_wrapper(self) -> None:
        raw = '{"type": "final", "content": "{\\"selected\\": [\\"s1\\"]}"}'
        payload = _extract_json(raw)
        self.assertEqual(payload, {"selected": ["s1"]})

    def test_returns_none_when_no_json(self) -> None:
        self.assertIsNone(_extract_json("no json here at all"))

    def test_picks_object_with_selected_key(self) -> None:
        raw = '{"note": {"selected": "x"}} then {"selected": ["s1"]}'
        payload = _extract_json(raw)
        self.assertEqual(payload, {"selected": ["s1"]})


class RepairRetryTest(unittest.TestCase):
    def test_repairs_after_unparseable_reply(self) -> None:
        replies = ["I cannot produce JSON.", '{"selected": ["c1"], "confidence": 0.7}']

        def chat(_messages):
            return replies.pop(0)

        selector = LLMSessionSelector(chat, max_retries=1)
        result = selector.select("query", [_candidate("c1")], workspace_path="/ws/devenv")
        self.assertFalse(result.degraded)
        self.assertEqual(result.session_ids, ("c1",))
        self.assertEqual(replies, [])

    def test_stays_degraded_when_repair_also_fails(self) -> None:
        calls = {"n": 0}

        def chat(_messages):
            calls["n"] += 1
            return "still not json"

        selector = LLMSessionSelector(chat, max_retries=1)
        result = selector.select("query", [_candidate("c1")], workspace_path="/ws/devenv")
        self.assertTrue(result.degraded)
        self.assertEqual(calls["n"], 2)


class SchemaSupportTest(unittest.TestCase):
    def test_detects_schema_capable_callables(self) -> None:
        def one_arg(_messages):
            return "{}"

        def two_arg(_messages, _schema=None):
            return "{}"

        self.assertFalse(_chat_accepts_schema(one_arg))
        self.assertTrue(_chat_accepts_schema(two_arg))

    def test_selector_passes_schema(self) -> None:
        seen: dict = {}

        def chat(_messages, schema=None):
            seen["schema"] = schema
            return json.dumps({"selected": ["c1"]})

        selector = LLMSessionSelector(chat)
        result = selector.select("query", [_candidate("c1")], workspace_path="/ws/devenv")
        self.assertEqual(result.session_ids, ("c1",))
        self.assertEqual(seen["schema"], SELECTOR_SCHEMA)

    def test_one_arg_chat_still_works(self) -> None:
        def chat(_messages):
            return json.dumps({"selected": ["c2"]})

        selector = LLMSessionSelector(chat)
        result = selector.select("query", [_candidate("c2")], workspace_path="/ws/devenv")
        self.assertEqual(result.session_ids, ("c2",))


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


class PermutationTest(unittest.TestCase):
    def test_majority_selection_survives_shuffles(self) -> None:
        import re

        seen_first: list[str] = []

        def chat(messages):
            ids = re.findall(r"id=(\S+)", messages[1]["content"])
            seen_first.append(ids[0])
            return json.dumps({"selected": ["c2"], "ordered": ["c2", "c1", "c3"]})

        selector = LLMSessionSelector(chat, permutations=3, rng=random.Random(0))
        result = selector.select(
            "query",
            [_candidate("c1"), _candidate("c2"), _candidate("c3")],
            workspace_path="/ws/devenv",
        )
        self.assertEqual(result.session_ids, ("c2",))
        self.assertGreater(len(set(seen_first)), 1, "candidate order should vary")

    def test_minority_selection_is_dropped(self) -> None:
        import re

        def chat(messages):
            ids = re.findall(r"id=(\S+)", messages[1]["content"])
            # Always pick whichever candidate happens to be first (position bias).
            return json.dumps({"selected": [ids[0]], "ordered": ids})

        selector = LLMSessionSelector(chat, permutations=3, rng=random.Random(1))
        result = selector.select(
            "query",
            [_candidate("c1"), _candidate("c2"), _candidate("c3")],
            workspace_path="/ws/devenv",
        )
        # No candidate is picked by a majority, so nothing survives.
        self.assertEqual(result.session_ids, ())
        self.assertFalse(result.degraded)

    def test_degraded_when_all_permutations_fail(self) -> None:
        def chat(_messages):
            raise RuntimeError("down")

        selector = LLMSessionSelector(chat, permutations=2)
        result = selector.select("query", [_candidate("c1")], workspace_path="/ws/devenv")
        self.assertTrue(result.degraded)


if __name__ == "__main__":
    unittest.main()
