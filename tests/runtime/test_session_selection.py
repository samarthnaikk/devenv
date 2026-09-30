from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

from core.runtime.session_selection import (
    SessionCandidate,
    SessionSelectionOrchestrator,
    SelectionResult,
    _looks_like_selector_meta,
    apply_project_gate,
    apply_tag_gate,
    build_session_orchestrator,
    project_gate_mode,
    session_selector_enabled,
)


class TagGateTest(unittest.TestCase):
    def _candidate(self, session_id: str, *tags: str) -> SessionCandidate:
        return SessionCandidate(
            session_id=session_id,
            provider="opencode",
            title=session_id,
            workspace_path="/ws/app",
            score=5,
            updated_at="2026-01-01",
            tags=tags,
        )

    def test_filter_mode_drops_excluded_tags(self) -> None:
        candidates = [self._candidate("a", "derived"), self._candidate("b")]
        gated = apply_tag_gate(candidates, mode="filter", exclude={"derived"})
        self.assertEqual([c.session_id for c in gated], ["b"])

    def test_demote_mode_keeps_but_reorders(self) -> None:
        candidates = [self._candidate("a", "derived"), self._candidate("b")]
        gated = apply_tag_gate(candidates, mode="demote", exclude={"derived"})
        self.assertEqual([c.session_id for c in gated], ["b", "a"])

    def test_off_mode_is_noop(self) -> None:
        candidates = [self._candidate("a", "derived")]
        gated = apply_tag_gate(candidates, mode="off", exclude={"derived"})
        self.assertEqual([c.session_id for c in gated], ["a"])

    def test_no_excluded_tags_is_noop(self) -> None:
        candidates = [self._candidate("a", "derived")]
        gated = apply_tag_gate(candidates, mode="filter", exclude=set())
        self.assertEqual([c.session_id for c in gated], ["a"])


class SelectorMetaFilterTest(unittest.TestCase):
    def test_selector_json_is_meta_not_evidence(self) -> None:
        self.assertTrue(_looks_like_selector_meta('{"ordered": ["s1"], "selected": ["s1"]}'))
        self.assertTrue(_looks_like_selector_meta('"evidence": {"s1": ["x"]}'))
        self.assertTrue(_looks_like_selector_meta(""))

    def test_real_evidence_is_kept(self) -> None:
        self.assertFalse(_looks_like_selector_meta("The retrieval engine fuses lexical and semantic recall with RRF."))


@dataclass
class FakeSummary:
    session_id: str
    provider: str = "codex"
    title: str = ""
    workspace_path: str | None = None
    updated_at: str = "2026-01-01T00:00:00"
    preview: str = ""


class FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeContextBuilder:
    def __init__(self, provider_matches: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._provider_matches = provider_matches or {}
        self.build_calls: list[tuple[str, int]] = []
        self.runtime = ("CONTEXT", ("s1",), {"index_ready": True})
        self.workspace_path = "/ws/app"

    def _candidate_provider_names(self) -> list[str]:
        return list(self._provider_matches)

    def _select_runtime_matches_for_provider(self, task: str, *, provider_name: str):
        matches = self._provider_matches.get(provider_name, [])
        return matches, FakeProvider(provider_name), {"index_ready": True}

    def _context_lines_for_fused_matches(self, task: str, fused, *, max_lines: int = 6):
        return tuple(getattr(match.get("summary"), "title", "") for match, _provider in fused)

    def build_runtime_memory_context(self, task: str, *, provider_name=None, max_lines: int = 6):
        self.build_calls.append((task, max_lines))
        return self.runtime


def _match(session_id: str, *, score: int = 5, workspace: str | None = "/ws/app") -> dict[str, Any]:
    return {
        "summary": FakeSummary(
            session_id=session_id,
            workspace_path=workspace,
            title=f"title {session_id}",
            preview=f"preview {session_id}",
        ),
        "score": score,
    }


class CollectCandidatesTest(unittest.TestCase):
    def test_collects_fused_candidates_with_project(self) -> None:
        builder = FakeContextBuilder(
            {
                "codex": [_match("c1"), _match("shared")],
                "opencode": [_match("o1"), _match("shared", score=9)],
            }
        )
        orchestrator = SessionSelectionOrchestrator(builder, enabled=True)
        candidates = orchestrator.collect_candidates("query")

        ids = [candidate.session_id for candidate in candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("c1", ids)
        self.assertIn("o1", ids)
        self.assertIn("shared", ids)
        by_id = {candidate.session_id: candidate for candidate in candidates}
        self.assertEqual(by_id["c1"].workspace_path, "/ws/app")
        self.assertEqual(by_id["c1"].snippet, "preview c1")

    def test_candidate_snippet_prefers_matched_chunks(self) -> None:
        class Chunk:
            def __init__(self, text: str) -> None:
                self.text = text

        match = {
            "summary": FakeSummary("s1", preview="unrelated preview"),
            "chunks": [Chunk("the matched passage")],
            "score": 5,
        }
        candidate = SessionSelectionOrchestrator._to_candidate(match)
        self.assertIn("the matched passage", candidate.snippet)
        self.assertNotIn("unrelated preview", candidate.snippet)

    def test_candidate_snippet_falls_back_to_preview(self) -> None:
        match = {"summary": FakeSummary("s1", preview="only preview"), "score": 5}
        candidate = SessionSelectionOrchestrator._to_candidate(match)
        self.assertEqual(candidate.snippet, "only preview")

    def test_chunkless_candidate_snippet_uses_session_text(self) -> None:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class Provider:
            def _session_messages(self, _session_id):
                return [Message("the JSONB startup error was sa.JSONB() and the fix was sa.JSON()")]

        match = {"summary": FakeSummary("s1", preview="unrelated preview"), "score": 5}
        candidate = SessionSelectionOrchestrator._to_candidate(
            match, task="what caused the JSONB startup error and the fix?", provider=Provider()
        )
        self.assertIn("JSONB", candidate.snippet)
        self.assertNotIn("unrelated preview", candidate.snippet)

    def test_empty_when_no_providers(self) -> None:
        orchestrator = SessionSelectionOrchestrator(FakeContextBuilder(), enabled=True)
        self.assertEqual(orchestrator.collect_candidates("query"), [])

    def test_caps_candidates(self) -> None:
        builder = FakeContextBuilder(
            {"codex": [_match(f"s{i}") for i in range(20)]}
        )
        orchestrator = SessionSelectionOrchestrator(builder, enabled=True, max_candidates=5)
        self.assertEqual(len(orchestrator.collect_candidates("query")), 5)


class PassthroughTest(unittest.TestCase):
    def test_disabled_delegates_to_engine(self) -> None:
        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(builder, enabled=False)
        context, session_ids, metadata = orchestrator.select("query", max_lines=4)
        self.assertEqual((context, session_ids), ("CONTEXT", ("s1",)))
        self.assertEqual(metadata["index_ready"], True)
        # The formatter layer receives a structured, raw evidence bundle.
        self.assertEqual(metadata["retrieval_evidence"]["source"], "engine")
        self.assertEqual(metadata["retrieval_evidence"]["lines"], ["CONTEXT"])
        self.assertEqual(builder.build_calls, [("query", 4)])

    def test_enabled_without_selector_is_passthrough(self) -> None:
        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(builder, enabled=True, selector=None)
        self.assertFalse(orchestrator.active)
        orchestrator.select("query")
        self.assertEqual(len(builder.build_calls), 1)

    def test_enabled_with_selector_builds_context_from_selection(self) -> None:
        class StubSelector:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c1",), confidence=0.9, reason="ok")

        builder = FakeContextBuilder({"codex": [_match("c1"), _match("c2")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=StubSelector(), recall_floor="off"
        )
        self.assertTrue(orchestrator.active)
        context, session_ids, metadata = orchestrator.select("query", max_lines=4)
        self.assertEqual(session_ids, ("c1",))
        self.assertIn("title c1", context)
        self.assertTrue(metadata["selector_applied"])
        self.assertEqual(metadata["selector_confidence"], 0.9)
        self.assertEqual(builder.build_calls, [])

    def test_degraded_selector_falls_back_to_engine(self) -> None:
        class Degraded:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=(), reason="boom", degraded=True)

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Degraded()
        )
        context, _ids, _meta = orchestrator.select("query")
        self.assertEqual(context, "CONTEXT")
        self.assertEqual(len(builder.build_calls), 1)

    def test_abstaining_selector_returns_empty_context_when_floor_off(self) -> None:
        class Empty:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=(), reason="nothing relevant")

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Empty(), recall_floor="off"
        )
        context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(context, "")
        self.assertEqual(session_ids, ())
        self.assertEqual(metadata["context_match_state"], "new_context")
        self.assertTrue(metadata["selector_abstained"])

    def test_abstaining_selector_falls_back_to_engine_on_hard_floor(self) -> None:
        class Empty:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=(), reason="nothing relevant")

        builder = FakeContextBuilder({"codex": [_match("c1"), _match("c2")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Empty(), recall_floor="hard", recall_floor_k=2
        )
        context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(session_ids, ("c1", "c2"))
        self.assertTrue(metadata["selector_abstained_fallback"])
        self.assertIn("title c1", context)

    def test_selector_evidence_is_used_for_context(self) -> None:
        class Evidenced:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(
                    session_ids=("c1",),
                    reason="ok",
                    evidence={"c1": ["the exact proof line"]},
                )

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Evidenced()
        )
        context, session_ids, metadata = orchestrator.select("query", max_lines=4)
        self.assertIn("the exact proof line", context)
        self.assertEqual(session_ids, ("c1",))
        self.assertTrue(metadata["selector_evidence_used"])
        bundle = metadata["retrieval_evidence"]
        self.assertEqual(bundle["source"], "selector")
        self.assertEqual(bundle["sessions"][0]["session_id"], "c1")
        self.assertIn("the exact proof line", bundle["lines"])

    def test_coverage_evidence_reaches_context(self) -> None:
        class Covered:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(
                    session_ids=("c1",),
                    coverage=(
                        {"subquestion": "the regex fix", "session_id": "c1", "evidence": ["coverage proof line"]},
                    ),
                )

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Covered()
        )
        context, _ids, metadata = orchestrator.select("query", max_lines=6)
        self.assertIn("coverage proof line", context)
        self.assertEqual(metadata["selector_coverage_subquestions"], 1)

    def test_context_merges_selector_evidence_and_engine_lines(self) -> None:
        class Evidenced:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c1",), evidence={"c1": ["selector proof line"]})

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Evidenced()
        )
        context, _ids, metadata = orchestrator.select("query", max_lines=6)
        self.assertIn("selector proof line", context)
        self.assertIn("title c1", context)  # engine line is kept too
        self.assertTrue(metadata["selector_evidence_used"])
        self.assertTrue(metadata["selector_engine_lines_used"])


class TaskAwareContextBuilder(FakeContextBuilder):
    def __init__(self, task_matches: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__({"codex": []})
        self.task_matches = task_matches

    def _select_runtime_matches_for_provider(self, task: str, *, provider_name: str):
        matches: list[dict[str, Any]] = []
        for marker, marker_matches in self.task_matches.items():
            if marker.lower() in task.lower():
                matches = marker_matches
        return matches, FakeProvider(provider_name), {"index_ready": True}


class RequeryTest(unittest.TestCase):
    def test_requery_merges_new_candidates(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        class Requerying:
            def select(self, task, candidates, *, workspace_path):
                calls.append((task, [c.session_id for c in candidates]))
                if len(calls) == 1:
                    return SelectionResult(
                        session_ids=(),
                        need_more=True,
                        confidence=0.1,
                        refined_query="opencode server",
                    )
                return SelectionResult(session_ids=("n1",), confidence=0.9)

        builder = TaskAwareContextBuilder(
            {
                "base query": [_match("b1")],
                "opencode server": [_match("n1")],
            }
        )
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Requerying(), max_attempts=2, recall_floor="off"
        )
        context, session_ids, metadata = orchestrator.select("base query")

        self.assertEqual(len(calls), 2)
        self.assertIn("n1", calls[1][1])
        self.assertEqual(session_ids, ("n1",))
        self.assertEqual(metadata["selector_attempts"], 2)
        self.assertEqual(metadata["selector_final_query"], "opencode server app")
        self.assertIn("n1", context)

    def test_max_attempts_bounds_requeries(self) -> None:
        calls: list[str] = []

        class AlwaysNeedMore:
            def select(self, task, candidates, *, workspace_path):
                calls.append(task)
                return SelectionResult(session_ids=(), need_more=True, refined_query="more")

        builder = TaskAwareContextBuilder({"base": [_match("b1")], "more": [_match("n1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=AlwaysNeedMore(), max_attempts=1
        )
        orchestrator.select("base")
        self.assertEqual(len(calls), 1)

    def test_refine_query_appends_project(self) -> None:
        from core.runtime.session_selection import SelectionResult

        refined = SessionSelectionOrchestrator._refine_query(
            "base", SelectionResult(session_ids=(), refined_query="opencode server"), "/ws/facepred"
        )
        self.assertEqual(refined, "opencode server facepred")


class ProjectGateTest(unittest.TestCase):
    def _candidate(self, sid: str, workspace: str | None) -> SessionCandidate:
        return SessionCandidate(
            session_id=sid,
            provider="codex",
            title=f"title {sid}",
            workspace_path=workspace,
            score=5,
            updated_at="2026-01-01T00:00:00",
        )

    def test_off_is_noop(self) -> None:
        candidates = [self._candidate("a", "/ws/other"), self._candidate("b", "/ws/app")]
        gated = apply_project_gate(candidates, workspace_path="/ws/app", mode="off")
        self.assertEqual([c.session_id for c in gated], ["a", "b"])

    def test_demote_moves_same_project_first(self) -> None:
        candidates = [self._candidate("a", "/ws/other"), self._candidate("b", "/ws/app")]
        gated = apply_project_gate(candidates, workspace_path="/ws/app", mode="demote")
        self.assertEqual([c.session_id for c in gated], ["b", "a"])

    def test_filter_drops_foreign(self) -> None:
        candidates = [
            self._candidate("a", "/ws/other"),
            self._candidate("b", "/ws/app"),
            self._candidate("c", None),
        ]
        gated = apply_project_gate(candidates, workspace_path="/ws/app", mode="filter")
        self.assertEqual([c.session_id for c in gated], ["b", "c"])

    def test_query_naming_project_keeps_it(self) -> None:
        candidates = [self._candidate("a", "/ws/other")]
        gated = apply_project_gate(
            candidates,
            workspace_path="/ws/app",
            query="how did we fix the other bug",
            mode="filter",
        )
        self.assertEqual([c.session_id for c in gated], ["a"])

    def test_gate_mode_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(project_gate_mode(), "off")
        with mock.patch.dict(os.environ, {"DEVENV_SESSION_PROJECT_GATE": "demote"}, clear=True):
            self.assertEqual(project_gate_mode(), "demote")
        with mock.patch.dict(os.environ, {"DEVENV_SESSION_PROJECT_GATE": "bogus"}, clear=True):
            self.assertEqual(project_gate_mode(), "off")

    def test_orchestrator_applies_gate_when_enabled(self) -> None:
        builder = FakeContextBuilder(
            {"codex": [_match("f1", workspace="/ws/other"), _match("c1", workspace="/ws/app")]}
        )
        orchestrator = SessionSelectionOrchestrator(builder, enabled=True)
        with mock.patch.dict(os.environ, {"DEVENV_SESSION_PROJECT_GATE": "demote"}, clear=True):
            candidates = orchestrator.collect_candidates("query")
        self.assertEqual([c.session_id for c in candidates], ["c1", "f1"])


class DrillEvidenceTest(unittest.TestCase):
    def test_snippet_uses_up_to_five_chunks(self) -> None:
        class Chunk:
            def __init__(self, text: str) -> None:
                self.text = text

        match = {
            "summary": FakeSummary("s1"),
            "chunks": [Chunk(f"passage-{i}") for i in range(6)],
            "score": 5,
        }
        candidate = SessionSelectionOrchestrator._to_candidate(match)
        self.assertIn("passage-4", candidate.snippet)
        self.assertNotIn("passage-5", candidate.snippet)

    def test_drill_evidence_finds_query_line(self) -> None:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class Detail:
            def __init__(self, messages) -> None:
                self.messages = messages

        class Provider:
            def get_session(self, _session_id):
                return Detail(
                    [
                        Message("generic unrelated chatter about deployment"),
                        Message(
                            "Found it! Look at line 55: `...matQ5.` — Q4's answer key "
                            "and Q5 marker run together on the same line."
                        ),
                    ]
                )

        class Builder:
            workspace_path = "/ws/app"

        orchestrator = SessionSelectionOrchestrator(Builder(), enabled=True)
        match = {"summary": FakeSummary("s1"), "score": 5}
        lines = orchestrator._drill_evidence(
            "Why did the vaxo1a parser miss the matQ5 boundary and the regex fix?",
            [(match, Provider())],
            max_lines=4,
        )
        self.assertTrue(any("matQ5" in line for line in lines))

    def test_drill_uses_full_stream_over_detail_window(self) -> None:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class Provider:
            def _session_messages(self, _session_id):
                return [
                    Message("early unrelated setup chatter"),
                    Message("LINE 55: `...matQ5.` boundary bug where the marker runs together"),
                ]

            def get_session(self, _session_id):
                # The truncated window does NOT contain the answering line.
                return type("Detail", (), {"messages": (Message("only the tail"),)})()

        class Builder:
            workspace_path = "/ws/app"

        orchestrator = SessionSelectionOrchestrator(Builder(), enabled=True)
        match = {"summary": FakeSummary("s1"), "score": 5}
        lines = orchestrator._drill_evidence(
            "why did the parser miss the matQ5 boundary bug?",
            [(match, Provider())],
            max_lines=4,
        )
        self.assertTrue(any("matQ5" in line for line in lines))

    def test_drill_respects_char_budget(self) -> None:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class Provider:
            def _session_messages(self, _session_id):
                return [Message("x" * 5000), Message("matQ5 boundary bug detail here")]

        class Builder:
            workspace_path = "/ws/app"

        orchestrator = SessionSelectionOrchestrator(
            Builder(), enabled=True, drill_char_budget=100
        )
        match = {"summary": FakeSummary("s1"), "score": 5}
        lines = orchestrator._drill_evidence(
            "why did the parser miss the matQ5 boundary bug?",
            [(match, Provider())],
            max_lines=4,
        )
        self.assertEqual(lines, [])


class RecallFloorTest(unittest.TestCase):
    def test_hard_floor_keeps_engine_top_k(self) -> None:
        class PicksOther:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c3",), confidence=0.9)

        builder = FakeContextBuilder(
            {"codex": [_match("c1"), _match("c2"), _match("c3")]}
        )
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=PicksOther(), recall_floor="hard", recall_floor_k=2
        )
        _context, session_ids, metadata = orchestrator.select("query")
        self.assertIn("c3", session_ids)
        self.assertIn("c1", session_ids)
        self.assertIn("c2", session_ids)
        self.assertEqual(metadata["selector_floor_ids"], ["c1", "c2"])

    def test_floor_off_does_not_widen_selection(self) -> None:
        class PicksOther:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c3",), confidence=0.9)

        builder = FakeContextBuilder(
            {"codex": [_match("c1"), _match("c2"), _match("c3")]}
        )
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=PicksOther(), recall_floor="off"
        )
        _context, session_ids, _meta = orchestrator.select("query")
        self.assertEqual(session_ids, ("c3",))

    def test_soft_floor_skipped_on_high_confidence(self) -> None:
        class ConfidentOther:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c3",), confidence=0.9)

        builder = FakeContextBuilder(
            {"codex": [_match("c1"), _match("c2"), _match("c3")]}
        )
        orchestrator = SessionSelectionOrchestrator(
            builder,
            enabled=True,
            selector=ConfidentOther(),
            recall_floor="soft",
            recall_floor_k=2,
            min_confidence=0.4,
        )
        _context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(session_ids, ("c3",))
        self.assertEqual(metadata["selector_floor_ids"], [])

    def test_recall_floor_mode_default_hard(self) -> None:
        from core.runtime.session_selection import recall_floor_mode

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(recall_floor_mode(), "hard")
        with mock.patch.dict(os.environ, {"DEVENV_SESSION_SELECTOR_RECALL_FLOOR": "off"}, clear=True):
            self.assertEqual(recall_floor_mode(), "off")

    def test_default_selector_model_is_longcat(self) -> None:
        from core.runtime.session_selection import DEFAULT_SELECTOR_MODEL

        self.assertEqual(DEFAULT_SELECTOR_MODEL, "opencode-go/longcat-2.5-preview-free")
        builder = FakeContextBuilder({"codex": [_match("c1")]})
        ai = FakeChatAI('{"selected": ["c1"]}')
        with mock.patch.dict(
            os.environ, {"DEVENV_SESSION_SELECTOR": "1"}, clear=True
        ):
            orchestrator = build_session_orchestrator(builder, ai, "/ws/app")
        self.assertEqual(orchestrator.selector.model, DEFAULT_SELECTOR_MODEL)


    def test_drill_query_includes_coverage_subquestions(self) -> None:
        result = SelectionResult(
            session_ids=("c1",),
            refined_query="opencode server",
            coverage=(
                {"subquestion": "what was the threshold", "session_id": "c1", "evidence": []},
            ),
        )
        query = SessionSelectionOrchestrator._build_drill_query("original question", result)
        self.assertIn("original question", query)
        self.assertIn("opencode server", query)
        self.assertIn("what was the threshold", query)

    def test_drill_query_deduplicates(self) -> None:
        result = SelectionResult(session_ids=("c1",), refined_query="original question")
        query = SessionSelectionOrchestrator._build_drill_query("original question", result)
        self.assertEqual(query, "original question")


class TimeoutGuardTest(unittest.TestCase):
    def test_run_with_timeout_raises_on_overrun(self) -> None:
        import time as _time

        from core.runtime.session_selection import _run_with_timeout

        with self.assertRaises(TimeoutError):
            _run_with_timeout(lambda: _time.sleep(2), 0.1)

    def test_run_with_timeout_returns_fast_result(self) -> None:
        from core.runtime.session_selection import _run_with_timeout

        self.assertEqual(_run_with_timeout(lambda: 42, 5), 42)

    def test_run_with_timeout_disabled_is_direct(self) -> None:
        from core.runtime.session_selection import _run_with_timeout

        self.assertEqual(_run_with_timeout(lambda: "x", 0), "x")


class EnvFlagTest(unittest.TestCase):
    def test_env_flag_parsing(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"DEVENV_SESSION_SELECTOR": "on"}):
            self.assertTrue(session_selector_enabled())
        with mock.patch.dict(os.environ, {"DEVENV_SESSION_SELECTOR": "off"}):
            self.assertFalse(session_selector_enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(session_selector_enabled())


class FakeChatAI:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[Any] = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        return type("Response", (), {"content": self._content, "metadata": {}})()


class FactoryTest(unittest.TestCase):
    def test_disabled_without_flags(self) -> None:
        builder = FakeContextBuilder({"codex": [_match("c1")]})
        with mock.patch.dict(os.environ, {}, clear=True):
            orchestrator = build_session_orchestrator(builder, FakeChatAI("{}"), "/ws/app")
        self.assertFalse(orchestrator.uses_selector)

    def test_enabled_via_env_uses_selector(self) -> None:
        builder = FakeContextBuilder({"codex": [_match("c1")]})
        ai = FakeChatAI('{"selected": ["c1"], "confidence": 0.8}')
        with mock.patch.dict(
            os.environ,
            {"DEVENV_SESSION_SELECTOR": "1", "DEVENV_SESSION_SELECTOR_MODEL": "fake-model"},
            clear=True,
        ):
            orchestrator = build_session_orchestrator(builder, ai, "/ws/app")
            self.assertTrue(orchestrator.uses_selector)
            context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(session_ids, ("c1",))
        self.assertTrue(metadata["selector_applied"])
        self.assertIn("title c1", context)

    def test_structured_output_is_unwrapped(self) -> None:
        class StructuredAI:
            def __init__(self) -> None:
                self.seen_schema = None

            def chat(self, messages, output_schema=None):
                self.seen_schema = output_schema
                return type(
                    "Response",
                    (),
                    {"content": "", "metadata": {"structured": {"selected": ["c1"], "confidence": 0.9}}},
                )()

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        ai = StructuredAI()
        with mock.patch.dict(
            os.environ,
            {"DEVENV_SESSION_SELECTOR": "1", "DEVENV_SESSION_SELECTOR_MODEL": "fake-model"},
            clear=True,
        ):
            orchestrator = build_session_orchestrator(builder, ai, "/ws/app")
            _context, session_ids, _meta = orchestrator.select("query")

        self.assertIn("c1", session_ids)
        self.assertIsNotNone(ai.seen_schema)

    def test_shadow_mode_returns_engine_context(self) -> None:
        class Stub:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c1",), confidence=0.9)

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, selector=Stub(), enabled=False, shadow=True
        )
        self.assertTrue(orchestrator.uses_selector)
        context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(context, "CONTEXT")
        self.assertEqual(session_ids, ("s1",))
        self.assertTrue(metadata["selector_shadow"])
        self.assertEqual(metadata["selector_session_ids"], ["c1"])
        self.assertEqual(metadata["selector_engine_ids"], ["s1"])


if __name__ == "__main__":
    unittest.main()
