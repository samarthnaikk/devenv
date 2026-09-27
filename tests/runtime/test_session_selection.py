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
    apply_project_gate,
    build_session_orchestrator,
    project_gate_mode,
    session_selector_enabled,
)


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
        self.assertEqual(metadata, {"index_ready": True})
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
            builder, enabled=True, selector=StubSelector()
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

    def test_abstaining_selector_returns_empty_context(self) -> None:
        class Empty:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=(), reason="nothing relevant")

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=Empty()
        )
        context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(context, "")
        self.assertEqual(session_ids, ())
        self.assertEqual(metadata["context_match_state"], "new_context")
        self.assertTrue(metadata["selector_abstained"])

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
            builder, enabled=True, selector=Requerying(), max_attempts=2
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

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return type("Response", (), {"content": self._content})()


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
            {"DEVENV_SESSION_SELECTOR": "1", "DEVENV_SESSION_SELECTOR_MODEL": ""},
            clear=True,
        ):
            orchestrator = build_session_orchestrator(builder, ai, "/ws/app")
            self.assertTrue(orchestrator.uses_selector)
            context, session_ids, metadata = orchestrator.select("query")
        self.assertEqual(session_ids, ("c1",))
        self.assertTrue(metadata["selector_applied"])
        self.assertIn("title c1", context)

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
