from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from core.runtime.session_selection import (
    SessionCandidate,
    SessionSelectionOrchestrator,
    SelectionResult,
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

    def _candidate_provider_names(self) -> list[str]:
        return list(self._provider_matches)

    def _select_runtime_matches_for_provider(self, task: str, *, provider_name: str):
        matches = self._provider_matches.get(provider_name, [])
        return matches, FakeProvider(provider_name), {"index_ready": True}

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

    def test_enabled_with_selector_currently_passthrough(self) -> None:
        class StubSelector:
            def select(self, task, candidates, *, workspace_path):
                return SelectionResult(session_ids=("c1",))

        builder = FakeContextBuilder({"codex": [_match("c1")]})
        orchestrator = SessionSelectionOrchestrator(
            builder, enabled=True, selector=StubSelector()
        )
        self.assertTrue(orchestrator.active)
        context, _ids, _meta = orchestrator.select("query")
        self.assertEqual(context, "CONTEXT")


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


if __name__ == "__main__":
    unittest.main()
