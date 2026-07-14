from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import dataclass

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.models import NodeEdge, VectorMatch
from core.memory.vector_index import InMemoryVectorIndex


class RetrievalFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.engine = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )
        now = time.time()
        self.engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "RxGPT uses React, Tailwind, and Django.",
            }
        )
        self.engine.update_associative_tree(
            {
                "node_id": "cmp_django_auth",
                "parent_id": "proj_rxgpt",
                "label": "Django Auth Setup",
                "category": "component",
                "summary": "Django authentication uses custom middleware and session cookies.",
            }
        )
        self.engine.update_associative_tree(
            {
                "node_id": "pref_react_functional",
                "label": "React Preference",
                "category": "preference",
                "summary": "User prefers functional React components over class components.",
            }
        )
        self.engine.store.replace_node_edges(
            "cmp_django_auth",
            [
                NodeEdge("cmp_django_auth", "pref_react_functional", "related_preference")
            ],
        )
        self.engine.store.touch_nodes(["proj_rxgpt"], accessed_at=now - 5)
        self.engine.store.touch_nodes(["cmp_django_auth"], accessed_at=now - 1)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_hierarchical_inheritance_returns_child_and_parent_context(self) -> None:
        result = self.engine.retrieve_context("How do I fix my django authentication errors?", top_k=5)

        self.assertIn("Django Auth Setup", result.markdown_context)
        self.assertIn("Project: RxGPT", result.markdown_context)
        self.assertEqual(self.engine.get_context_trace().markdown_context, result.markdown_context)

    def test_preference_recall_returns_preference_context(self) -> None:
        result = self.engine.retrieve_context("Let's draft a new login component view.", top_k=5)

        self.assertIn("functional React components", result.markdown_context)
        candidate_relationships = {
            candidate.node.node_id: candidate.relationship for candidate in result.trace.expanded_candidates
        }
        self.assertIn(candidate_relationships["pref_react_functional"], {"seed", "related"})

    def test_graph_expansion_stays_at_depth_one_for_parent_context(self) -> None:
        self.engine.update_associative_tree(
            {
                "node_id": "org_root",
                "label": "Org Root",
                "category": "workspace",
                "summary": "Top-level organization memory.",
            }
        )
        self.engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "parent_id": "org_root",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "RxGPT uses React, Tailwind, and Django.",
            }
        )

        candidates = self.engine.retrieval_service._expand_candidates(
            [VectorMatch(node_id="cmp_django_auth", similarity=0.95, text_chunk="Django authentication uses custom middleware.")]
        )
        relationships = {candidate.node.node_id: candidate.relationship for candidate in candidates}
        self.assertEqual(relationships["proj_rxgpt"], "parent")
        self.assertNotIn("org_root", relationships)

    def test_empty_index_returns_working_memory_only(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        engine = MemoryEngine(
            db_path=f"{tempdir.name}/memory.db",
            vector_dir=f"{tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )
        engine.record_working_memory(
            messages=[{"role": "user", "content": "Working on the auth bug"}],
            active_state={"file": "core/memory/engine.py"},
        )

        result = engine.retrieve_context("No stored memory yet", top_k=3)

        self.assertIn("Working on the auth bug", result.markdown_context)
        self.assertNotIn("Retrieved Memory", result.markdown_context)

    def test_trace_contains_normalized_scores(self) -> None:
        result = self.engine.retrieve_context("How do I fix my django authentication errors?", top_k=5)

        self.assertTrue(result.trace.expanded_candidates)
        for candidate in result.trace.expanded_candidates:
            self.assertGreaterEqual(candidate.similarity, 0.0)
            self.assertLessEqual(candidate.similarity, 1.0)
            self.assertGreaterEqual(candidate.frequency_score, 0.0)
            self.assertLessEqual(candidate.frequency_score, 1.0)
            self.assertGreaterEqual(candidate.recency_score, 0.0)
            self.assertLessEqual(candidate.recency_score, 1.0)

    def test_vague_follow_up_uses_recent_working_memory_context(self) -> None:
        self.engine.add_episodic_log(
            "We were building a calendar project with a Python backend and React frontend.",
            "I'll remember that stack.",
            metadata={"workspace_path": self.tempdir.name},
        )
        self.engine.record_working_memory(
            messages=[
                {"role": "user", "content": "Do you know about the calendar project we were building?"},
                {"role": "assistant", "content": "Yes, tell me more about it."},
                {"role": "user", "content": "Not in the current directory, I mean the project we were working on earlier."},
            ],
            active_state={"workspace_path": self.tempdir.name},
        )

        result = self.engine.retrieve_context(
            "Not in the current directory, I mean the project we were working on earlier.",
            top_k=5,
        )

        self.assertIn("calendar project", result.markdown_context.lower())
        self.assertIn("python backend", result.markdown_context.lower())

    def test_compose_query_strips_unrelated_recent_context_on_topic_shift(self) -> None:
        self.engine.record_working_memory(
            messages=[
                {"role": "user", "content": "Let's plan the travel itinerary for Kyoto and Osaka next month."},
                {"role": "assistant", "content": "We should compare train routes and hotel areas."},
            ],
            active_state={"workspace_path": self.tempdir.name},
        )

        query = self.engine.retrieval_service._compose_query("How do I fix the django authentication middleware bug?")

        self.assertEqual(query, "How do I fix the django authentication middleware bug?")

    def test_compose_query_keeps_recent_context_for_referential_follow_up(self) -> None:
        self.engine.record_working_memory(
            messages=[
                {"role": "user", "content": "Do you remember the calendar project we were building?"},
                {"role": "assistant", "content": "Yes, it used a Python backend and React frontend."},
            ],
            active_state={"workspace_path": self.tempdir.name},
        )

        query = self.engine.retrieval_service._compose_query("Can you explain it again?")

        self.assertIn("Do you remember the calendar project we were building?", query)
        self.assertIn("Yes, it used a Python backend and React frontend.", query)

    def test_rehydrates_vector_index_from_stored_nodes_on_new_session(self) -> None:
        self.engine.add_episodic_log(
            "The calendar project used a React frontend and Python backend.",
            "Stored for future recall.",
            metadata={"workspace_path": self.tempdir.name},
        )

        reopened = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )

        result = reopened.retrieve_context("What was the calendar project backend?", top_k=5)

        self.assertIn("calendar project", result.markdown_context.lower())
        self.assertIn("python backend", result.markdown_context.lower())

    def test_hybrid_retrieval_can_return_fts_seed_when_vector_lookup_misses(self) -> None:
        if not getattr(self.engine.store, "_fts_enabled", False):
            self.skipTest("SQLite FTS5 is not available in this environment")

        class NoMatchVectorIndex(InMemoryVectorIndex):
            def query(self, vector: list[float], top_k: int, min_similarity: float):
                return []

        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        engine = MemoryEngine(
            db_path=f"{tempdir.name}/memory.db",
            vector_dir=f"{tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=NoMatchVectorIndex(),
        )
        engine.update_associative_tree(
            {
                "node_id": "path_server_py",
                "label": "server.py entrypoint",
                "category": "file",
                "summary": "The server.py entrypoint bootstraps the Flask backend runtime.",
            }
        )

        result = engine.retrieve_context("server.py entrypoint", top_k=3)

        self.assertIn("server.py entrypoint", result.markdown_context.lower())

    def test_high_confidence_seed_short_circuits_graph_expansion(self) -> None:
        class HighConfidenceVectorIndex(InMemoryVectorIndex):
            def query(self, vector: list[float], top_k: int, min_similarity: float):
                return [VectorMatch(node_id="cmp_django_auth", similarity=0.95, text_chunk="Django authentication uses middleware.")]

        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        engine = MemoryEngine(
            db_path=f"{tempdir.name}/memory.db",
            vector_dir=f"{tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=HighConfidenceVectorIndex(),
        )
        engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "RxGPT uses React, Tailwind, and Django.",
            }
        )
        engine.update_associative_tree(
            {
                "node_id": "cmp_django_auth",
                "parent_id": "proj_rxgpt",
                "label": "Django Auth Setup",
                "category": "component",
                "summary": "Django authentication uses custom middleware and session cookies.",
            }
        )

        result = engine.retrieve_context("How do I fix django auth?", top_k=5)

        self.assertTrue(result.trace.expanded_candidates)
        self.assertTrue(all(candidate.relationship == "seed" for candidate in result.trace.expanded_candidates))

    def test_skips_rehydration_when_vector_index_is_already_persisted(self) -> None:
        @dataclass
        class CountingEmbedder:
            dimension: int = 8
            calls: int = 0

            def embed(self, text: str) -> list[float]:
                self.calls += 1
                return [0.0] * self.dimension

        class PersistedVectorIndex(InMemoryVectorIndex):
            def has_persisted_state(self) -> bool:
                return True

        self.engine.add_episodic_log(
            "The calendar project used a React frontend and Python backend.",
            "Stored for future recall.",
            metadata={"workspace_path": self.tempdir.name},
        )

        embedder = CountingEmbedder()
        reopened = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=embedder,
            vector_index=PersistedVectorIndex(),
        )
        self.assertEqual(embedder.calls, 0)


if __name__ == "__main__":
    unittest.main()
