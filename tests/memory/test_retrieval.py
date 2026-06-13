from __future__ import annotations

import tempfile
import time
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.models import NodeEdge
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


if __name__ == "__main__":
    unittest.main()
