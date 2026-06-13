from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex


class ConsolidationFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.engine = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )
        self.engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "Initial project summary.",
            }
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_consolidation_creates_and_updates_nodes(self) -> None:
        self.engine.add_episodic_log(
            "We introduced a Django auth component.",
            "I'll remember the backend shape.",
            node_id="proj_rxgpt",
            metadata={
                "project": "RxGPT",
                "memory_entities": [
                    {
                        "node_id": "cmp_django_auth",
                        "label": "Django Auth Setup",
                        "category": "component",
                        "summary": "Django auth relies on session cookies and middleware.",
                        "parent_id": "proj_rxgpt",
                    }
                ],
            },
        )

        result = self.engine.run_consolidation()

        self.assertEqual(result.processed_logs, 1)
        self.assertEqual(result.created_nodes, ("cmp_django_auth",))
        self.assertEqual(result.detected_project, "RxGPT")
        updated_project = self.engine.store.get_node("proj_rxgpt")
        created_component = self.engine.store.get_node("cmp_django_auth")
        self.assertIn("We introduced a Django auth component.", updated_project.summary)
        self.assertIsNotNone(created_component)

    def test_consolidation_watermark_prevents_reprocessing(self) -> None:
        self.engine.add_episodic_log(
            "Remember the component split.",
            "Noted.",
            metadata={
                "memory_entities": [
                    {
                        "label": "Component Split",
                        "category": "component",
                        "summary": "The frontend and backend were separated.",
                    }
                ]
            },
        )

        first = self.engine.run_consolidation()
        second = self.engine.run_consolidation()

        self.assertEqual(first.processed_logs, 1)
        self.assertEqual(second.processed_logs, 0)


if __name__ == "__main__":
    unittest.main()
