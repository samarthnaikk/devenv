from __future__ import annotations

import json
import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex


class MemoryEngineCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.engine = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_update_associative_tree_writes_store_and_vector_index(self) -> None:
        node_id = self.engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "React and Django product workspace.",
            }
        )

        stored = self.engine.store.get_node(node_id)

        self.assertEqual(node_id, "proj_rxgpt")
        self.assertIsNotNone(stored)
        self.assertIn(node_id, self.engine.vector_index.records)

    def test_add_episodic_log_persists_serialized_interaction(self) -> None:
        self.engine.update_associative_tree(
            {
                "node_id": "proj_rxgpt",
                "label": "Project: RxGPT",
                "category": "project",
                "summary": "React and Django product workspace.",
            }
        )

        log_id = self.engine.add_episodic_log(
            "Investigate auth bug",
            "I checked the Django middleware chain.",
            node_id="proj_rxgpt",
            metadata={"command": "pytest"},
        )

        logs = self.engine.store.list_logs_since(0.0)

        self.assertEqual(logs[0].log_id, log_id)
        payload = json.loads(logs[0].raw_interaction)
        self.assertEqual(payload["user"], "Investigate auth bug")
        self.assertEqual(payload["metadata"]["command"], "pytest")


if __name__ == "__main__":
    unittest.main()
