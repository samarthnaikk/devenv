from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex


class ManualControlTest(unittest.TestCase):
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
                "node_id": "cmp_django_auth",
                "label": "Django Auth Setup",
                "category": "component",
                "summary": "Django authentication uses middleware and sessions.",
            }
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_rewrite_strategy_preserves_node_but_replaces_summary(self) -> None:
        self.assertTrue(self.engine.forget_node("cmp_django_auth", strategy="rewrite"))

        node = self.engine.store.get_node("cmp_django_auth")

        self.assertIsNotNone(node)
        self.assertEqual(node.summary, "Memory intentionally cleared by user request.")
        self.assertIn("cmp_django_auth", self.engine.vector_index.records)

    def test_prune_strategy_removes_node_and_vector(self) -> None:
        self.assertTrue(self.engine.forget_node("cmp_django_auth", strategy="prune"))

        self.assertIsNone(self.engine.store.get_node("cmp_django_auth"))
        self.assertNotIn("cmp_django_auth", self.engine.vector_index.records)


if __name__ == "__main__":
    unittest.main()
