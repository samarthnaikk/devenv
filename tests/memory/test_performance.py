from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.models import NodeEdge
from core.memory.vector_index import InMemoryVectorIndex


class RetrievalOrchestrationBoundaryTest(unittest.TestCase):
    def test_retrieval_handles_small_graph_without_exploding_candidate_count(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        engine = MemoryEngine(
            db_path=f"{tempdir.name}/memory.db",
            vector_dir=f"{tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )

        for index in range(8):
            engine.update_associative_tree(
                {
                    "node_id": f"node_{index}",
                    "parent_id": None if index == 0 else "node_0",
                    "label": f"Node {index}",
                    "category": "component",
                    "summary": f"Graph node {index} stores Django auth and React details {index}.",
                }
            )

        engine.store.replace_node_edges(
            "node_1",
            [
                NodeEdge("node_1", "node_2", "related"),
                NodeEdge("node_1", "node_3", "related"),
                NodeEdge("node_1", "node_4", "related"),
            ],
        )

        result = engine.retrieve_context("Need the Django auth details again", top_k=5)

        self.assertLessEqual(len(result.trace.expanded_candidates), 8)
        self.assertLessEqual(len(result.selected_nodes), 5)


if __name__ == "__main__":
    unittest.main()
