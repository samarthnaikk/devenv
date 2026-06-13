from __future__ import annotations

import unittest

from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex


class InMemoryVectorIndexTest(unittest.TestCase):
    def test_query_returns_most_similar_nodes(self) -> None:
        embedder = HashingEmbedder(dimension=8)
        index = InMemoryVectorIndex()
        index.upsert("react_pref", "Functional React components", embedder.embed("functional react components"))
        index.upsert("django_auth", "Django auth setup", embedder.embed("django authentication backend"))

        matches = index.query(embedder.embed("react login component"), top_k=2, min_similarity=0.1)

        self.assertEqual(matches[0].node_id, "react_pref")
        self.assertGreaterEqual(matches[0].similarity, matches[1].similarity)


if __name__ == "__main__":
    unittest.main()
