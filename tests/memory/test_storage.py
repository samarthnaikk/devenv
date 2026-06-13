from __future__ import annotations

import tempfile
import unittest

from core.memory.models import EpisodicLog, MemoryNode, NodeEdge
from core.memory.storage import SQLiteMemoryStore


class SQLiteMemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(f"{self.tempdir.name}/memory.db")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_upsert_and_get_node(self) -> None:
        node = MemoryNode(
            node_id="proj_rxgpt",
            parent_id=None,
            label="Project: RxGPT",
            category="project",
            summary="Main project memory.",
            created_at=1.0,
            last_accessed=1.0,
            access_count=2,
        )

        self.store.upsert_node(node)
        stored = self.store.get_node("proj_rxgpt")

        self.assertEqual(stored, node)

    def test_replace_edges(self) -> None:
        source = MemoryNode(
            node_id="source",
            parent_id=None,
            label="Source",
            category="project",
            summary="Source node",
            created_at=1.0,
            last_accessed=1.0,
            access_count=0,
        )
        target = MemoryNode(
            node_id="target",
            parent_id=None,
            label="Target",
            category="tech",
            summary="Target node",
            created_at=1.0,
            last_accessed=1.0,
            access_count=0,
        )
        self.store.upsert_node(source)
        self.store.upsert_node(target)

        self.store.replace_node_edges("source", [NodeEdge("source", "target", "uses_tech")])
        edges = self.store.list_edges_for_node("source")

        self.assertEqual(edges, [NodeEdge("source", "target", "uses_tech")])

    def test_insert_log_and_state(self) -> None:
        log = EpisodicLog(
            log_id="log-1",
            timestamp=10.0,
            associated_node_id=None,
            raw_interaction='{"user": "hello", "agent": "world"}',
        )

        self.store.insert_log(log)
        self.store.set_state("last_consolidated_at", "10.0")

        self.assertEqual(self.store.list_logs_since(0.0), [log])
        self.assertEqual(self.store.get_state("last_consolidated_at"), "10.0")


if __name__ == "__main__":
    unittest.main()
