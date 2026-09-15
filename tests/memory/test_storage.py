from __future__ import annotations

import tempfile
import unittest

from core.memory.models import EpisodicLog, ExternalSessionEmbedding, MemoryNode, NodeEdge
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

    def test_external_session_embeddings_round_trip(self) -> None:
        record = ExternalSessionEmbedding(
            unified_session_id="codex:session-123",
            provider="codex",
            session_id="session-123",
            title="Session 123",
            workspace_path="/tmp/workspace",
            source_path="/tmp/.codex",
            updated_at="2026-08-20T10:00:00Z",
            content_hash="abc123",
            content_text="entire session text",
            embedding=(0.1, 0.2, 0.3),
            indexed_at=123.0,
        )

        self.store.upsert_external_session_embedding(record)

        self.assertEqual(
            self.store.get_external_session_embedding("codex:session-123"),
            record,
        )
        self.assertEqual(self.store.list_external_session_embeddings("codex"), [record])

    def test_fts_search_helpers_return_indexed_nodes_and_logs(self) -> None:
        if not getattr(self.store, "_fts_enabled", False):
            self.skipTest("SQLite FTS5 is not available in this environment")

        node = MemoryNode(
            node_id="auth_node",
            parent_id=None,
            label="Django Auth Setup",
            category="component",
            summary="Session cookies and middleware for django authentication.",
            created_at=1.0,
            last_accessed=1.0,
            access_count=0,
        )
        log = EpisodicLog(
            log_id="log-auth",
            timestamp=11.0,
            associated_node_id=None,
            raw_interaction='{"user": "Need django auth help", "agent": "Check the middleware chain."}',
        )

        self.store.upsert_node(node)
        self.store.insert_log(log)

        self.assertEqual(self.store.search_nodes_fts("django auth middleware", limit=2)[0].node_id, "auth_node")
        self.assertEqual(self.store.search_logs_fts("django auth help", limit=2)[0].log_id, "log-auth")

    def test_fts_query_normalization_keeps_repeated_project_terms_for_vague_follow_up(self) -> None:
        if not getattr(self.store, "_fts_enabled", False):
            self.skipTest("SQLite FTS5 is not available in this environment")

        self.store.upsert_node(
            MemoryNode(
                node_id="proj_calendar",
                parent_id=None,
                label="Project: Calendar",
                category="project",
                summary="Calendar project with a React frontend and Python backend.",
                created_at=1.0,
                last_accessed=1.0,
                access_count=0,
            )
        )
        self.store.upsert_node(
            MemoryNode(
                node_id="proj_jobs",
                parent_id=None,
                label="Project: Jobs",
                category="project",
                summary="Jobs project with a Django backend and React admin UI.",
                created_at=1.0,
                last_accessed=1.0,
                access_count=0,
            )
        )

        results = self.store.search_nodes_fts(
            "Not in the current directory, I mean the project we were working on earlier.\n"
            "Do you know about the calendar project we were building?",
            limit=2,
        )

        self.assertEqual(results[0].node_id, "proj_calendar")


if __name__ == "__main__":
    unittest.main()
