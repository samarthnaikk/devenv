from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex
from core.tools.manage_memory import ManageMemoryTool


class ManageMemoryToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(dir="sample-test")
        self.addCleanup(self.tempdir.cleanup)
        self.memory = MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(),
            vector_index=InMemoryVectorIndex(),
        )
        self.memory.update_associative_tree(
            {
                "node_id": "proj_calendar",
                "label": "Calendar Project",
                "category": "project",
                "summary": "Calendar backend with reminders.",
                "edges": (),
            }
        )
        self.tool = ManageMemoryTool(self.memory)

    def test_update_mode_rewrites_summary(self) -> None:
        result = self.tool.execute(node_id="proj_calendar", mode="update", text="Updated calendar memory.")

        self.assertTrue(result.success)
        self.assertEqual(self.memory.store.get_node("proj_calendar").summary, "Updated calendar memory.")
        self.assertIn("sync_state", result.data)

    def test_prune_mode_deletes_node(self) -> None:
        result = self.tool.execute(node_id="proj_calendar", mode="prune")

        self.assertTrue(result.success)
        self.assertIsNone(self.memory.store.get_node("proj_calendar"))
        self.assertIn("sync_state", result.data)

    def test_update_mode_creates_manual_node_when_missing(self) -> None:
        result = self.tool.execute(node_id="manual_fact", mode="update", text="Manual fact for later recall.")

        self.assertTrue(result.success)
        node = self.memory.store.get_node("manual_fact")
        self.assertIsNotNone(node)
        self.assertEqual(node.category, "manual")
        self.assertEqual(node.label, "Manual Fact")
        self.assertEqual(node.summary, "Manual fact for later recall.")

    def test_update_mode_rejects_blank_text(self) -> None:
        result = self.tool.execute(node_id="proj_calendar", mode="update", text="   ")

        self.assertFalse(result.success)
        self.assertIn("non-empty text", result.output)

    def test_prune_mode_reports_missing_node_clearly(self) -> None:
        result = self.tool.execute(node_id="missing_fact", mode="prune")

        self.assertFalse(result.success)
        self.assertFalse(result.data["deleted"])
        self.assertEqual(result.output, "Memory node not found: missing_fact")
