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
