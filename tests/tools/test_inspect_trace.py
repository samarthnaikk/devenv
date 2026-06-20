from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex
from core.tools.inspect_trace import InspectTraceTool


class InspectTraceToolTest(unittest.TestCase):
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
        self.memory.record_working_memory(
            messages=[{"role": "user", "content": "Tell me about reminders"}],
            active_state={"workspace_path": "sample-test/tool-fixtures"},
        )
        self.memory.retrieve_context("Tell me about reminders")
        self.tool = InspectTraceTool(self.memory)

    def test_last_retrieval_mode_returns_trace(self) -> None:
        result = self.tool.execute(mode="last_retrieval")

        self.assertTrue(result.success)
        self.assertIn("markdown_context", result.data["trace"])

    def test_node_history_mode_returns_node_details(self) -> None:
        result = self.tool.execute(mode="node_history", node_id="proj_calendar")

        self.assertTrue(result.success)
        self.assertEqual(result.data["node"]["node_id"], "proj_calendar")
        self.assertTrue(result.data["vector_present"])
