from __future__ import annotations

import tempfile
import unittest

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex
from core.memory.working_memory import WorkingMemoryManager


class WorkingMemoryManagerTest(unittest.TestCase):
    def test_record_keeps_bounded_recent_messages(self) -> None:
        manager = WorkingMemoryManager(max_messages=3)
        manager.record(
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            active_state={"file": "core/memory/engine.py"},
        )

        snapshot = manager.snapshot()

        self.assertEqual(len(snapshot.messages), 3)
        self.assertEqual(snapshot.messages[0].content, "two")
        self.assertEqual(snapshot.active_state["file"], "core/memory/engine.py")

    def test_engine_records_working_memory(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        engine = MemoryEngine(
            db_path=f"{tempdir.name}/memory.db",
            vector_dir=f"{tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
        )

        engine.record_working_memory(
            messages=[{"role": "user", "content": "Investigate auth flow"}],
            active_state={"shell_error": "ImportError"},
        )

        snapshot = engine.working_memory.snapshot()
        self.assertEqual(snapshot.messages[0].content, "Investigate auth flow")
        self.assertEqual(snapshot.active_state["shell_error"], "ImportError")


if __name__ == "__main__":
    unittest.main()
