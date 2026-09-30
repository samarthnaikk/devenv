from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.runtime.plan_store import PlanEntry, PlanStore


class PlanStoreTest(unittest.TestCase):
    def test_save_list_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = PlanStore(tempdir)
            entry = store.save(
                objective="Add a dark mode toggle",
                raw_plan_markdown="# Plan\n\n- [ ] Inspect theme\n- [ ] Add toggle",
                tasks=[{"task_id": 1, "description": "Inspect theme"}],
                edges=[{"from": 1, "to": 2}],
            )
            entries = store.list()
            loaded = store.load(entry.plan_id)

        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIn("dark mode", loaded.objective)
        self.assertIn("Inspect theme", loaded.raw_plan_markdown)
        self.assertEqual(loaded.tasks[0]["task_id"], 1)

    def test_save_uses_slugged_chronological_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = PlanStore(tempdir)
            entry = store.save(objective="Fix the AUTH flow!", created_at=1000.0)
        self.assertTrue(entry.plan_id.startswith("1000-"))
        self.assertIn("fix-the-auth-flow", entry.plan_id)

    def test_collision_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = PlanStore(tempdir)
            first = store.save(objective="Same", created_at=5.0)
            second = store.save(objective="Same", created_at=5.0)
            entries = store.list()
        self.assertEqual(len(entries), 2)
        self.assertNotEqual(first.plan_id, second.plan_id)

    def test_delete_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = PlanStore(tempdir)
            entry = store.save(objective="Export me", raw_plan_markdown="# Body")
            destination = Path(tempdir) / "out.md"
            written = store.export_to(entry.plan_id, destination)
            exported_text = destination.read_text(encoding="utf-8")
            removed = store.delete(entry.plan_id)
            remaining = store.list()

        self.assertEqual(written, destination)
        self.assertEqual(exported_text, "# Body")
        self.assertTrue(removed)
        self.assertEqual(remaining, [])

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = PlanStore(tempdir)
            self.assertIsNone(store.load("nope"))
            self.assertEqual(store.list(), [])

    def test_entry_from_dict_defaults(self) -> None:
        entry = PlanEntry.from_dict({"plan_id": "p1", "objective": "x"})
        self.assertEqual(entry.plan_id, "p1")
        self.assertEqual(entry.tasks, ())


if __name__ == "__main__":
    unittest.main()
