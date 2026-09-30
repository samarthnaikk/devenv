from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.memory.storage import SQLiteMemoryStore
from core.runtime.audit import (
    AuditRecorder,
    build_recorder,
    prune_audit_files,
    read_audit_file,
)


class AuditRecorderTest(unittest.TestCase):
    def test_records_to_file_and_db_with_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            recorder = build_recorder(str(workspace), store=store)

            recorder.record("turn.start", {"a": 1}, turn_id="t1")
            recorder.record("tool.call", {"tool_name": "read_file"}, turn_id="t1")
            recorder.record("turn.end", {"outcome": "success"}, turn_id="t1")

            events = list(reversed(store.list_runtime_events(limit=100)))
            ok, detail = AuditRecorder.verify_chain(events)

            files = list((workspace / ".devenv" / "audit").glob("audit-*.jsonl"))
            file_events = read_audit_file(files[0]) if files else []

        self.assertEqual(len(events), 3)
        self.assertTrue(ok, detail)
        self.assertEqual(len(file_events), 3)
        self.assertEqual(file_events[0]["prev_hash"], "")

    def test_tampering_breaks_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            recorder = build_recorder(str(workspace), store=store)
            recorder.record("turn.start", {"a": 1})
            recorder.record("turn.end", {"b": 2})

            events = list(reversed(store.list_runtime_events(limit=100)))
            events[0]["payload_json"] = '{"tampered":true}'
            ok, _detail = AuditRecorder.verify_chain(events)

        self.assertFalse(ok)

    def test_disabled_audit_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            with mock.patch.dict(os.environ, {"DEVENV_AUDIT": "0"}):
                recorder = build_recorder(str(workspace), store=store)
                result = recorder.record("turn.start", {"a": 1})
            events = store.list_runtime_events(limit=10)

        self.assertIsNone(result)
        self.assertEqual(events, [])

    def test_prune_audit_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            audit_dir = workspace / ".devenv" / "audit"
            audit_dir.mkdir(parents=True)
            stale = audit_dir / "audit-20200101.jsonl"
            stale.write_text("{}\n", encoding="utf-8")
            os.utime(stale, (1.0, 1.0))

            removed = prune_audit_files(str(workspace), retention_days=1)

        self.assertEqual(removed, 1)


class StorageRuntimeEventsTest(unittest.TestCase):
    def test_append_and_list_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(str(Path(tempdir) / "memory.db"))
            for index in range(3):
                store.append_runtime_event(
                    {
                        "event_id": f"e{index}",
                        "ts": 100.0 + index,
                        "turn_id": "t1",
                        "event_type": "tool.call",
                        "payload_json": json.dumps({"i": index}),
                        "prev_hash": f"h{index}",
                        "hash": f"h{index + 1}",
                    }
                )
            events = store.list_runtime_events(turn_id="t1", limit=10)
            removed = store.prune_runtime_events(before_ts=102.0)

        self.assertEqual(len(events), 3)
        self.assertEqual(removed, 2)

    def test_last_runtime_event_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(str(Path(tempdir) / "memory.db"))
            self.assertEqual(store.last_runtime_event_hash(), "")
            store.append_runtime_event({"event_id": "e1", "ts": 1.0, "event_type": "x", "hash": "abc"})
            self.assertEqual(store.last_runtime_event_hash(), "abc")


class AuditCliTest(unittest.TestCase):
    def test_query_and_verify_roundtrip(self) -> None:
        from core.runtime import audit_cli

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            store = SQLiteMemoryStore(str(workspace / "memory.db"))
            recorder = build_recorder(str(workspace), store=store)
            recorder.record("turn.start", {"a": 1}, turn_id="t1")

            with mock.patch("sys.argv", ["devenv-audit", str(workspace), "verify"]):
                code = audit_cli.main()

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
