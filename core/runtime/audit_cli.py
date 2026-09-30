"""``devenv-audit`` command-line surface for the durable audit trail."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.logging_utils import configure_logging
from core.memory.storage import SQLiteMemoryStore
from core.runtime.audit import (
    AuditRecorder,
    audit_file_reports,
    prune_audit_files,
    read_audit_file,
)
from core.runtime.state import resolve_memory_paths


def _open_store(workspace: Path, db_path: str) -> SQLiteMemoryStore:
    resolved_db, _vector_dir = resolve_memory_paths(db_path, "vectors", workspace_path=str(workspace))
    return SQLiteMemoryStore(resolved_db)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query and verify the Devenv runtime audit trail.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace root.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="List recent audit events.")
    query.add_argument("--type", dest="event_type", default=None)
    query.add_argument("--turn-id", default=None)
    query.add_argument("--limit", type=int, default=50)
    query.add_argument("--json", action="store_true")

    sub.add_parser("verify", help="Verify the audit hash chain.")

    export = sub.add_parser("export", help="Export the audit trail to a JSONL file.")
    export.add_argument("--out", default=None)

    prune = sub.add_parser("prune", help="Delete audit files older than N days.")
    prune.add_argument("--days", type=int, default=int(os.getenv("DEVENV_AUDIT_RETENTION_DAYS", "30")))

    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    configure_logging(args.log_level, workspace=str(workspace))
    store = _open_store(workspace, args.db_path)

    if args.command == "query":
        events = store.list_runtime_events(
            event_type=args.event_type, turn_id=args.turn_id, limit=args.limit
        )
        if args.json:
            print(json.dumps(events, indent=2))
        else:
            for event in reversed(events):
                print(
                    f"{event.get('event_type'):<20} turn={str(event.get('turn_id'))[:8]} "
                    f"backend={event.get('backend') or '-':<10} {event.get('payload_json')}"
                )
        return 0

    if args.command == "verify":
        events = list(reversed(store.list_runtime_events(limit=1_000_000)))
        ok, detail = AuditRecorder.verify_chain(events)
        print(f"chain: {'OK' if ok else 'BROKEN'} ({detail}); events={len(events)}")
        return 0 if ok else 1

    if args.command == "export":
        out_path = Path(args.out) if args.out else workspace / ".devenv" / "audit" / "audit-export.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        events = list(reversed(store.list_runtime_events(limit=1_000_000)))
        with out_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(f"exported {len(events)} event(s) to {out_path}")
        return 0

    if args.command == "prune":
        removed = prune_audit_files(str(workspace), retention_days=args.days)
        print(f"pruned {removed} audit file(s) older than {args.days} day(s)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
