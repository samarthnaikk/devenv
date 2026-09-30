"""``devenv-plans`` command-line surface for saved plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.runtime.plan_store import PlanStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage saved Devenv plans.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace root.")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="List saved plans.")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="Show one plan.")
    show.add_argument("plan_id")
    show.add_argument("--json", action="store_true")

    export = sub.add_parser("export", help="Export a plan to markdown or JSON.")
    export.add_argument("plan_id")
    export.add_argument("--out", required=True)

    delete = sub.add_parser("delete", help="Delete a plan.")
    delete.add_argument("plan_id")

    args = parser.parse_args()
    store = PlanStore(str(Path(args.workspace).expanduser().resolve()))

    if args.command == "list":
        entries = store.list(limit=args.limit)
        if args.json:
            print(json.dumps([entry.to_dict() for entry in entries], indent=2))
        else:
            for entry in entries:
                print(f"{entry.plan_id}\t{entry.objective[:80]}")
        return 0

    if args.command == "show":
        entry = store.load(args.plan_id)
        if entry is None:
            print(f"plan not found: {args.plan_id}")
            return 1
        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(entry.raw_plan_markdown or "(no markdown body)")
        return 0

    if args.command == "export":
        written = store.export_to(args.plan_id, Path(args.out).expanduser())
        if written is None:
            print(f"plan not found: {args.plan_id}")
            return 1
        print(f"exported {args.plan_id} to {written}")
        return 0

    if args.command == "delete":
        print(f"deleted {args.plan_id}" if store.delete(args.plan_id) else f"plan not found: {args.plan_id}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
