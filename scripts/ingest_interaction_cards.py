from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from core.memory.cards import build_interaction_cards
from core.memory.embeddings import build_card_embedder
from core.memory.vector_index import LanceDBVectorIndex
from core.runtime.context_builder import ContextBuilderService, _default_provider_configs

META_TITLE_PATTERNS = (
    "Retrieval engine branch changes review",
    "Explore retrieval engine code",
    "Mine ",
    "sessions (@explore subagent)",
)


def load_excluded_session_ids(cutoff_iso: str, opencode_db: str) -> set[str]:
    excluded: set[str] = set()
    if not cutoff_iso or not os.path.exists(opencode_db):
        return excluded
    cutoff = datetime.fromisoformat(cutoff_iso).replace(tzinfo=timezone.utc)
    connection = sqlite3.connect(opencode_db)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute("select id, title, time_updated from session"):
            title = str(row["title"] or "")
            updated = datetime.fromtimestamp(int(row["time_updated"]) / 1000, tz=timezone.utc)
            if updated >= cutoff or any(pattern in title for pattern in META_TITLE_PATTERNS):
                excluded.add(str(row["id"]))
    finally:
        connection.close()
    return excluded


def purge_excluded(store, vector_index, excluded: set[str]) -> int:
    removed = 0
    for session_id in excluded:
        for card_id in store.delete_interaction_cards_for_session(session_id):
            try:
                vector_index.delete(card_id)
            except Exception:
                pass
            removed += 1
    return removed


def run_once(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    service = ContextBuilderService(str(workspace), provider_configs=_default_provider_configs())
    store = service._get_session_embedding_store()
    if store is None:
        print("no memory store available", flush=True)
        return 1
    embedder = build_card_embedder()
    vector_index = LanceDBVectorIndex(
        str(workspace / "vectors"), table_name="interaction_cards", dimension=embedder.dimension
    )
    print(f"embedder: {type(embedder).__name__}", flush=True)

    excluded: set[str] = set()
    if not args.no_exclude:
        excluded = load_excluded_session_ids(
            args.exclude_after, str(Path.home() / ".local" / "share" / "opencode" / "opencode.db")
        )
        removed = purge_excluded(store, vector_index, excluded)
        print(f"excluded {len(excluded)} meta session(s); purged {removed} card(s)", flush=True)

    providers = ["codex", "opencode"] if args.provider == "all" else [args.provider]
    for provider_name in providers:
        provider = service._get_provider(provider_name)
        try:
            summaries = provider.list_sessions()
        except Exception as exc:
            print(f"{provider_name}: list failed: {exc}", flush=True)
            continue
        if args.limit:
            summaries = summaries[: args.limit]
        print(f"{provider_name}: {len(summaries)} session(s)", flush=True)
        written = 0
        for index, summary in enumerate(summaries, start=1):
            if summary.session_id in excluded:
                continue
            state_key = f"card_ingest:{provider_name}:{summary.session_id}"
            if summary.updated_at and store.get_state(state_key) == summary.updated_at:
                continue
            try:
                chunks = provider.build_index_chunks(summary.session_id)
            except Exception:
                continue
            if not chunks:
                continue
            project = Path(summary.workspace_path).name if summary.workspace_path else ""
            cards = build_interaction_cards(
                provider=provider_name,
                session_id=summary.session_id,
                project=project,
                workspace_path=summary.workspace_path,
                messages=chunks,
                ts=summary.updated_at,
            )
            for card in cards:
                existing = store.get_interaction_card(card.card_id)
                if existing is not None and existing.content_hash == card.content_hash:
                    continue
                store.upsert_interaction_card(card)
                vector = embedder.embed(card.search_text)
                vector_index.upsert(card.card_id, card.search_text, list(vector))
                written += 1
            if summary.updated_at:
                store.set_state(state_key, summary.updated_at)
            if index % 20 == 0 or index == len(summaries):
                print(f"{provider_name}: {index}/{len(summaries)} ({written} cards)", flush=True)
    print("done", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest deterministic interaction cards from session archives.")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--provider", choices=("codex", "opencode", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="max sessions per provider (for testing)")
    parser.add_argument("--exclude-after", default="2026-09-15T00:00:00", help="skip sessions updated after this")
    parser.add_argument("--no-exclude", action="store_true", help="disable meta-session exclusion")
    parser.add_argument("--watch", action="store_true", help="poll the archives and card new/changed sessions")
    parser.add_argument("--interval", type=int, default=300, help="seconds between watch passes")
    parser.add_argument("--watch-iterations", type=int, default=0, help="stop after N watch passes (0 = forever)")
    args = parser.parse_args()

    if not args.watch:
        return run_once(args)

    iterations = 0
    while True:
        iterations += 1
        try:
            run_once(args)
        except Exception as exc:
            print(f"watch pass failed: {exc}", flush=True)
        if args.watch_iterations and iterations >= args.watch_iterations:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
