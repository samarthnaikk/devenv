from __future__ import annotations

import argparse
from pathlib import Path

from core.memory.cards import build_interaction_cards
from core.memory.embeddings import build_card_embedder
from core.memory.vector_index import LanceDBVectorIndex
from core.runtime.context_builder import ContextBuilderService, _default_provider_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest deterministic interaction cards from session archives.")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--provider", choices=("codex", "opencode", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="max sessions per provider (for testing)")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    service = ContextBuilderService(str(workspace), provider_configs=_default_provider_configs())
    store = service._get_session_embedding_store()
    if store is None:
        print("no memory store available")
        return 1
    embedder = build_card_embedder()
    vector_index = LanceDBVectorIndex(str(workspace / "vectors"), table_name="interaction_cards", dimension=embedder.dimension)
    print(f"embedder: {type(embedder).__name__}", flush=True)

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
            if index % 20 == 0 or index == len(summaries):
                print(f"{provider_name}: {index}/{len(summaries)} ({written} cards)", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
