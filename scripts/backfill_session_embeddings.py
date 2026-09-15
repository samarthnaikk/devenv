from __future__ import annotations

import argparse
from pathlib import Path

from core.runtime.context_builder import ContextBuilderService
from core.runtime.models import ExternalSessionProviderConfig

CODEX_ROOT = str(Path.home() / ".codex")
OPENCODE_DB = str(Path.home() / ".local" / "share" / "opencode" / "opencode.db")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill whole-session and chunk-level embeddings for Codex and OpenCode archives."
    )
    parser.add_argument("--workspace", default=str(Path.cwd()), help="Workspace root where memory.db will be written.")
    parser.add_argument("--codex-root", default=CODEX_ROOT)
    parser.add_argument("--opencode-db", default=OPENCODE_DB)
    parser.add_argument("--provider", choices=("codex", "opencode", "all"), default="all")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    configs = (
        ExternalSessionProviderConfig(
            provider="codex",
            root_path=str(Path(args.codex_root).expanduser()),
            index_path="session_index.jsonl",
        ),
        ExternalSessionProviderConfig(
            provider="opencode",
            root_path=str(Path(args.opencode_db).expanduser()),
        ),
    )
    service = ContextBuilderService(str(workspace), provider_configs=configs)
    print(f"embedder: {type(service._session_embedder).__name__}", flush=True)

    providers = ["codex", "opencode"] if args.provider == "all" else [args.provider]
    for provider_name in providers:
        provider = service._get_provider(provider_name)
        try:
            summaries = provider.list_sessions()
        except Exception as exc:
            print(f"{provider_name}: failed to list sessions: {exc}", flush=True)
            continue
        print(f"{provider_name}: found {len(summaries)} session(s)", flush=True)
        for index, summary in enumerate(summaries, start=1):
            try:
                service._with_session_embedding(provider, summary)
            except Exception as exc:
                print(f"{provider_name}: failed on {summary.session_id}: {exc}", flush=True)
            if index % 20 == 0 or index == len(summaries):
                print(f"{provider_name}: {index}/{len(summaries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
