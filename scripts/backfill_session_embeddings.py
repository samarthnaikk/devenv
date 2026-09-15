from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from core.memory.embeddings import HashingEmbedder
from core.memory.models import ExternalSessionEmbedding
from core.memory.storage import SQLiteMemoryStore
from core.runtime.context_builder import (
    _extract_opencode_message_part,
    _extract_session_messages,
    _millis_to_iso,
    _normalize_whitespace,
    _safe_json_loads,
    _session_id_from_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill whole-session embeddings for Codex and OpenCode archives.")
    parser.add_argument("--workspace", default=str(Path.cwd()), help="Workspace root where memory.db will be written.")
    parser.add_argument("--codex-root", default=str(Path.home() / ".codex"))
    parser.add_argument("--opencode-db", default=str(Path.home() / ".local" / "share" / "opencode" / "opencode.db"))
    parser.add_argument("--provider", choices=("codex", "opencode", "all"), default="all")
    parser.add_argument("--dimension", type=int, default=384)
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    store = SQLiteMemoryStore(str(workspace / "memory.db"))
    embedder = HashingEmbedder(dimension=args.dimension)

    if args.provider in {"codex", "all"}:
        backfill_codex(store=store, embedder=embedder, codex_root=Path(args.codex_root).expanduser())
    if args.provider in {"opencode", "all"}:
        backfill_opencode(store=store, embedder=embedder, db_path=Path(args.opencode_db).expanduser())
    return 0


def backfill_codex(*, store: SQLiteMemoryStore, embedder: HashingEmbedder, codex_root: Path) -> None:
    session_files = sorted(codex_root.glob("sessions/**/*.jsonl"))
    print(f"codex: found {len(session_files)} session files", flush=True)
    for index, session_file in enumerate(session_files, start=1):
        session_id = _session_id_from_file(session_file)
        if not session_id:
            continue
        title = session_file.stem
        workspace_path = None
        updated_at = ""
        parts: list[str] = []
        for raw_line in session_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            timestamp = str(row.get("timestamp") or "")
            row_type = row.get("type")
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if row_type == "session_meta":
                inner = payload or {}
                title = str(inner.get("title") or inner.get("thread_name") or title)
                workspace_path = inner.get("cwd") or workspace_path
                updated_at = str(inner.get("timestamp") or timestamp or updated_at)
                continue
            for message in _extract_session_messages(row_type=row_type, payload=payload, timestamp=timestamp):
                text = _normalize_whitespace(message.content)
                if text:
                    parts.append(text)
        document = "\n".join(part for part in [title, workspace_path or "", *parts] if part).strip()
        upsert_embedding(
            store=store,
            embedder=embedder,
            provider="codex",
            session_id=session_id,
            title=title,
            workspace_path=workspace_path,
            source_path=str(session_file),
            updated_at=updated_at,
            document=document or title or session_id,
        )
        if index % 20 == 0 or index == len(session_files):
            print(f"codex: {index}/{len(session_files)}", flush=True)


def backfill_opencode(*, store: SQLiteMemoryStore, embedder: HashingEmbedder, db_path: Path) -> None:
    if not db_path.exists():
        print(f"opencode: database not found at {db_path}", flush=True)
        return
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        sessions = connection.execute(
            """
            select id, title, directory, time_updated
            from session
            where time_archived is null
            order by time_updated desc
            """
        ).fetchall()
        print(f"opencode: found {len(sessions)} sessions", flush=True)
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["id"])
            title = _normalize_whitespace(str(session["title"] or "")) or "Untitled session"
            workspace_path = str(session["directory"] or "") or None
            updated_at = _millis_to_iso(session["time_updated"])
            rows = connection.execute(
                """
                select m.id as message_id, m.data as message_data, p.data as part_data, p.time_created as part_time
                from message m
                left join part p on p.message_id = m.id
                where m.session_id = ?
                order by m.time_created asc, p.time_created asc, p.id asc
                """,
                (session_id,),
            ).fetchall()
            messages_by_id: dict[str, dict[str, object]] = {}
            parts: list[str] = []
            seen: set[tuple[str, str]] = set()
            for row in rows:
                message_id = str(row["message_id"])
                if message_id not in messages_by_id:
                    messages_by_id[message_id] = _safe_json_loads(str(row["message_data"] or ""))
                part_payload = _safe_json_loads(str(row["part_data"] or ""))
                if not part_payload:
                    continue
                parent = messages_by_id.get(message_id, {})
                role = str(parent.get("role") or "assistant")
                message = _extract_opencode_message_part(role=role, payload=part_payload, timestamp=_millis_to_iso(row["part_time"]))
                if message is None:
                    continue
                signature = (message.role, message.content)
                if signature in seen:
                    continue
                seen.add(signature)
                text = _normalize_whitespace(message.content)
                if text:
                    parts.append(text)
            document = "\n".join(part for part in [title, workspace_path or "", *parts] if part).strip()
            upsert_embedding(
                store=store,
                embedder=embedder,
                provider="opencode",
                session_id=session_id,
                title=title,
                workspace_path=workspace_path,
                source_path=str(db_path),
                updated_at=updated_at,
                document=document or title or session_id,
            )
            if index % 20 == 0 or index == len(sessions):
                print(f"opencode: {index}/{len(sessions)}", flush=True)
    finally:
        connection.close()


def upsert_embedding(
    *,
    store: SQLiteMemoryStore,
    embedder: HashingEmbedder,
    provider: str,
    session_id: str,
    title: str,
    workspace_path: str | None,
    source_path: str | None,
    updated_at: str,
    document: str,
) -> None:
    unified_session_id = f"{provider}:{session_id}"
    content_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
    existing = store.get_external_session_embedding(unified_session_id)
    if existing is not None and existing.content_hash == content_hash:
        return
    embedding = tuple(float(value) for value in embedder.embed(document))
    store.upsert_external_session_embedding(
        ExternalSessionEmbedding(
            unified_session_id=unified_session_id,
            provider=provider,
            session_id=session_id,
            title=title,
            workspace_path=workspace_path,
            source_path=source_path,
            updated_at=updated_at,
            content_hash=content_hash,
            content_text=document,
            embedding=embedding,
            indexed_at=time.time(),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
