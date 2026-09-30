from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import (
    EpisodicLog,
    ExternalSessionChunkEmbedding,
    ExternalSessionEmbedding,
    InteractionCard,
    MemoryNode,
    NodeEdge,
)


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memory_nodes (
        node_id TEXT PRIMARY KEY,
        parent_id TEXT,
        label TEXT NOT NULL,
        category TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_accessed REAL NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (parent_id) REFERENCES memory_nodes(node_id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS node_edges (
        source_node_id TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        PRIMARY KEY (source_node_id, target_node_id),
        FOREIGN KEY (source_node_id) REFERENCES memory_nodes(node_id) ON DELETE CASCADE,
        FOREIGN KEY (target_node_id) REFERENCES memory_nodes(node_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS episodic_logs (
        log_id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        associated_node_id TEXT,
        raw_interaction TEXT NOT NULL,
        external_context_query TEXT,
        agent_response TEXT,
        FOREIGN KEY (associated_node_id) REFERENCES memory_nodes(node_id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS engine_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS external_session_embeddings (
        unified_session_id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        session_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        workspace_path TEXT,
        source_path TEXT,
        updated_at TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL,
        content_text TEXT NOT NULL DEFAULT '',
        embedding_json TEXT NOT NULL,
        indexed_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_nodes_parent_id ON memory_nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodic_logs_timestamp ON episodic_logs(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_node_edges_target_id ON node_edges(target_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_external_session_embeddings_provider ON external_session_embeddings(provider)",
    """
    CREATE TABLE IF NOT EXISTS external_session_chunk_embeddings (
        unified_session_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        session_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        role TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        indexed_at REAL NOT NULL,
        PRIMARY KEY (unified_session_id, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_external_session_chunk_embeddings_provider ON external_session_chunk_embeddings(provider)",
    """
    CREATE TABLE IF NOT EXISTS interaction_cards (
        card_id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        session_id TEXT NOT NULL,
        project TEXT NOT NULL DEFAULT '',
        workspace_path TEXT,
        turn_index INTEGER NOT NULL DEFAULT 0,
        intent_text TEXT NOT NULL DEFAULT '',
        answer_text TEXT NOT NULL DEFAULT '',
        ts TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL,
        search_text TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_interaction_cards_provider ON interaction_cards(provider)",
    "CREATE INDEX IF NOT EXISTS idx_interaction_cards_session_id ON interaction_cards(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_external_session_embeddings_session_id ON external_session_embeddings(session_id)",
    """
    CREATE TABLE IF NOT EXISTS session_tags (
        unified_session_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'user',
        created_at REAL NOT NULL,
        PRIMARY KEY (unified_session_id, tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_session_tags_tag ON session_tags(tag)",
    "CREATE INDEX IF NOT EXISTS idx_session_tags_source ON session_tags(source)",
    """
    CREATE TABLE IF NOT EXISTS runtime_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        ts REAL NOT NULL,
        turn_id TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL DEFAULT '',
        workspace TEXT NOT NULL DEFAULT '',
        event_type TEXT NOT NULL,
        backend TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        prev_hash TEXT NOT NULL DEFAULT '',
        hash TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_type ON runtime_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_turn_id ON runtime_events(turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_runtime_events_ts ON runtime_events(ts)",
)

FTS_SCHEMA_STATEMENTS = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_nodes_fts
    USING fts5(node_id, label, category, summary)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_logs_fts
    USING fts5(log_id, raw_interaction)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS interaction_cards_fts
    USING fts5(card_id, intent_text, answer_text, search_text)
    """,
)


FTS_SCHEMA_VERSION = "2"


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._fts_enabled = False
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self.transaction() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            try:
                for statement in FTS_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(episodic_logs)").fetchall()}
            if "external_context_query" not in columns:
                connection.execute("ALTER TABLE episodic_logs ADD COLUMN external_context_query TEXT")
            if "agent_response" not in columns:
                connection.execute("ALTER TABLE episodic_logs ADD COLUMN agent_response TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_logs_external_context_query ON episodic_logs(external_context_query)"
            )
            if self._fts_enabled:
                self._ensure_fts_current(connection)

    def _ensure_fts_current(self, connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT value FROM engine_state WHERE key = 'fts_schema_version'"
        ).fetchone()
        if row is not None and str(row["value"]) == FTS_SCHEMA_VERSION:
            return
        self._rebuild_fts(connection)
        connection.execute(
            """
            INSERT INTO engine_state (key, value)
            VALUES ('fts_schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (FTS_SCHEMA_VERSION,),
        )

    def upsert_external_session_embedding(self, record: ExternalSessionEmbedding) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO external_session_embeddings (
                    unified_session_id,
                    provider,
                    session_id,
                    title,
                    workspace_path,
                    source_path,
                    updated_at,
                    content_hash,
                    content_text,
                    embedding_json,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(unified_session_id) DO UPDATE SET
                    provider = excluded.provider,
                    session_id = excluded.session_id,
                    title = excluded.title,
                    workspace_path = excluded.workspace_path,
                    source_path = excluded.source_path,
                    updated_at = excluded.updated_at,
                    content_hash = excluded.content_hash,
                    content_text = excluded.content_text,
                    embedding_json = excluded.embedding_json,
                    indexed_at = excluded.indexed_at
                """,
                (
                    record.unified_session_id,
                    record.provider,
                    record.session_id,
                    record.title,
                    record.workspace_path,
                    record.source_path,
                    record.updated_at,
                    record.content_hash,
                    record.content_text,
                    json.dumps(list(record.embedding)),
                    record.indexed_at,
                ),
            )

    def get_external_session_embedding(self, unified_session_id: str) -> ExternalSessionEmbedding | None:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    unified_session_id,
                    provider,
                    session_id,
                    title,
                    workspace_path,
                    source_path,
                    updated_at,
                    content_hash,
                    content_text,
                    embedding_json,
                    indexed_at
                FROM external_session_embeddings
                WHERE unified_session_id = ?
                """,
                (unified_session_id,),
            ).fetchone()
        return _row_to_external_session_embedding(row) if row else None

    def list_external_session_embeddings(self, provider: str | None = None) -> list[ExternalSessionEmbedding]:
        query = """
            SELECT
                unified_session_id,
                provider,
                session_id,
                title,
                workspace_path,
                source_path,
                updated_at,
                content_hash,
                content_text,
                embedding_json,
                indexed_at
            FROM external_session_embeddings
        """
        params: tuple[str, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        query += " ORDER BY provider, session_id"
        with self.transaction() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_external_session_embedding(row) for row in rows]

    def list_external_session_embedding_vectors(
        self, provider: str | None = None
    ) -> list[ExternalSessionEmbedding]:
        """Lightweight projection returning only ``session_id`` + ``embedding``.

        Avoids transferring the (large) ``content_text`` column when only the
        vector is needed, e.g. semantic recall over session-level embeddings.
        """
        query = "SELECT unified_session_id, provider, session_id, embedding_json FROM external_session_embeddings"
        params: tuple[str, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        query += " ORDER BY provider, session_id"
        with self.transaction() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_external_session_embedding_vector(row) for row in rows]

    def add_session_tag(self, unified_session_id: str, tag: str, *, source: str = "user", created_at: float = 0.0) -> None:
        cleaned = str(tag or "").strip().lower()
        if not unified_session_id or not cleaned:
            return
        import time as _time

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO session_tags (unified_session_id, tag, source, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(unified_session_id, tag) DO UPDATE SET source = excluded.source
                """,
                (unified_session_id, cleaned, str(source or "user"), created_at or _time.time()),
            )

    def remove_session_tag(self, unified_session_id: str, tag: str) -> None:
        cleaned = str(tag or "").strip().lower()
        if not unified_session_id or not cleaned:
            return
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM session_tags WHERE unified_session_id = ? AND tag = ?",
                (unified_session_id, cleaned),
            )

    def set_session_tags(self, unified_session_id: str, tags: list[tuple[str, str]] | list[str]) -> None:
        """Replace all tags for a session. Accepts ``str`` or ``(tag, source)`` items.

        The ``source='auto'`` tags are retained: they are recomputed from the
        archive and are not user-editable.
        """

        normalized: list[tuple[str, str]] = []
        for item in tags:
            if isinstance(item, tuple):
                tag, source = item
            else:
                tag, source = item, "user"
            cleaned = str(tag or "").strip().lower()
            if cleaned:
                normalized.append((cleaned, str(source or "user")))
        import time as _time

        now = _time.time()
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM session_tags WHERE unified_session_id = ? AND source != 'auto'",
                (unified_session_id,),
            )
            connection.executemany(
                """
                INSERT INTO session_tags (unified_session_id, tag, source, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(unified_session_id, tag) DO UPDATE SET source = excluded.source
                """,
                [(unified_session_id, tag, source, now) for tag, source in normalized],
            )

    def list_session_tags(self, unified_session_id: str) -> list[tuple[str, str]]:
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT tag, source FROM session_tags WHERE unified_session_id = ? ORDER BY tag",
                (unified_session_id,),
            ).fetchall()
        return [(str(row["tag"]), str(row["source"])) for row in rows]

    def tags_for_sessions(self, unified_session_ids: list[str]) -> dict[str, list[str]]:
        """Return ``{unified_session_id: [tag, ...]}`` for the requested ids."""

        if not unified_session_ids:
            return {}
        result: dict[str, list[str]] = {sid: [] for sid in unified_session_ids}
        placeholders = ",".join("?" for _ in unified_session_ids)
        with self.transaction() as connection:
            rows = connection.execute(
                f"SELECT unified_session_id, tag FROM session_tags WHERE unified_session_id IN ({placeholders}) ORDER BY tag",
                tuple(unified_session_ids),
            ).fetchall()
        for row in rows:
            result.setdefault(str(row["unified_session_id"]), []).append(str(row["tag"]))
        return result

    def all_session_tags(self) -> dict[str, list[tuple[str, str]]]:
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT unified_session_id, tag, source FROM session_tags ORDER BY unified_session_id, tag"
            ).fetchall()
        result: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            result.setdefault(str(row["unified_session_id"]), []).append(
                (str(row["tag"]), str(row["source"]))
            )
        return result

    def delete_session_tags(self, unified_session_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM session_tags WHERE unified_session_id = ?",
                (unified_session_id,),
            )

    def append_runtime_event(self, event: dict) -> None:
        """Append one audit event row. ``event`` is a plain dict (see audit.py)."""

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO runtime_events (
                    event_id, ts, turn_id, session_id, workspace, event_type,
                    backend, model, payload_json, prev_hash, hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.get("event_id") or ""),
                    float(event.get("ts") or 0.0),
                    str(event.get("turn_id") or ""),
                    str(event.get("session_id") or ""),
                    str(event.get("workspace") or ""),
                    str(event.get("event_type") or ""),
                    str(event.get("backend") or ""),
                    str(event.get("model") or ""),
                    str(event.get("payload_json") or "{}"),
                    str(event.get("prev_hash") or ""),
                    str(event.get("hash") or ""),
                ),
            )

    def list_runtime_events(
        self,
        *,
        event_type: str | None = None,
        turn_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        query = "SELECT * FROM runtime_events"
        clauses: list[str] = []
        params: list[object] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if turn_id:
            clauses.append("turn_id = ?")
            params.append(turn_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq DESC LIMIT ?"
        params.append(int(limit))
        with self.transaction() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def last_runtime_event_hash(self) -> str:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT hash FROM runtime_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return str(row["hash"]) if row is not None else ""

    def prune_runtime_events(self, *, before_ts: float) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM runtime_events WHERE ts < ?", (float(before_ts),)
            )
            return int(cursor.rowcount or 0)

    def delete_external_session_embedding(self, unified_session_id: str) -> None:
        """Remove a whole-session embedding and its chunk rows."""

        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM external_session_embeddings WHERE unified_session_id = ?",
                (unified_session_id,),
            )
            connection.execute(
                "DELETE FROM external_session_chunk_embeddings WHERE unified_session_id = ?",
                (unified_session_id,),
            )

    def prune_external_session_embeddings(
        self,
        live_unified_ids: set[str],
        *,
        provider: str | None = None,
    ) -> int:
        """Delete embeddings whose session is no longer present.

        Returns the number of whole-session rows removed. Chunk rows for the
        same ids are removed too so the index cannot keep scoring deleted or
        archived sessions.
        """

        query = "SELECT unified_session_id FROM external_session_embeddings"
        params: tuple[str, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        with self.transaction() as connection:
            rows = connection.execute(query, params).fetchall()
            stale = [
                str(row["unified_session_id"])
                for row in rows
                if str(row["unified_session_id"]) not in live_unified_ids
            ]
            if not stale:
                return 0
            placeholders = ",".join("?" for _ in stale)
            connection.execute(
                f"DELETE FROM external_session_embeddings WHERE unified_session_id IN ({placeholders})",
                tuple(stale),
            )
            connection.execute(
                f"DELETE FROM external_session_chunk_embeddings WHERE unified_session_id IN ({placeholders})",
                tuple(stale),
            )
        return len(stale)

    def replace_external_session_chunk_embeddings(
        self,
        unified_session_id: str,
        records: list[ExternalSessionChunkEmbedding],
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM external_session_chunk_embeddings WHERE unified_session_id = ?",
                (unified_session_id,),
            )
            connection.executemany(
                """
                INSERT INTO external_session_chunk_embeddings (
                    unified_session_id,
                    provider,
                    session_id,
                    chunk_index,
                    role,
                    source,
                    text,
                    content_hash,
                    embedding_json,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.unified_session_id,
                        record.provider,
                        record.session_id,
                        record.chunk_index,
                        record.role,
                        record.source,
                        record.text,
                        record.content_hash,
                        json.dumps(list(record.embedding)),
                        record.indexed_at,
                    )
                    for record in records
                ],
            )

    def list_external_session_chunk_embeddings(
        self, provider: str | None = None
    ) -> list[ExternalSessionChunkEmbedding]:
        query = """
            SELECT
                unified_session_id,
                provider,
                session_id,
                chunk_index,
                role,
                source,
                text,
                content_hash,
                embedding_json,
                indexed_at
            FROM external_session_chunk_embeddings
        """
        params: tuple[str, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        query += " ORDER BY provider, session_id, chunk_index"
        with self.transaction() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_external_session_chunk_embedding(row) for row in rows]

    def upsert_interaction_card(self, card: InteractionCard) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO interaction_cards (
                    card_id, provider, session_id, project, workspace_path, turn_index,
                    intent_text, answer_text, ts, content_hash, search_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    provider = excluded.provider,
                    session_id = excluded.session_id,
                    project = excluded.project,
                    workspace_path = excluded.workspace_path,
                    turn_index = excluded.turn_index,
                    intent_text = excluded.intent_text,
                    answer_text = excluded.answer_text,
                    ts = excluded.ts,
                    content_hash = excluded.content_hash,
                    search_text = excluded.search_text
                """,
                (
                    card.card_id,
                    card.provider,
                    card.session_id,
                    card.project,
                    card.workspace_path,
                    card.turn_index,
                    card.intent_text,
                    card.answer_text,
                    card.ts,
                    card.content_hash,
                    card.search_text,
                ),
            )
            self._upsert_interaction_card_fts(connection, card)

    def get_interaction_card(self, card_id: str) -> InteractionCard | None:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT card_id, provider, session_id, project, workspace_path, turn_index,
                       intent_text, answer_text, ts, content_hash, search_text
                FROM interaction_cards WHERE card_id = ?
                """,
                (card_id,),
            ).fetchone()
        return _row_to_interaction_card(row) if row else None

    def list_interaction_cards(self, provider: str | None = None) -> list[InteractionCard]:
        query = (
            "SELECT card_id, provider, session_id, project, workspace_path, turn_index,"
            " intent_text, answer_text, ts, content_hash, search_text FROM interaction_cards"
        )
        params: tuple[str, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        query += " ORDER BY provider, session_id, turn_index"
        with self.transaction() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_interaction_card(row) for row in rows]

    def search_interaction_cards_fts(self, query: str, limit: int = 5) -> list[InteractionCard]:
        cleaned_query = _normalize_fts_query(query)
        if not cleaned_query or not self._fts_enabled:
            return []
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT c.card_id, c.provider, c.session_id, c.project, c.workspace_path, c.turn_index,
                       c.intent_text, c.answer_text, c.ts, c.content_hash, c.search_text
                FROM interaction_cards_fts fts
                JOIN interaction_cards c ON c.card_id = fts.card_id
                WHERE interaction_cards_fts MATCH ?
                ORDER BY bm25(interaction_cards_fts)
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
        return [_row_to_interaction_card(row) for row in rows]

    def delete_interaction_cards_for_session(self, session_id: str) -> list[str]:
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT card_id FROM interaction_cards WHERE session_id = ?", (session_id,)
            ).fetchall()
            card_ids = [str(row["card_id"]) for row in rows]
            connection.execute("DELETE FROM interaction_cards WHERE session_id = ?", (session_id,))
            for card_id in card_ids:
                connection.execute("DELETE FROM interaction_cards_fts WHERE card_id = ?", (card_id,))
        return card_ids

    def upsert_node(self, node: MemoryNode) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO memory_nodes (
                    node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    parent_id = excluded.parent_id,
                    label = excluded.label,
                    category = excluded.category,
                    summary = excluded.summary,
                    created_at = memory_nodes.created_at,
                    last_accessed = excluded.last_accessed,
                    access_count = excluded.access_count
                """,
                (
                    node.node_id,
                    node.parent_id,
                    node.label,
                    node.category,
                    node.summary,
                    node.created_at,
                    node.last_accessed,
                    node.access_count,
                ),
            )
            self._upsert_node_fts(connection, node)

    def get_node(self, node_id: str) -> MemoryNode | None:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                FROM memory_nodes
                WHERE node_id = ?
                """,
                (node_id,),
            ).fetchone()
        return _row_to_node(row) if row else None

    def list_nodes(self) -> list[MemoryNode]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                FROM memory_nodes
                ORDER BY node_id
                """
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def replace_node_edges(self, source_node_id: str, edges: list[NodeEdge]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM node_edges WHERE source_node_id = ?", (source_node_id,))
            for edge in edges:
                connection.execute(
                    """
                    INSERT INTO node_edges (source_node_id, target_node_id, relationship_type)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source_node_id, target_node_id) DO UPDATE SET
                        relationship_type = excluded.relationship_type
                    """,
                    (edge.source_node_id, edge.target_node_id, edge.relationship_type),
                )

    def list_edges_for_node(self, node_id: str) -> list[NodeEdge]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT source_node_id, target_node_id, relationship_type
                FROM node_edges
                WHERE source_node_id = ? OR target_node_id = ?
                ORDER BY source_node_id, target_node_id
                """,
                (node_id, node_id),
            ).fetchall()
        return [NodeEdge(**dict(row)) for row in rows]

    def delete_node(self, node_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM memory_nodes WHERE node_id = ?", (node_id,))
            self._delete_node_fts(connection, node_id)
        return cursor.rowcount > 0

    def insert_log(self, log: EpisodicLog) -> None:
        external_context_query = _extract_external_context_query(log.raw_interaction)
        agent_response = _extract_agent_response(log.raw_interaction)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction, external_context_query, agent_response)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (log.log_id, log.timestamp, log.associated_node_id, log.raw_interaction, external_context_query, agent_response),
            )
            self._upsert_log_fts(connection, log)

    def list_logs_since(self, since: float) -> list[EpisodicLog]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT log_id, timestamp, associated_node_id, raw_interaction
                FROM episodic_logs
                WHERE timestamp > ?
                ORDER BY timestamp ASC
                """,
                (since,),
            ).fetchall()
        return [EpisodicLog(**dict(row)) for row in rows]

    def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
        cleaned_terms = [term.strip().lower() for term in terms if term and term.strip()]
        if not cleaned_terms:
            return []

        score_sql = " + ".join("(CASE WHEN lower(raw_interaction) LIKE ? THEN 1 ELSE 0 END)" for _ in cleaned_terms)
        where_sql = " OR ".join("lower(raw_interaction) LIKE ?" for _ in cleaned_terms)
        score_params = [f"%{term}%" for term in cleaned_terms]
        where_params = [f"%{term}%" for term in cleaned_terms]

        with self.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT log_id, timestamp, associated_node_id, raw_interaction
                FROM episodic_logs
                WHERE {where_sql}
                ORDER BY ({score_sql}) DESC, timestamp DESC
                LIMIT ?
                """,
                (*where_params, *score_params, limit),
            ).fetchall()
        return [EpisodicLog(**dict(row)) for row in rows]

    def search_logs_fts(self, query: str, limit: int = 5) -> list[EpisodicLog]:
        cleaned_query = _normalize_fts_query(query)
        if not cleaned_query or not self._fts_enabled:
            return []

        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT el.log_id, el.timestamp, el.associated_node_id, el.raw_interaction
                FROM episodic_logs_fts fts
                JOIN episodic_logs el ON el.log_id = fts.log_id
                WHERE episodic_logs_fts MATCH ?
                ORDER BY bm25(episodic_logs_fts)
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
        return [EpisodicLog(**dict(row)) for row in rows]

    def search_logs_for_external_query(self, query: str, limit: int = 5) -> list[EpisodicLog]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT log_id, timestamp, associated_node_id, raw_interaction
                FROM episodic_logs
                WHERE external_context_query = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
            if not rows:
                escaped_query = cleaned_query.replace("\\", "\\\\").replace('"', '\\"')
                pattern = f'%\"external_context_query\": \"{escaped_query}\"%'
                rows = connection.execute(
                    """
                    SELECT log_id, timestamp, associated_node_id, raw_interaction
                    FROM episodic_logs
                    WHERE raw_interaction LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (pattern, limit),
                ).fetchall()
                if rows:
                    connection.executemany(
                        """
                        UPDATE episodic_logs
                        SET external_context_query = ?
                        WHERE log_id = ? AND external_context_query IS NULL
                        """,
                        [(cleaned_query, str(row["log_id"])) for row in rows],
                    )
        return [EpisodicLog(**dict(row)) for row in rows]

    def search_agent_responses_for_external_query(self, query: str, limit: int = 5) -> list[str]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        with self.transaction() as connection:
            direct_rows = connection.execute(
                """
                SELECT agent_response
                FROM episodic_logs
                WHERE external_context_query = ?
                  AND agent_response IS NOT NULL
                  AND agent_response != ''
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
            direct_responses = [str(row["agent_response"]).strip() for row in direct_rows if str(row["agent_response"]).strip()]
            if direct_responses:
                return direct_responses

            rows = connection.execute(
                """
                SELECT log_id, raw_interaction
                FROM episodic_logs
                WHERE external_context_query = ?
                  AND (agent_response IS NULL OR agent_response = '')
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
            responses: list[str] = []
            missing_backfills: list[tuple[str, str]] = []
            for row in rows:
                agent_response = _extract_agent_response(str(row["raw_interaction"]) or "") or ""
                if agent_response:
                    missing_backfills.append((agent_response, str(row["log_id"])))
                    responses.append(agent_response)
            if responses:
                if missing_backfills:
                    connection.executemany(
                        """
                        UPDATE episodic_logs
                        SET agent_response = ?
                        WHERE log_id = ? AND (agent_response IS NULL OR agent_response = '')
                        """,
                        missing_backfills,
                    )
                return responses

            escaped_query = cleaned_query.replace("\\", "\\\\").replace('"', '\\"')
            pattern = f'%\"external_context_query\": \"{escaped_query}\"%'
            fallback_rows = connection.execute(
                """
                SELECT log_id, raw_interaction
                FROM episodic_logs
                WHERE raw_interaction LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
            fallback_responses: list[str] = []
            backfills: list[tuple[str, str, str]] = []
            for row in fallback_rows:
                raw_interaction = str(row["raw_interaction"]) or ""
                agent_response = _extract_agent_response(raw_interaction) or ""
                if agent_response:
                    fallback_responses.append(agent_response)
                    backfills.append((cleaned_query, agent_response, str(row["log_id"])))
            if backfills:
                connection.executemany(
                    """
                    UPDATE episodic_logs
                    SET external_context_query = ?, agent_response = ?
                    WHERE log_id = ?
                    """,
                    backfills,
                )
            return fallback_responses

    def search_nodes(self, terms: list[str], limit: int = 5) -> list[MemoryNode]:
        cleaned_terms = [term.strip().lower() for term in terms if term and term.strip()]
        if not cleaned_terms:
            return []

        score_sql = " + ".join(
            "(CASE WHEN lower(label) LIKE ? OR lower(summary) LIKE ? THEN 1 ELSE 0 END)" for _ in cleaned_terms
        )
        where_sql = " OR ".join("(lower(label) LIKE ? OR lower(summary) LIKE ?)" for _ in cleaned_terms)
        score_params: list[str] = []
        where_params: list[str] = []
        for term in cleaned_terms:
            pattern = f"%{term}%"
            where_params.extend([pattern, pattern])
            score_params.extend([pattern, pattern])

        with self.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                FROM memory_nodes
                WHERE {where_sql}
                ORDER BY ({score_sql}) DESC, last_accessed DESC
                LIMIT ?
                """,
                (*where_params, *score_params, limit),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def search_nodes_fts(self, query: str, limit: int = 5) -> list[MemoryNode]:
        cleaned_query = _normalize_fts_query(query)
        if not cleaned_query or not self._fts_enabled:
            return []

        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT mn.node_id, mn.parent_id, mn.label, mn.category, mn.summary, mn.created_at, mn.last_accessed, mn.access_count
                FROM memory_nodes_fts fts
                JOIN memory_nodes mn ON mn.node_id = fts.node_id
                WHERE memory_nodes_fts MATCH ?
                ORDER BY bm25(memory_nodes_fts)
                LIMIT ?
                """,
                (cleaned_query, limit),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def get_state(self, key: str) -> str | None:
        with self.transaction() as connection:
            row = connection.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_state(self, key: str, value: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO engine_state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def touch_nodes(self, node_ids: list[str], accessed_at: float) -> None:
        with self.transaction() as connection:
            for node_id in node_ids:
                connection.execute(
                    """
                    UPDATE memory_nodes
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE node_id = ?
                    """,
                    (accessed_at, node_id),
                )

    def get_parent_chain(self, node_id: str) -> list[MemoryNode]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                WITH RECURSIVE parent_chain(node_id, parent_id, label, category, summary, created_at, last_accessed, access_count) AS (
                    SELECT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                    FROM memory_nodes
                    WHERE node_id = ?
                    UNION ALL
                    SELECT mn.node_id, mn.parent_id, mn.label, mn.category, mn.summary, mn.created_at, mn.last_accessed, mn.access_count
                    FROM memory_nodes mn
                    JOIN parent_chain pc ON mn.node_id = pc.parent_id
                )
                SELECT DISTINCT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                FROM parent_chain
                """,
                (node_id,),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def get_related_nodes(self, node_id: str) -> list[MemoryNode]:
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT mn.node_id, mn.parent_id, mn.label, mn.category, mn.summary, mn.created_at, mn.last_accessed, mn.access_count
                FROM memory_nodes mn
                JOIN node_edges ne
                    ON mn.node_id = ne.target_node_id
                    OR mn.node_id = ne.source_node_id
                WHERE (ne.source_node_id = ? OR ne.target_node_id = ?)
                    AND mn.node_id != ?
                """,
                (node_id, node_id, node_id),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def get_sibling_nodes(self, node_id: str, parent_id: str | None) -> list[MemoryNode]:
        if parent_id is None:
            return []

        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT node_id, parent_id, label, category, summary, created_at, last_accessed, access_count
                FROM memory_nodes
                WHERE parent_id = ? AND node_id != ?
                ORDER BY node_id
                """,
                (parent_id, node_id),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def _rebuild_fts(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM memory_nodes_fts")
        connection.execute("DELETE FROM episodic_logs_fts")
        node_rows = connection.execute(
            """
            SELECT node_id, label, category, summary
            FROM memory_nodes
            """
        ).fetchall()
        for row in node_rows:
            connection.execute(
                """
                INSERT INTO memory_nodes_fts (node_id, label, category, summary)
                VALUES (?, ?, ?, ?)
                """,
                (str(row["node_id"]), str(row["label"]), str(row["category"]), str(row["summary"])),
            )
        log_rows = connection.execute(
            """
            SELECT log_id, raw_interaction
            FROM episodic_logs
            """
        ).fetchall()
        for row in log_rows:
            connection.execute(
                """
                INSERT INTO episodic_logs_fts (log_id, raw_interaction)
                VALUES (?, ?)
                """,
                (str(row["log_id"]), str(row["raw_interaction"])),
            )
        connection.execute("DELETE FROM interaction_cards_fts")
        card_rows = connection.execute(
            """
            SELECT card_id, intent_text, answer_text, search_text
            FROM interaction_cards
            """
        ).fetchall()
        for row in card_rows:
            connection.execute(
                """
                INSERT INTO interaction_cards_fts (card_id, intent_text, answer_text, search_text)
                VALUES (?, ?, ?, ?)
                """,
                (str(row["card_id"]), str(row["intent_text"]), str(row["answer_text"]), str(row["search_text"])),
            )

    def _upsert_node_fts(self, connection: sqlite3.Connection, node: MemoryNode) -> None:
        if not self._fts_enabled:
            return
        connection.execute("DELETE FROM memory_nodes_fts WHERE node_id = ?", (node.node_id,))
        connection.execute(
            """
            INSERT INTO memory_nodes_fts (node_id, label, category, summary)
            VALUES (?, ?, ?, ?)
            """,
            (node.node_id, node.label, node.category, node.summary),
        )

    def _upsert_interaction_card_fts(self, connection: sqlite3.Connection, card: InteractionCard) -> None:
        if not self._fts_enabled:
            return
        connection.execute("DELETE FROM interaction_cards_fts WHERE card_id = ?", (card.card_id,))
        connection.execute(
            """
            INSERT INTO interaction_cards_fts (card_id, intent_text, answer_text, search_text)
            VALUES (?, ?, ?, ?)
            """,
            (card.card_id, card.intent_text, card.answer_text, card.search_text),
        )

    def _delete_node_fts(self, connection: sqlite3.Connection, node_id: str) -> None:
        if not self._fts_enabled:
            return
        connection.execute("DELETE FROM memory_nodes_fts WHERE node_id = ?", (node_id,))

    def _upsert_log_fts(self, connection: sqlite3.Connection, log: EpisodicLog) -> None:
        if not self._fts_enabled:
            return
        connection.execute("DELETE FROM episodic_logs_fts WHERE log_id = ?", (log.log_id,))
        connection.execute(
            """
            INSERT INTO episodic_logs_fts (log_id, raw_interaction)
            VALUES (?, ?)
            """,
            (log.log_id, log.raw_interaction),
        )


def _row_to_node(row: sqlite3.Row) -> MemoryNode:
    return MemoryNode(
        node_id=str(row["node_id"]),
        parent_id=row["parent_id"],
        label=str(row["label"]),
        category=str(row["category"]),
        summary=str(row["summary"]),
        created_at=float(row["created_at"]),
        last_accessed=float(row["last_accessed"]),
        access_count=int(row["access_count"]),
    )


def _row_to_interaction_card(row: sqlite3.Row) -> InteractionCard:
    return InteractionCard(
        card_id=str(row["card_id"]),
        provider=str(row["provider"]),
        session_id=str(row["session_id"]),
        project=str(row["project"] or ""),
        workspace_path=str(row["workspace_path"]) if row["workspace_path"] is not None else None,
        turn_index=int(row["turn_index"]),
        intent_text=str(row["intent_text"] or ""),
        answer_text=str(row["answer_text"] or ""),
        ts=str(row["ts"] or ""),
        content_hash=str(row["content_hash"]),
        search_text=str(row["search_text"] or ""),
    )


def _row_to_external_session_embedding(row: sqlite3.Row) -> ExternalSessionEmbedding:
    raw_embedding = row["embedding_json"]
    try:
        parsed = json.loads(str(raw_embedding))
    except (TypeError, json.JSONDecodeError):
        parsed = []
    embedding = tuple(float(value) for value in parsed if isinstance(value, (int, float)))
    return ExternalSessionEmbedding(
        unified_session_id=str(row["unified_session_id"]),
        provider=str(row["provider"]),
        session_id=str(row["session_id"]),
        title=str(row["title"] or ""),
        workspace_path=str(row["workspace_path"]) if row["workspace_path"] is not None else None,
        source_path=str(row["source_path"]) if row["source_path"] is not None else None,
        updated_at=str(row["updated_at"] or ""),
        content_hash=str(row["content_hash"]),
        content_text=str(row["content_text"] or ""),
        embedding=embedding,
        indexed_at=float(row["indexed_at"]),
    )


def _row_to_external_session_embedding_vector(row: sqlite3.Row) -> ExternalSessionEmbedding:
    try:
        parsed = json.loads(str(row["embedding_json"]))
    except (TypeError, json.JSONDecodeError):
        parsed = []
    embedding = tuple(float(value) for value in parsed if isinstance(value, (int, float)))
    return ExternalSessionEmbedding(
        unified_session_id=str(row["unified_session_id"]),
        provider=str(row["provider"]),
        session_id=str(row["session_id"]),
        content_hash="",
        embedding=embedding,
    )


def _row_to_external_session_chunk_embedding(row: sqlite3.Row) -> ExternalSessionChunkEmbedding:
    raw_embedding = row["embedding_json"]
    try:
        parsed = json.loads(str(raw_embedding))
    except (TypeError, json.JSONDecodeError):
        parsed = []
    embedding = tuple(float(value) for value in parsed if isinstance(value, (int, float)))
    return ExternalSessionChunkEmbedding(
        unified_session_id=str(row["unified_session_id"]),
        provider=str(row["provider"]),
        session_id=str(row["session_id"]),
        chunk_index=int(row["chunk_index"]),
        content_hash=str(row["content_hash"]),
        embedding=embedding,
        role=str(row["role"] or ""),
        source=str(row["source"] or ""),
        text=str(row["text"] or ""),
        indexed_at=float(row["indexed_at"]),
    )


def _extract_external_context_query(raw_interaction: str) -> str | None:
    try:
        payload = json.loads(raw_interaction)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    query = metadata.get("external_context_query")
    if not isinstance(query, str):
        return None
    cleaned = query.strip()
    return cleaned or None


def _extract_agent_response(raw_interaction: str) -> str | None:
    try:
        payload = json.loads(raw_interaction)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    agent = payload.get("agent")
    if not isinstance(agent, str):
        return None
    cleaned = agent.strip()
    return cleaned or None


def _normalize_fts_query(query: str) -> str:
    tokens = [token.strip().lower() for token in json.dumps(query).strip('"').replace("\\n", " ").split() if token.strip()]
    cleaned_tokens: list[str] = []
    for token in tokens:
        replaced = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", "."})
        if not replaced or len(replaced) <= 2:
            continue
        if replaced in {"the", "and", "for", "with", "what", "was", "were", "about", "again"}:
            continue
        cleaned_tokens.append(replaced)
    if not cleaned_tokens:
        return ""

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, token in enumerate(cleaned_tokens):
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, index)

    ranked = sorted(
        counts,
        key=lambda token: (-counts[token], -len(token), first_seen[token]),
    )
    return " OR ".join(f'"{token}"' for token in ranked[:8])
