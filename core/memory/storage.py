from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import EpisodicLog, MemoryNode, NodeEdge


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
    "CREATE INDEX IF NOT EXISTS idx_memory_nodes_parent_id ON memory_nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodic_logs_timestamp ON episodic_logs(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_node_edges_target_id ON node_edges(target_node_id)",
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
)


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
                self._rebuild_fts(connection)

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
    tokens = [token.strip() for token in json.dumps(query).strip('"').replace("\\n", " ").split() if token.strip()]
    cleaned_tokens = [replaced for token in tokens if (replaced := "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", "."}))]
    if not cleaned_tokens:
        return ""
    return " ".join(f'"{token}"' for token in cleaned_tokens[:8])
