from __future__ import annotations

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


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
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
        return cursor.rowcount > 0

    def insert_log(self, log: EpisodicLog) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction)
                VALUES (?, ?, ?, ?)
                """,
                (log.log_id, log.timestamp, log.associated_node_id, log.raw_interaction),
            )

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
