from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .consolidation import ConsolidationService
from .embeddings import Embedder, SentenceTransformerEmbedder
from .extractors import ConsolidationExtractor
from .interface import MemoryEngineInterface
from .models import (
    ConsolidationResult,
    EpisodicLog,
    LogInteraction,
    MemoryNode,
    NodeEdge,
    NodeUpsertPayload,
    RetrievalResult,
    RetrievalTrace,
)
from .storage import SQLiteMemoryStore
from .retrieval import RetrievalService
from .vector_index import LanceDBVectorIndex, VectorIndex
from .working_memory import WorkingMemoryManager


class MemoryEngine(MemoryEngineInterface):
    def __init__(
        self,
        db_path: str = "memory.db",
        vector_dir: str = "vectors/",
        *,
        embedder: Embedder | None = None,
        vector_index: VectorIndex | None = None,
        store: SQLiteMemoryStore | None = None,
        extractor: ConsolidationExtractor | None = None,
    ):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_index = vector_index or LanceDBVectorIndex(vector_dir=vector_dir, dimension=self.embedder.dimension)
        self.store = store or SQLiteMemoryStore(db_path)
        self.working_memory = WorkingMemoryManager()
        self.retrieval_service = RetrievalService(
            store=self.store,
            vector_index=self.vector_index,
            embedder=self.embedder,
            working_memory=self.working_memory,
        )
        self.consolidation_service = ConsolidationService(
            store=self.store,
            vector_index=self.vector_index,
            embedder=self.embedder,
            extractor=extractor,
        )
        self._last_trace = RetrievalTrace()

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        timestamp = time.time()
        log_id = str(uuid.uuid4())
        interaction = LogInteraction(user=user_prompt, agent=agent_response, metadata=metadata or {})
        log = EpisodicLog(
            log_id=log_id,
            timestamp=timestamp,
            associated_node_id=node_id,
            raw_interaction=json.dumps(interaction.__dict__, sort_keys=True),
        )
        self.store.insert_log(log)
        self._index_episodic_log(log=log, interaction=interaction)
        return log_id

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> RetrievalResult:
        result = self.retrieval_service.retrieve(current_prompt=current_prompt, top_k=top_k)
        self._last_trace = result.trace
        return result

    def update_associative_tree(self, node_data: dict[str, Any]) -> str:
        now = time.time()
        payload = _coerce_payload(node_data)
        existing = self.store.get_node(payload.node_id)
        node = MemoryNode(
            node_id=payload.node_id,
            parent_id=payload.parent_id,
            label=payload.label,
            category=payload.category,
            summary=payload.summary,
            created_at=existing.created_at if existing else now,
            last_accessed=now,
            access_count=existing.access_count if existing else 0,
        )
        self.store.upsert_node(node)
        self.store.replace_node_edges(payload.node_id, list(payload.edges))
        self.vector_index.upsert(payload.node_id, payload.summary, self.embedder.embed(payload.summary))
        return payload.node_id

    def run_consolidation(self, since: float | None = None) -> ConsolidationResult:
        return self.consolidation_service.run(since=since)

    def forget_node(self, node_id: str, strategy: str = "prune") -> bool:
        existing = self.store.get_node(node_id)
        if existing is None:
            return False

        if strategy == "prune":
            self.vector_index.delete(node_id)
            return self.store.delete_node(node_id)

        if strategy == "rewrite":
            rewritten = MemoryNode(
                node_id=existing.node_id,
                parent_id=existing.parent_id,
                label=existing.label,
                category=existing.category,
                summary="Memory intentionally cleared by user request.",
                created_at=existing.created_at,
                last_accessed=time.time(),
                access_count=existing.access_count,
            )
            self.store.upsert_node(rewritten)
            self.vector_index.upsert(node_id, rewritten.summary, self.embedder.embed(rewritten.summary))
            return True

        raise ValueError(f"Unsupported forget strategy: {strategy}")

    def get_context_trace(self) -> RetrievalTrace:
        return self._last_trace

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        self.working_memory.record(messages=messages, active_state=active_state)

    def _index_episodic_log(self, *, log: EpisodicLog, interaction: LogInteraction) -> None:
        summary = " | ".join(piece for piece in (interaction.user.strip(), interaction.agent.strip()) if piece).strip()
        if not summary:
            summary = "Conversation turn with empty content."

        node = MemoryNode(
            node_id=f"episodic_{log.log_id}",
            parent_id=None,
            label=f"Episodic Memory {log.log_id[:8]}",
            category="episode",
            summary=summary,
            created_at=log.timestamp,
            last_accessed=log.timestamp,
            access_count=0,
        )
        self.store.upsert_node(node)
        self.vector_index.upsert(node.node_id, summary, self.embedder.embed(summary))


def _coerce_payload(node_data: dict[str, Any]) -> NodeUpsertPayload:
    edges: list[NodeEdge] = []
    for edge_data in node_data.get("edges", []):
        if isinstance(edge_data, NodeEdge):
            edges.append(edge_data)
            continue
        edges.append(
            NodeEdge(
                source_node_id=str(edge_data["source_node_id"]),
                target_node_id=str(edge_data["target_node_id"]),
                relationship_type=str(edge_data["relationship_type"]),
            )
        )

    return NodeUpsertPayload(
        node_id=str(node_data["node_id"]),
        parent_id=node_data.get("parent_id"),
        label=str(node_data["label"]),
        category=str(node_data["category"]),
        summary=str(node_data["summary"]),
        edges=tuple(edges),
    )
