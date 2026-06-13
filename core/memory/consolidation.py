from __future__ import annotations

import time

from .embeddings import Embedder
from .extractors import ConsolidationExtractor, HeuristicConsolidationExtractor, slugify_label
from .models import ConsolidationResult, MemoryNode
from .storage import SQLiteMemoryStore
from .vector_index import VectorIndex

LAST_CONSOLIDATED_AT_KEY = "last_consolidated_at"


class ConsolidationService:
    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        embedder: Embedder,
        extractor: ConsolidationExtractor | None = None,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.embedder = embedder
        self.extractor = extractor or HeuristicConsolidationExtractor()

    def run(self, since: float | None = None) -> ConsolidationResult:
        lower_bound = since
        if lower_bound is None:
            lower_bound = float(self.store.get_state(LAST_CONSOLIDATED_AT_KEY) or 0.0)

        logs = self.store.list_logs_since(lower_bound)
        if not logs:
            return ConsolidationResult(processed_logs=0)

        existing_nodes = self.store.list_nodes()
        extraction = self.extractor.extract(logs, existing_nodes)
        created_nodes: list[str] = []
        updated_nodes: list[str] = []

        for entity in extraction.new_entities:
            node_id = entity.node_id or slugify_label(entity.label)
            now = time.time()
            node = MemoryNode(
                node_id=node_id,
                parent_id=entity.parent_id,
                label=entity.label,
                category=entity.category,
                summary=entity.summary,
                created_at=now,
                last_accessed=now,
                access_count=0,
            )
            self.store.upsert_node(node)
            self.vector_index.upsert(node_id, entity.summary, self.embedder.embed(entity.summary))
            created_nodes.append(node_id)

        for update in extraction.updates_to_existing_nodes:
            node = self.store.get_node(update.node_id)
            if node is None:
                continue
            combined_summary = f"{node.summary}\n{update.append_summary}".strip()
            rewritten = MemoryNode(
                node_id=node.node_id,
                parent_id=node.parent_id,
                label=node.label,
                category=node.category,
                summary=combined_summary,
                created_at=node.created_at,
                last_accessed=time.time(),
                access_count=node.access_count,
            )
            self.store.upsert_node(rewritten)
            self.vector_index.upsert(node.node_id, combined_summary, self.embedder.embed(combined_summary))
            updated_nodes.append(node.node_id)

        self.store.set_state(LAST_CONSOLIDATED_AT_KEY, str(max(log.timestamp for log in logs)))
        return ConsolidationResult(
            processed_logs=len(logs),
            created_nodes=tuple(created_nodes),
            updated_nodes=tuple(updated_nodes),
            detected_project=extraction.detected_project,
        )
