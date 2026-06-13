from __future__ import annotations

import time
from dataclasses import replace

from .embeddings import Embedder
from .models import MemoryNode, RetrievalCandidate, RetrievalResult, RetrievalSelectedNode, RetrievalTrace, VectorMatch
from .storage import SQLiteMemoryStore
from .vector_index import VectorIndex
from .working_memory import WorkingMemoryManager

PARENT_RELATIONSHIP_FACTOR = 0.92
SIBLING_RELATIONSHIP_FACTOR = 0.84
EDGE_RELATIONSHIP_FACTOR = 0.8


class RetrievalService:
    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        embedder: Embedder,
        working_memory: WorkingMemoryManager,
        similarity_threshold: float = 0.2,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.embedder = embedder
        self.working_memory = working_memory
        self.similarity_threshold = similarity_threshold

    def retrieve(self, current_prompt: str, top_k: int) -> RetrievalResult:
        query_vector = self.embedder.embed(current_prompt)
        matches = self.vector_index.query(query_vector, top_k=max(top_k, 5), min_similarity=self.similarity_threshold)
        if not matches:
            markdown = self._compile_markdown([], include_working_memory=True)
            trace = RetrievalTrace(markdown_context=markdown)
            return RetrievalResult(markdown_context=markdown, selected_nodes=(), trace=trace)

        candidates = self._expand_candidates(matches)
        scored = self._score_candidates(candidates)
        selected = tuple(
            RetrievalSelectedNode(
                node_id=candidate.node.node_id,
                label=candidate.node.label,
                category=candidate.node.category,
                summary=candidate.node.summary,
                score=candidate.final_score,
                relationship=candidate.relationship,
            )
            for candidate in scored[:top_k]
        )
        markdown = self._compile_markdown(selected, include_working_memory=True)
        trace = RetrievalTrace(
            matched_nodes=tuple(matches),
            expanded_candidates=tuple(scored),
            selected_nodes=selected,
            markdown_context=markdown,
        )
        self.store.touch_nodes([node.node_id for node in selected], accessed_at=time.time())
        return RetrievalResult(markdown_context=markdown, selected_nodes=selected, trace=trace)

    def _expand_candidates(self, matches: list[VectorMatch]) -> list[RetrievalCandidate]:
        expanded: dict[str, RetrievalCandidate] = {}
        for match in matches:
            node = self.store.get_node(match.node_id)
            if node is None:
                continue
            self._merge_candidate(
                expanded,
                RetrievalCandidate(node=node, source_node_id=node.node_id, relationship="seed", similarity=match.similarity),
            )
            for parent in self.store.get_parent_chain(node.node_id):
                if parent.node_id == node.node_id:
                    continue
                self._merge_candidate(
                    expanded,
                    RetrievalCandidate(
                        node=parent,
                        source_node_id=node.node_id,
                        relationship="parent",
                        similarity=match.similarity * PARENT_RELATIONSHIP_FACTOR,
                    ),
                )
            for sibling in self.store.get_sibling_nodes(node.node_id, node.parent_id):
                self._merge_candidate(
                    expanded,
                    RetrievalCandidate(
                        node=sibling,
                        source_node_id=node.node_id,
                        relationship="sibling",
                        similarity=match.similarity * SIBLING_RELATIONSHIP_FACTOR,
                    ),
                )
            for related in self.store.get_related_nodes(node.node_id):
                self._merge_candidate(
                    expanded,
                    RetrievalCandidate(
                        node=related,
                        source_node_id=node.node_id,
                        relationship="related",
                        similarity=match.similarity * EDGE_RELATIONSHIP_FACTOR,
                    ),
                )
        return list(expanded.values())

    def _merge_candidate(self, expanded: dict[str, RetrievalCandidate], candidate: RetrievalCandidate) -> None:
        existing = expanded.get(candidate.node.node_id)
        if existing is None or candidate.similarity > existing.similarity:
            expanded[candidate.node.node_id] = candidate

    def _score_candidates(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        similarity_values = [candidate.similarity for candidate in candidates]
        frequency_values = [float(candidate.node.access_count) for candidate in candidates]
        recency_values = [self._recency_raw(candidate.node) for candidate in candidates]

        similarity_scores = _normalize(similarity_values)
        frequency_scores = _normalize(frequency_values)
        recency_scores = _normalize(recency_values)

        scored: list[RetrievalCandidate] = []
        for index, candidate in enumerate(candidates):
            final_score = (similarity_scores[index] * 0.6) + (frequency_scores[index] * 0.2) + (recency_scores[index] * 0.2)
            scored.append(
                replace(
                    candidate,
                    similarity=similarity_scores[index],
                    frequency_score=frequency_scores[index],
                    recency_score=recency_scores[index],
                    final_score=final_score,
                )
            )
        scored.sort(key=lambda item: item.final_score, reverse=True)
        return scored

    def _compile_markdown(self, selected: list[RetrievalSelectedNode] | tuple[RetrievalSelectedNode, ...], include_working_memory: bool) -> str:
        lines: list[str] = []
        if include_working_memory:
            working_memory_block = self.working_memory.as_prompt_block()
            if working_memory_block.strip() != "## Working Memory":
                lines.append(working_memory_block)

        if selected:
            lines.append("## Retrieved Memory")
            for item in selected:
                lines.append(f"- [{item.category}] {item.label}: {item.summary}")

        return "\n".join(lines)

    def _recency_raw(self, node: MemoryNode) -> float:
        elapsed = max(time.time() - node.last_accessed, 1.0)
        return 1.0 / elapsed


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [1.0 for _ in values]

    return [(value - minimum) / (maximum - minimum) for value in values]
