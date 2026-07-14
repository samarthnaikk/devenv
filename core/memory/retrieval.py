from __future__ import annotations

import re
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
MIN_CONTEXT_DRIFT_JACCARD = 0.12
RRF_K = 60
MAX_SIBLING_CANDIDATES = 3
MAX_RELATED_CANDIDATES = 3
REFERENTIAL_CONTEXT_MARKERS = {
    "again",
    "earlier",
    "it",
    "that",
    "them",
    "those",
    "this",
}


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
        query_text = self._compose_query(current_prompt)
        query_vector = self.embedder.embed(query_text)
        vector_matches = self.vector_index.query(query_vector, top_k=max(top_k, 5), min_similarity=self.similarity_threshold)
        lexical_matches = self._lexical_seed_matches(query_text, top_k=max(top_k, 5))
        matches = self._fuse_seed_matches(vector_matches, lexical_matches, top_k=max(top_k, 5))
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
            parent = self.store.get_node(node.parent_id) if node.parent_id else None
            if parent is not None:
                self._merge_candidate(
                    expanded,
                    RetrievalCandidate(
                        node=parent,
                        source_node_id=node.node_id,
                        relationship="parent",
                        similarity=match.similarity * PARENT_RELATIONSHIP_FACTOR,
                    ),
                )
            for sibling in self._top_structural_neighbors(
                self.store.get_sibling_nodes(node.node_id, node.parent_id),
                limit=MAX_SIBLING_CANDIDATES,
            ):
                self._merge_candidate(
                    expanded,
                    RetrievalCandidate(
                        node=sibling,
                        source_node_id=node.node_id,
                        relationship="sibling",
                        similarity=match.similarity * SIBLING_RELATIONSHIP_FACTOR,
                    ),
                )
            for related in self._top_structural_neighbors(
                self.store.get_related_nodes(node.node_id),
                limit=MAX_RELATED_CANDIDATES,
            ):
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

    def _top_structural_neighbors(self, nodes: list[MemoryNode], *, limit: int) -> list[MemoryNode]:
        ranked = sorted(
            nodes,
            key=lambda node: (node.access_count, node.last_accessed, node.created_at),
            reverse=True,
        )
        return ranked[:limit]

    def _lexical_seed_matches(self, query_text: str, top_k: int) -> list[VectorMatch]:
        if not hasattr(self.store, "search_nodes_fts"):
            return []
        matches: list[VectorMatch] = []
        for rank, node in enumerate(self.store.search_nodes_fts(query_text, limit=top_k), start=1):
            matches.append(
                VectorMatch(
                    node_id=node.node_id,
                    similarity=1.0 / (RRF_K + rank),
                    text_chunk=node.summary,
                )
            )
        return matches

    def _fuse_seed_matches(
        self,
        vector_matches: list[VectorMatch],
        lexical_matches: list[VectorMatch],
        top_k: int,
    ) -> list[VectorMatch]:
        if not vector_matches and not lexical_matches:
            return []

        fused_scores: dict[str, float] = {}
        text_chunks: dict[str, str] = {}

        for rank, match in enumerate(vector_matches, start=1):
            fused_scores[match.node_id] = fused_scores.get(match.node_id, 0.0) + (1.0 / (RRF_K + rank))
            text_chunks.setdefault(match.node_id, match.text_chunk)
        for rank, match in enumerate(lexical_matches, start=1):
            fused_scores[match.node_id] = fused_scores.get(match.node_id, 0.0) + (1.0 / (RRF_K + rank))
            text_chunks.setdefault(match.node_id, match.text_chunk)

        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            VectorMatch(
                node_id=node_id,
                similarity=score,
                text_chunk=text_chunks.get(node_id, ""),
            )
            for node_id, score in ordered[:top_k]
        ]

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

    def _compose_query(self, current_prompt: str) -> str:
        snapshot = self.working_memory.snapshot()
        recent_context: list[str] = []
        for message in snapshot.messages[-4:]:
            if message.content == current_prompt:
                continue
            if message.role not in {"user", "assistant"}:
                continue
            recent_context.append(message.content)

        if not recent_context:
            return current_prompt
        if _should_strip_recent_context(current_prompt, recent_context):
            return current_prompt

        return "\n".join([current_prompt, *recent_context])


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [1.0 for _ in values]

    return [(value - minimum) / (maximum - minimum) for value in values]


def _should_strip_recent_context(current_prompt: str, recent_context: list[str]) -> bool:
    prompt_tokens = _context_tokens(current_prompt)
    if len(prompt_tokens) < 4:
        return False
    if prompt_tokens & REFERENTIAL_CONTEXT_MARKERS:
        return False

    context_tokens: set[str] = set()
    for line in recent_context:
        context_tokens.update(_context_tokens(line))
    if not context_tokens:
        return False
    return _jaccard_overlap(prompt_tokens, context_tokens) < MIN_CONTEXT_DRIFT_JACCARD


def _context_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "from", "into", "about"}
    }


def _jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
