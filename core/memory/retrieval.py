from __future__ import annotations

import math
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
HIGH_CONFIDENCE_VECTOR_MATCH = 0.88
RECENCY_HALF_LIFE_SECONDS = 60.0 * 60.0 * 24.0 * 7.0
REFERENTIAL_CONTEXT_MARKERS = {
    "again",
    "earlier",
    "it",
    "that",
    "them",
    "those",
    "this",
}
MAX_QUERY_VARIANTS = 4


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
        query_variants = self._query_variants(current_prompt, query_text)
        matches, short_circuit_expansion = self._retrieve_seed_matches(query_variants, top_k=max(top_k, 5))
        if not matches:
            markdown = self._compile_markdown([], include_working_memory=True)
            trace = RetrievalTrace(markdown_context=markdown)
            return RetrievalResult(markdown_context=markdown, selected_nodes=(), trace=trace)

        candidates = self._seed_candidates(matches) if short_circuit_expansion else self._expand_candidates(matches)
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

    def _retrieve_seed_matches(self, query_variants: list[str], top_k: int) -> tuple[list[VectorMatch], bool]:
        vector_variants: list[list[VectorMatch]] = []
        lexical_variants: list[list[VectorMatch]] = []
        short_circuit_expansion = False
        for variant in query_variants:
            query_vector = self.embedder.embed(variant)
            vector_matches = self.vector_index.query(
                query_vector,
                top_k=top_k,
                min_similarity=self.similarity_threshold,
            )
            vector_variants.append(vector_matches)
            short_circuit_expansion = short_circuit_expansion or self._should_short_circuit_expansion(vector_matches)
            lexical_variants.append(self._lexical_seed_matches(variant, top_k=top_k))
        return self._fuse_variant_seed_matches(vector_variants, lexical_variants, top_k=top_k), short_circuit_expansion

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

    def _seed_candidates(self, matches: list[VectorMatch]) -> list[RetrievalCandidate]:
        seeds: list[RetrievalCandidate] = []
        for match in matches:
            node = self.store.get_node(match.node_id)
            if node is None:
                continue
            seeds.append(
                RetrievalCandidate(
                    node=node,
                    source_node_id=node.node_id,
                    relationship="seed",
                    similarity=match.similarity,
                )
            )
        return seeds

    def _top_structural_neighbors(self, nodes: list[MemoryNode], *, limit: int) -> list[MemoryNode]:
        ranked = sorted(
            nodes,
            key=lambda node: (node.access_count, node.last_accessed, node.created_at),
            reverse=True,
        )
        return ranked[:limit]

    def _should_short_circuit_expansion(self, vector_matches: list[VectorMatch]) -> bool:
        return bool(vector_matches and vector_matches[0].similarity >= HIGH_CONFIDENCE_VECTOR_MATCH)

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

    def _fuse_variant_seed_matches(
        self,
        vector_variants: list[list[VectorMatch]],
        lexical_variants: list[list[VectorMatch]],
        top_k: int,
    ) -> list[VectorMatch]:
        fused_scores: dict[str, float] = {}
        text_chunks: dict[str, str] = {}

        for vector_matches, lexical_matches in zip(vector_variants, lexical_variants):
            fused = self._fuse_seed_matches(vector_matches, lexical_matches, top_k=top_k)
            for rank, match in enumerate(fused, start=1):
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
        elapsed = max(time.time() - node.last_accessed, 0.0)
        if RECENCY_HALF_LIFE_SECONDS <= 0:
            return 1.0
        return math.exp((-math.log(2.0) * elapsed) / RECENCY_HALF_LIFE_SECONDS)

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

    def _query_variants(self, current_prompt: str, query_text: str) -> list[str]:
        variants = [query_text]
        if query_text != current_prompt:
            variants.append(current_prompt)
        for fragment in _split_query_fragments(current_prompt):
            if fragment not in variants:
                variants.append(fragment)
            if len(variants) >= MAX_QUERY_VARIANTS:
                break
        return variants[:MAX_QUERY_VARIANTS]


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


def _split_query_fragments(text: str) -> list[str]:
    normalized = _normalize_fragment(text)
    if not normalized:
        return []

    fragments: list[str] = []
    seen: set[str] = set()
    for raw_part in re.split(r"[?.!;]+|\b(?:and|also|then|plus|while|versus|vs)\b", normalized, flags=re.IGNORECASE):
        fragment = _normalize_fragment(raw_part)
        if len(_context_tokens(fragment)) < 2 or fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)
        if len(fragments) >= MAX_QUERY_VARIANTS - 1:
            break
    return fragments


def _normalize_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.;:-")
