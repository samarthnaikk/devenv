from __future__ import annotations

from dataclasses import dataclass

from .embeddings import BGE_QUERY_PREFIX, build_card_embedder
from .models import InteractionCard, VectorMatch
from .storage import SQLiteMemoryStore
from .vector_index import VectorIndex

RRF_K = 60
DEFAULT_LANE_TOP_K = 5
DEFAULT_TOP_K = 8
DEFAULT_MIN_SIMILARITY = 0.15
DEFAULT_MIN_SCORE = 0.02


@dataclass(frozen=True)
class CardMatch:
    card: InteractionCard
    score: float
    lane: str


class CardRetriever:
    """Lane-isolated hybrid retrieval over interaction cards (dense + FTS5, RRF)."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        embedder=None,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.embedder = embedder or build_card_embedder()

    def search_lane(self, lane_query: str, top_k: int = DEFAULT_LANE_TOP_K) -> list[CardMatch]:
        lane_query = (lane_query or "").strip()
        if not lane_query:
            return []
        dense_matches: list[VectorMatch] = []
        try:
            query_vector = self.embedder.embed(BGE_QUERY_PREFIX + lane_query)
            dense_matches = self.vector_index.query(
                list(query_vector), top_k=top_k * 3, min_similarity=DEFAULT_MIN_SIMILARITY
            )
        except Exception:
            dense_matches = []
        lexical_cards = self.store.search_interaction_cards_fts(lane_query, limit=top_k * 3)
        return self._fuse_lane(dense_matches, lexical_cards, lane_query, top_k)

    def _fuse_lane(
        self,
        dense_matches: list[VectorMatch],
        lexical_cards: list[InteractionCard],
        lane: str,
        top_k: int,
    ) -> list[CardMatch]:
        scores: dict[str, float] = {}
        cards: dict[str, InteractionCard] = {}
        for rank, match in enumerate(dense_matches, start=1):
            scores[match.node_id] = scores.get(match.node_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, card in enumerate(lexical_cards, start=1):
            scores[card.card_id] = scores.get(card.card_id, 0.0) + 1.0 / (RRF_K + rank)
            cards.setdefault(card.card_id, card)
        for card_id in list(scores):
            if card_id not in cards:
                card = self.store.get_interaction_card(card_id)
                if card is not None:
                    cards[card_id] = card
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [CardMatch(card=cards[card_id], score=score, lane=lane) for card_id, score in ordered if card_id in cards]

    def retrieve(
        self,
        lanes: list[str],
        per_lane: int = DEFAULT_LANE_TOP_K,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[CardMatch]:
        lane_results = [(lane, self.search_lane(lane, top_k=per_lane)) for lane in lanes if (lane or "").strip()]
        lane_results = [(lane, matches) for lane, matches in lane_results if matches]
        if not lane_results:
            return []

        chosen: dict[str, CardMatch] = {}
        for _lane, matches in lane_results:
            best = matches[0]
            chosen.setdefault(best.card.card_id, best)

        pooled = sorted(
            (match for _lane, matches in lane_results for match in matches),
            key=lambda match: (-match.score, match.card.card_id),
        )
        for match in pooled:
            if len(chosen) >= top_k:
                break
            chosen.setdefault(match.card.card_id, match)
        return sorted(chosen.values(), key=lambda match: (-match.score, match.card.card_id))[:top_k]
