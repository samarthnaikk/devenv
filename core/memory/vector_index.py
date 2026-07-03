from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import VectorMatch


class VectorIndex(Protocol):
    def upsert(self, node_id: str, text_chunk: str, vector: list[float]) -> None:
        ...

    def delete(self, node_id: str) -> None:
        ...

    def query(self, vector: list[float], top_k: int, min_similarity: float) -> list[VectorMatch]:
        ...

    def has_persisted_state(self) -> bool:
        ...


@dataclass
class LanceDBVectorIndex:
    vector_dir: str
    table_name: str = "memory_nodes"
    dimension: int = 384

    def __post_init__(self) -> None:
        self._db = None
        self._table = None
        Path(self.vector_dir).mkdir(parents=True, exist_ok=True)

    def upsert(self, node_id: str, text_chunk: str, vector: list[float]) -> None:
        table = self._table_instance()
        payload = [{"node_id": node_id, "vector": vector, "text_chunk": text_chunk}]
        try:
            table.merge_insert("node_id").when_matched_update_all().when_not_matched_insert_all().execute(payload)
        except AttributeError:
            self.delete(node_id)
            table.add(payload)

    def delete(self, node_id: str) -> None:
        table = self._table_instance()
        table.delete(f"node_id = '{node_id}'")

    def query(self, vector: list[float], top_k: int, min_similarity: float) -> list[VectorMatch]:
        table = self._table_instance()
        try:
            rows = table.search(vector).limit(top_k).to_list()
        except AttributeError as exc:
            raise RuntimeError("Installed LanceDB version does not support the configured search API.") from exc

        matches: list[VectorMatch] = []
        for row in rows:
            similarity = self._coerce_similarity(row.get("_distance"), row.get("score"), vector, row.get("vector"))
            if similarity >= min_similarity:
                matches.append(
                    VectorMatch(
                        node_id=str(row["node_id"]),
                        similarity=similarity,
                        text_chunk=str(row.get("text_chunk", "")),
                    )
                )
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def has_persisted_state(self) -> bool:
        try:
            import lancedb
        except ImportError:
            return False

        if self._db is None:
            self._db = lancedb.connect(self.vector_dir)

        try:
            self._db.open_table(self.table_name)
        except Exception:
            return False
        return True

    def _coerce_similarity(
        self,
        distance: float | None,
        score: float | None,
        query_vector: list[float],
        stored_vector: list[float] | None,
    ) -> float:
        if score is not None:
            return float(score)
        if distance is not None:
            return 1.0 / (1.0 + float(distance))
        if stored_vector is None:
            return 0.0
        return _cosine_similarity(query_vector, [float(value) for value in stored_vector])

    def _table_instance(self):
        if self._table is not None:
            return self._table

        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError(
                "lancedb is required for the production vector index. "
                "Inject an in-memory vector index in tests or install the dependency."
            ) from exc

        if self._db is None:
            self._db = lancedb.connect(self.vector_dir)

        try:
            self._table = self._db.open_table(self.table_name)
        except Exception:
            bootstrap = [{"node_id": "__bootstrap__", "vector": [0.0] * self.dimension, "text_chunk": ""}]
            self._table = self._db.create_table(self.table_name, data=bootstrap, mode="overwrite")
            self._table.delete("node_id = '__bootstrap__'")
        return self._table


@dataclass
class InMemoryVectorIndex:
    records: dict[str, tuple[str, list[float]]] = field(default_factory=dict)

    def upsert(self, node_id: str, text_chunk: str, vector: list[float]) -> None:
        self.records[node_id] = (text_chunk, vector)

    def delete(self, node_id: str) -> None:
        self.records.pop(node_id, None)

    def query(self, vector: list[float], top_k: int, min_similarity: float) -> list[VectorMatch]:
        matches: list[VectorMatch] = []
        for node_id, (text_chunk, candidate_vector) in self.records.items():
            similarity = _cosine_similarity(vector, candidate_vector)
            if similarity >= min_similarity:
                matches.append(VectorMatch(node_id=node_id, similarity=similarity, text_chunk=text_chunk))

        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def has_persisted_state(self) -> bool:
        return False


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return numerator / (left_magnitude * right_magnitude)
