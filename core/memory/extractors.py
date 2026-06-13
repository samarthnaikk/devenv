from __future__ import annotations

import json
import re
from typing import Protocol

from .models import ConsolidationEntity, ConsolidationExtraction, ConsolidationUpdate, EpisodicLog, MemoryNode


class ConsolidationExtractor(Protocol):
    def extract(self, logs: list[EpisodicLog], existing_nodes: list[MemoryNode]) -> ConsolidationExtraction:
        ...


class HeuristicConsolidationExtractor:
    def extract(self, logs: list[EpisodicLog], existing_nodes: list[MemoryNode]) -> ConsolidationExtraction:
        label_index = {node.label.lower(): node for node in existing_nodes}
        new_entities: list[ConsolidationEntity] = []
        updates: list[ConsolidationUpdate] = []
        detected_project: str | None = None

        for log in logs:
            payload = json.loads(log.raw_interaction)
            metadata = payload.get("metadata", {})
            detected_project = detected_project or metadata.get("project")

            memory_entities = metadata.get("memory_entities", [])
            for entity in memory_entities:
                label = str(entity["label"])
                summary = str(entity["summary"])
                category = str(entity.get("category", "component"))
                parent_id = entity.get("parent_id")
                known = label_index.get(label.lower())
                if known is None:
                    new_entities.append(
                        ConsolidationEntity(
                            node_id=entity.get("node_id"),
                            label=label,
                            category=category,
                            summary=summary,
                            parent_id=parent_id,
                        )
                    )
                else:
                    updates.append(ConsolidationUpdate(node_id=known.node_id, append_summary=summary))

            memory_updates = metadata.get("memory_updates", [])
            for update in memory_updates:
                updates.append(
                    ConsolidationUpdate(
                        node_id=str(update["node_id"]),
                        append_summary=str(update["append_summary"]),
                    )
                )

            if log.associated_node_id:
                updates.append(
                    ConsolidationUpdate(
                        node_id=log.associated_node_id,
                        append_summary=_summarize_log(payload),
                    )
                )

        return ConsolidationExtraction(
            detected_project=detected_project,
            new_entities=tuple(new_entities),
            updates_to_existing_nodes=tuple(updates),
        )


def slugify_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return normalized or "memory_node"


def _summarize_log(payload: dict[str, object]) -> str:
    user = str(payload.get("user", "")).strip()
    agent = str(payload.get("agent", "")).strip()
    pieces = [piece for piece in (user, agent) if piece]
    return " | ".join(pieces)
