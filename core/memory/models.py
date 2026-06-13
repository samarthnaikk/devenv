from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeEdge:
    source_node_id: str
    target_node_id: str
    relationship_type: str


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    label: str
    category: str
    summary: str
    parent_id: str | None = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0


@dataclass(frozen=True)
class NodeUpsertPayload:
    node_id: str
    label: str
    category: str
    summary: str
    parent_id: str | None = None
    edges: tuple[NodeEdge, ...] = ()


@dataclass(frozen=True)
class EpisodicLog:
    log_id: str
    timestamp: float
    raw_interaction: str
    associated_node_id: str | None = None


@dataclass(frozen=True)
class WorkingMemoryMessage:
    role: str
    content: str
    timestamp: float | None = None


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    messages: tuple[WorkingMemoryMessage, ...]
    active_state: dict[str, Any]


@dataclass(frozen=True)
class VectorMatch:
    node_id: str
    similarity: float
    text_chunk: str


@dataclass(frozen=True)
class RetrievalCandidate:
    node: MemoryNode
    source_node_id: str
    relationship: str
    similarity: float
    frequency_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0


@dataclass(frozen=True)
class RetrievalSelectedNode:
    node_id: str
    label: str
    category: str
    summary: str
    score: float
    relationship: str


@dataclass(frozen=True)
class RetrievalTrace:
    matched_nodes: tuple[VectorMatch, ...] = ()
    expanded_candidates: tuple[RetrievalCandidate, ...] = ()
    selected_nodes: tuple[RetrievalSelectedNode, ...] = ()
    markdown_context: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    markdown_context: str
    selected_nodes: tuple[RetrievalSelectedNode, ...]
    trace: RetrievalTrace


@dataclass(frozen=True)
class ConsolidationEntity:
    label: str
    category: str
    summary: str
    parent_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class ConsolidationUpdate:
    node_id: str
    append_summary: str


@dataclass(frozen=True)
class ConsolidationExtraction:
    detected_project: str | None = None
    new_entities: tuple[ConsolidationEntity, ...] = ()
    updates_to_existing_nodes: tuple[ConsolidationUpdate, ...] = ()


@dataclass(frozen=True)
class ConsolidationResult:
    processed_logs: int
    created_nodes: tuple[str, ...] = ()
    updated_nodes: tuple[str, ...] = ()
    detected_project: str | None = None


@dataclass
class LogInteraction:
    user: str
    agent: str
    metadata: dict[str, Any] = field(default_factory=dict)

