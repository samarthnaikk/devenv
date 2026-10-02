"""Swappable decision layer for Devenv.

Phase 0 exposes the reusable foundation: configuration, the System One HTTP
client, provider protocols, and the default heuristic provider. Intent and
memory-write gates are layered on top in later phases.
"""

from __future__ import annotations

from .config import (
    GATE_INTENT,
    GATE_MEMORY_WRITE,
    DecisionConfig,
    KNOWN_PROVIDERS,
)
from .models import (
    DecisionError,
    DecisionMode,
    DecisionRecord,
    DecisionTimeout,
    DecisionUnavailable,
    WriteDecision,
)
from .provider import (
    HeuristicDecisionProvider,
    MemoryWriteDecisionProvider,
    RouteDecisionProvider,
    build_decision_provider,
)
from .redaction import looks_sensitive, prepare_remote_state, redact_state
from .systemone import SystemOneClient, SystemOneResult

__all__ = [
    "DecisionConfig",
    "DecisionError",
    "DecisionMode",
    "DecisionRecord",
    "DecisionTimeout",
    "DecisionUnavailable",
    "GATE_INTENT",
    "GATE_MEMORY_WRITE",
    "HeuristicDecisionProvider",
    "KNOWN_PROVIDERS",
    "MemoryWriteDecisionProvider",
    "RouteDecisionProvider",
    "SystemOneClient",
    "SystemOneResult",
    "WriteDecision",
    "build_decision_provider",
    "looks_sensitive",
    "prepare_remote_state",
    "redact_state",
]
