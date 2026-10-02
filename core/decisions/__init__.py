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
    SystemOneDecisionProvider,
    build_decision_provider,
)
from .questions import (
    INTENT_ROUTES,
    IntentResult,
    build_intent_questions,
    parse_intent_answers,
)
from .redaction import looks_sensitive, prepare_remote_state, redact_state
from .router import IntentRouter, build_intent_router
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
    "INTENT_ROUTES",
    "IntentResult",
    "IntentRouter",
    "KNOWN_PROVIDERS",
    "MemoryWriteDecisionProvider",
    "RouteDecisionProvider",
    "SystemOneClient",
    "SystemOneDecisionProvider",
    "SystemOneResult",
    "WriteDecision",
    "build_decision_provider",
    "build_intent_questions",
    "build_intent_router",
    "looks_sensitive",
    "parse_intent_answers",
    "prepare_remote_state",
    "redact_state",
]
