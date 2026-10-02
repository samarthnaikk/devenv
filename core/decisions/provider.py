"""Decision provider protocols and the default heuristic provider.

A *provider* is a small object that answers one or more bounded decision gates.
The heuristic provider reproduces Devenv's existing behavior; System One
providers (hosted TypeSafe Jev or a local Jev-compatible server) are layered on
top and always fall back to the heuristic on failure.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from core.runtime.local_router import LocalIntentRouter, LocalRouteDecision

from .config import DecisionConfig
from .models import WriteDecision


@runtime_checkable
class RouteDecisionProvider(Protocol):
    def decide(self, prompt: str) -> LocalRouteDecision: ...


@runtime_checkable
class MemoryWriteDecisionProvider(Protocol):
    def judge_writes(
        self,
        candidates: Sequence[Any],
        existing_nodes: Sequence[Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[WriteDecision]: ...


class HeuristicDecisionProvider:
    """Default provider. Preserves today's behavior exactly."""

    name = "heuristic"

    def __init__(self, *, router: LocalIntentRouter | None = None) -> None:
        self._router = router or LocalIntentRouter()

    def decide(self, prompt: str) -> LocalRouteDecision:
        return self._router.decide(prompt)

    def judge_writes(
        self,
        candidates: Sequence[Any],
        existing_nodes: Sequence[Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[WriteDecision]:
        del existing_nodes, context
        return [
            WriteDecision(
                should_write=True,
                confidence=1.0,
                reason="heuristic passthrough",
                source=self.name,
            )
            for _ in candidates
        ]


def build_decision_provider(
    config: DecisionConfig,
    *,
    offline: bool = False,
) -> HeuristicDecisionProvider:
    """Build the configured provider.

    Phase 0 ships the heuristic provider only. System One providers are added in
    later phases and selected here.
    """

    del config, offline
    return HeuristicDecisionProvider()


__all__ = [
    "HeuristicDecisionProvider",
    "MemoryWriteDecisionProvider",
    "RouteDecisionProvider",
    "build_decision_provider",
]
