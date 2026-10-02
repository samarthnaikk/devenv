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
from .models import DecisionError, DecisionUnavailable, WriteDecision
from .redaction import prepare_remote_state
from .systemone import SystemOneClient


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


class SystemOneDecisionProvider:
    """Consult a hosted TypeSafe Jev or local Jev-compatible server.

    The provider is intentionally "pure": it either returns a decision or raises
    :class:`~core.decisions.models.DecisionError`. The gate orchestrator owns
    fallback and recording so failures are visible in the audit trail.
    """

    name = "systemone"

    def __init__(
        self,
        config: DecisionConfig,
        *,
        client: SystemOneClient | None = None,
        offline: bool = False,
    ) -> None:
        self.config = config
        self.provider_name = config.provider
        self._client = client or SystemOneClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.timeout,
            retries=config.retries,
        )
        self._offline = offline

    def _guard_offline(self) -> None:
        if not self.config.is_remote:
            return
        from core.runtime.connectivity import is_offline

        if is_offline():
            raise DecisionUnavailable("offline: remote decision provider is disabled")

    def decide(self, prompt: str) -> LocalRouteDecision:
        from .questions import build_intent_questions, parse_intent_answers

        self._guard_offline()
        state = prepare_remote_state(prompt, redact=self.config.redact)
        if state is None:
            raise DecisionUnavailable("state looks sensitive; refusing to send it remotely")
        result = self._client.evaluate(state, build_intent_questions())
        intent = parse_intent_answers(result.answers)
        return LocalRouteDecision(
            use_local_knowledge=intent.use_local_knowledge,
            confidence=float(intent.confidence),
            knowledge_score=float(intent.confidence) if intent.use_local_knowledge else 0.0,
            remote_score=0.0 if intent.use_local_knowledge else float(intent.confidence),
            reason=intent.reason,
        )


def build_decision_provider(
    config: DecisionConfig,
    *,
    offline: bool = False,
) -> HeuristicDecisionProvider | SystemOneDecisionProvider:
    """Build the configured provider (heuristic by default)."""

    if config.uses_systemone and not (offline and config.is_remote):
        return SystemOneDecisionProvider(config, offline=offline)
    return HeuristicDecisionProvider()


__all__ = [
    "HeuristicDecisionProvider",
    "MemoryWriteDecisionProvider",
    "RouteDecisionProvider",
    "SystemOneDecisionProvider",
    "build_decision_provider",
]
