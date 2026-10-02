"""Intent-routing orchestrator.

``IntentRouter`` is a drop-in replacement for ``LocalIntentRouter`` at the
kernel call site: it exposes the same ``.decide(prompt) -> LocalRouteDecision``
contract. It always computes the heuristic answer first, then optionally
consults a System One provider in ``shadow`` (record only) or ``enforce``
(act on the answer when confident) mode, and always falls back to the
heuristic on any failure.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable

from core.runtime.local_router import LocalIntentRouter, LocalRouteDecision

from .config import GATE_INTENT, DecisionConfig
from .models import DecisionError, DecisionMode, DecisionRecord
from .provider import (
    HeuristicDecisionProvider,
    RouteDecisionProvider,
    build_decision_provider,
)

logger = logging.getLogger(__name__)

Recorder = Callable[[DecisionRecord], None]


class IntentRouter:
    def __init__(
        self,
        *,
        heuristic: LocalIntentRouter | None = None,
        provider: RouteDecisionProvider | None = None,
        mode: DecisionMode = DecisionMode.OFF,
        provider_name: str = "heuristic",
        min_confidence: float = 0.65,
        recorder: Recorder | None = None,
    ) -> None:
        self._heuristic = heuristic or LocalIntentRouter()
        self._provider = provider
        self._mode = mode
        self._provider_name = provider_name
        self._min_confidence = float(min_confidence)
        self._recorder = recorder

    # Kept for structural compatibility with the tests that swap in a fake.
    @property
    def threshold(self) -> float:
        return getattr(self._heuristic, "threshold", 0.44)

    def decide(self, prompt: str) -> LocalRouteDecision:
        base = self._heuristic.decide(prompt)
        if self._mode is DecisionMode.OFF or self._provider is None:
            return base

        state_hash = _hash_prompt(prompt)
        started = time.perf_counter()
        try:
            decision = self._provider.decide(prompt)
        except DecisionError as exc:
            self._emit(base, latency_ms=_ms(started), state_hash=state_hash, fallback=True, error=str(exc))
            return base
        except Exception as exc:  # pragma: no cover - defensive; provider must not raise raw
            logger.debug("Intent decision provider failed: %s", exc, exc_info=True)
            self._emit(
                base,
                latency_ms=_ms(started),
                state_hash=state_hash,
                fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            return base

        latency_ms = _ms(started)
        if self._mode is DecisionMode.SHADOW:
            self._emit(decision, latency_ms=latency_ms, state_hash=state_hash, shadow=True)
            return base
        if decision.confidence >= self._min_confidence:
            self._emit(decision, latency_ms=latency_ms, state_hash=state_hash)
            return decision
        self._emit(
            decision,
            latency_ms=latency_ms,
            state_hash=state_hash,
            fallback=True,
            error="below confidence threshold",
        )
        return base

    def _emit(
        self,
        decision: LocalRouteDecision,
        *,
        latency_ms: float,
        state_hash: str,
        fallback: bool = False,
        shadow: bool | None = None,
        error: str = "",
    ) -> None:
        if self._recorder is None:
            return
        resolved_shadow = (self._mode is DecisionMode.SHADOW) if shadow is None else shadow
        answer = {
            "use_local_knowledge": bool(decision.use_local_knowledge),
            "reason": decision.reason,
        }
        if error:
            answer["error"] = error
        record = DecisionRecord(
            gate=GATE_INTENT,
            domain="intent",
            provider=self._provider_name,
            mode=self._mode.value,
            answer=answer,
            confidence=float(decision.confidence),
            reason=decision.reason,
            latency_ms=float(latency_ms),
            fallback=bool(fallback),
            shadow=bool(resolved_shadow),
            state_hash=state_hash,
        )
        try:
            self._recorder(record)
        except Exception:  # pragma: no cover - recording must never break routing
            logger.debug("Decision recorder failed", exc_info=True)


def build_intent_router(
    config: DecisionConfig,
    *,
    recorder: Recorder | None = None,
    heuristic: LocalIntentRouter | None = None,
    provider: RouteDecisionProvider | None = None,
) -> IntentRouter:
    selected = provider if provider is not None else build_decision_provider(config, offline=False)
    name = str(getattr(selected, "provider_name", "heuristic"))
    if name == "heuristic":
        selected = None
    return IntentRouter(
        heuristic=heuristic,
        provider=selected,
        mode=config.mode_for(GATE_INTENT),
        provider_name=name,
        min_confidence=config.min_confidence,
        recorder=recorder,
    )


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()[:16]


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


# Re-exported so callers can build a heuristic provider when needed.
__all__ = ["IntentRouter", "build_intent_router", "HeuristicDecisionProvider"]
