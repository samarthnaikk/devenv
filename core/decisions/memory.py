"""Decision-gated consolidation.

``DecisionConsolidationExtractor`` wraps any ``ConsolidationExtractor`` and asks
a decision provider whether each newly extracted entity is durable enough to
become a memory node. It never touches updates to existing nodes (write gate
only) and fails open: on provider error, low confidence, or a sensitive
candidate, the entity is kept exactly as the wrapped extractor produced it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable

from core.memory.extractors import ConsolidationExtractor, HeuristicConsolidationExtractor
from core.memory.models import ConsolidationEntity, ConsolidationExtraction

from .config import GATE_MEMORY_WRITE, DecisionConfig
from .models import DecisionError, DecisionMode, DecisionRecord, WriteDecision
from .provider import MemoryWriteDecisionProvider, build_decision_provider

logger = logging.getLogger(__name__)

Recorder = Callable[[DecisionRecord], None]


class DecisionConsolidationExtractor:
    def __init__(
        self,
        *,
        inner: ConsolidationExtractor | None = None,
        provider: MemoryWriteDecisionProvider,
        mode: DecisionMode = DecisionMode.OFF,
        min_confidence: float = 0.65,
        recorder: Recorder | None = None,
    ) -> None:
        self._inner = inner or HeuristicConsolidationExtractor()
        self._provider = provider
        self._mode = mode
        self._min_confidence = float(min_confidence)
        self._recorder = recorder
        self.provider_name = str(getattr(provider, "provider_name", "heuristic"))

    def extract(self, logs, existing_nodes) -> ConsolidationExtraction:
        base = self._inner.extract(logs, existing_nodes)
        if self._mode is DecisionMode.OFF or not base.new_entities:
            return base

        candidates = list(base.new_entities)
        started = time.perf_counter()
        batch_failure = False
        error = ""
        try:
            decisions = self._provider.judge_writes(
                candidates,
                existing_nodes,
                context={"source": "consolidation"},
            )
            if len(decisions) != len(candidates):
                raise DecisionError("provider returned a mismatched decision count")
        except DecisionError as exc:
            batch_failure = True
            error = str(exc)
            decisions = _keep_all(len(candidates))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Memory write decision provider failed: %s", exc, exc_info=True)
            batch_failure = True
            error = f"{type(exc).__name__}: {exc}"
            decisions = _keep_all(len(candidates))

        latency_ms = _ms(started)
        shadow = self._mode is DecisionMode.SHADOW
        for candidate, decision in zip(candidates, decisions):
            self._record(
                candidate,
                decision,
                fallback=batch_failure or decision.source != "systemone",
                shadow=shadow,
                latency_ms=latency_ms,
                error=error,
            )

        if shadow:
            return base

        kept = tuple(candidate for candidate, decision in zip(candidates, decisions) if decision.should_write)
        return ConsolidationExtraction(
            detected_project=base.detected_project,
            new_entities=kept,
            updates_to_existing_nodes=base.updates_to_existing_nodes,
        )

    def _record(
        self,
        candidate: ConsolidationEntity,
        decision: WriteDecision,
        *,
        fallback: bool,
        shadow: bool,
        latency_ms: float,
        error: str,
    ) -> None:
        if self._recorder is None:
            return
        label = str(getattr(candidate, "label", "") or "")
        answer: dict[str, Any] = {
            "should_write": bool(decision.should_write),
            "label": label,
        }
        if error:
            answer["error"] = error
        record = DecisionRecord(
            gate=GATE_MEMORY_WRITE,
            domain="memory",
            provider=self.provider_name,
            mode=self._mode.value,
            answer=answer,
            confidence=float(decision.confidence),
            reason=decision.reason,
            latency_ms=float(latency_ms),
            fallback=bool(fallback),
            shadow=bool(shadow),
            state_hash=_hash_text(label or str(getattr(candidate, "summary", ""))),
        )
        try:
            self._recorder(record)
        except Exception:  # pragma: no cover - recording must never break consolidation
            logger.debug("Decision recorder failed", exc_info=True)


def build_memory_extractor(
    config: DecisionConfig,
    *,
    inner: ConsolidationExtractor | None = None,
    provider: MemoryWriteDecisionProvider | None = None,
    recorder: Recorder | None = None,
) -> ConsolidationExtractor:
    """Return the wrapped extractor when the gate is configured, else ``inner``."""

    base = inner or HeuristicConsolidationExtractor()
    if not config.enabled_for(GATE_MEMORY_WRITE):
        return base
    selected = provider if provider is not None else build_decision_provider(config, offline=False)
    name = str(getattr(selected, "provider_name", "heuristic"))
    if name == "heuristic":
        return base
    return DecisionConsolidationExtractor(
        inner=base,
        provider=selected,
        mode=config.mode_for(GATE_MEMORY_WRITE),
        min_confidence=config.min_confidence,
        recorder=recorder,
    )


def _keep_all(count: int) -> list[WriteDecision]:
    return [
        WriteDecision(should_write=True, confidence=1.0, reason="provider failed; kept", source="heuristic")
        for _ in range(count)
    ]


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


__all__ = ["DecisionConsolidationExtractor", "build_memory_extractor"]