from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


class DecisionMode(str, Enum):
    """How a decision seam should treat provider output."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @classmethod
    def parse(cls, value: str | None, *, default: "DecisionMode | None" = None) -> "DecisionMode":
        fallback = default if default is not None else cls.OFF
        if value is None:
            return fallback
        cleaned = str(value).strip().lower()
        if not cleaned:
            return fallback
        try:
            return cls(cleaned)
        except ValueError:
            return fallback


class DecisionError(RuntimeError):
    """Base error for decision-layer failures."""


class DecisionTimeout(DecisionError):
    """Raised when a decision provider call exceeds its timeout."""


class DecisionUnavailable(DecisionError):
    """Raised when a requested decision provider is not configured/available."""


@dataclass(frozen=True)
class WriteDecision:
    """Whether an extracted memory candidate should be written to the store."""

    should_write: bool
    confidence: float = 1.0
    reason: str = ""
    source: str = "heuristic"
    latency_ms: float = 0.0
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_write": bool(self.should_write),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "source": self.source,
            "latency_ms": float(self.latency_ms),
            "category": self.category,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """Auditable record of a single decision produced by a provider."""

    gate: str
    domain: str
    provider: str
    mode: str
    answer: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    reason: str = ""
    latency_ms: float = 0.0
    fallback: bool = False
    shadow: bool = False
    state_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gate": self.gate,
            "domain": self.domain,
            "provider": self.provider,
            "mode": self.mode,
            "answer": dict(self.answer),
            "reason": self.reason,
            "latency_ms": float(self.latency_ms),
            "fallback": bool(self.fallback),
            "shadow": bool(self.shadow),
        }
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.state_hash:
            payload["state_hash"] = self.state_hash
        return payload


__all__ = [
    "DecisionError",
    "DecisionMode",
    "DecisionRecord",
    "DecisionTimeout",
    "DecisionUnavailable",
    "WriteDecision",
]
