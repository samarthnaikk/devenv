from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .models import DecisionMode

_TRUE_VALUES = {"1", "true", "yes", "on"}

DEFAULT_PROVIDER = "off"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_TIMEOUT = 2.0
DEFAULT_RETRIES = 1
DEFAULT_REDACT = True
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_CACHE_SECONDS = 300.0

KNOWN_PROVIDERS = ("off", "heuristic", "typesafe", "local", "auto")

GATE_INTENT = "intent"
GATE_MEMORY_WRITE = "memory_write"


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE_VALUES


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DecisionConfig:
    """Resolved configuration for the decision layer.

    ``provider`` selects the backend; ``intent_mode`` / ``memory_write_mode``
    select whether that backend is consulted and whether its answer is enforced.
    Defaults reproduce today's behavior exactly (heuristic passthrough).
    """

    provider: str = DEFAULT_PROVIDER
    intent_mode: DecisionMode = DecisionMode.OFF
    memory_write_mode: DecisionMode = DecisionMode.OFF
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    redact: bool = DEFAULT_REDACT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    cache_seconds: float = DEFAULT_CACHE_SECONDS

    @property
    def uses_systemone(self) -> bool:
        return self.provider in {"typesafe", "local", "auto"}

    @property
    def is_remote(self) -> bool:
        return self.provider == "typesafe" or (
            self.provider == "auto" and not _is_local_url(self.base_url)
        )

    def mode_for(self, gate: str) -> DecisionMode:
        if gate == GATE_INTENT:
            return self.intent_mode
        if gate == GATE_MEMORY_WRITE:
            return self.memory_write_mode
        return DecisionMode.OFF

    def enabled_for(self, gate: str) -> bool:
        return self.mode_for(gate) is not DecisionMode.OFF

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DecisionConfig":
        def raw(name: str) -> str | None:
            if env is not None:
                return env.get(name)
            return os.getenv(name)

        def get(name: str, default: str) -> str:
            value = raw(name)
            return default if value is None else str(value)

        provider = get("DEVENV_DECISION_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
        if provider not in KNOWN_PROVIDERS:
            provider = DEFAULT_PROVIDER

        default_mode = DecisionMode.parse(get("DEVENV_DECISION_MODE", ""), default=DecisionMode.OFF)
        intent_mode = DecisionMode.parse(get("DEVENV_DECISION_INTENT", ""), default=default_mode)
        memory_mode = DecisionMode.parse(get("DEVENV_DECISION_MEMORY_WRITE", ""), default=default_mode)

        base_url = get("DEVENV_SYSTEMONE_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        model = get("DEVENV_SYSTEMONE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        api_key = get("TYPESAFE_API_KEY", "").strip()

        return cls(
            provider=provider,
            intent_mode=intent_mode,
            memory_write_mode=memory_mode,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=max(0.0, _parse_float(raw("DEVENV_DECISION_TIMEOUT"), DEFAULT_TIMEOUT)),
            retries=max(0, _parse_int(raw("DEVENV_DECISION_RETRIES"), DEFAULT_RETRIES)),
            redact=_parse_bool(raw("DEVENV_DECISION_REDACT"), DEFAULT_REDACT),
            min_confidence=min(
                1.0, max(0.0, _parse_float(raw("DEVENV_DECISION_MIN_CONFIDENCE"), DEFAULT_MIN_CONFIDENCE))
            ),
            cache_seconds=max(0.0, _parse_float(raw("DEVENV_DECISION_CACHE_SECONDS"), DEFAULT_CACHE_SECONDS)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "intent_mode": self.intent_mode.value,
            "memory_write_mode": self.memory_write_mode.value,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "redact": self.redact,
            "min_confidence": self.min_confidence,
        }


def _is_local_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(token in lowered for token in ("127.0.0.1", "localhost", "0.0.0.0", "::1"))


__all__ = [
    "DecisionConfig",
    "GATE_INTENT",
    "GATE_MEMORY_WRITE",
    "KNOWN_PROVIDERS",
]
