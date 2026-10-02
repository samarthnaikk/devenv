"""Minimal client for the System One ``/v1/systemone`` contract.

Speaks the same request/response shape for hosted TypeSafe Jev and for local
Jev-compatible servers (Laya ``laya-serve``, Lichen). Uses the standard library
only so the decision layer adds no hard dependency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import DecisionError, DecisionTimeout

DEFAULT_MODEL = "jev-1.13.0"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
Transport = Callable[[urllib.request.Request, float], "tuple[int, bytes]"]


@dataclass(frozen=True)
class SystemOneResult:
    model: str
    answers: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def answer(self, question_id: str) -> dict[str, Any]:
        value = self.answers.get(question_id)
        return value if isinstance(value, dict) else {}


def _default_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (explicit http client)
            return int(getattr(response, "status", 200) or 200), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class SystemOneClient:
    """Transport-only client. Callers build questions and parse answers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        timeout: float = 2.0,
        retries: int = 1,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cleaned = str(base_url or "").strip().rstrip("/")
        if not cleaned:
            raise DecisionError("SystemOne base_url is required")
        self.base_url = cleaned
        self.api_key = str(api_key or "").strip()
        self.model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.timeout = max(0.0, float(timeout))
        self.retries = max(0, int(retries))
        self._transport = transport or _default_transport
        self._sleep = sleep

    def evaluate(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        model: str | None = None,
    ) -> SystemOneResult:
        if not isinstance(questions, dict) or not questions:
            raise DecisionError("questions must be a non-empty mapping")

        payload = {
            "state": state,
            "model": str(model or self.model),
            "questions": questions,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/v1/systemone"

        last_error: DecisionError | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                status, raw = self._transport(request, self.timeout)
            except TimeoutError as exc:
                last_error = DecisionTimeout(f"SystemOne request timed out after {self.timeout}s")
                if attempt < self.retries:
                    self._sleep(_backoff(attempt))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, TimeoutError):
                    last_error = DecisionTimeout(f"SystemOne request timed out after {self.timeout}s")
                else:
                    last_error = DecisionError(f"SystemOne transport error: {reason}")
                if attempt < self.retries:
                    self._sleep(_backoff(attempt))
                    continue
                raise last_error from exc

            latency_ms = (time.perf_counter() - started) * 1000.0

            if status in _RETRYABLE_STATUS:
                last_error = DecisionError(_message_from_body(raw) or f"SystemOne returned HTTP {status}")
                if attempt < self.retries:
                    self._sleep(_backoff(attempt))
                    continue
                raise last_error

            if status >= 400:
                detail = _message_from_body(raw) or f"HTTP {status}"
                raise DecisionError(f"SystemOne request failed ({status}): {detail}")

            return _parse_result(raw, latency_ms=latency_ms)

        raise last_error or DecisionError("SystemOne request failed")


def _parse_result(raw: bytes, *, latency_ms: float) -> SystemOneResult:
    try:
        payload = json.loads(raw.decode("utf-8") if raw else "{}")
    except (ValueError, UnicodeDecodeError) as exc:
        raise DecisionError(f"SystemOne returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecisionError("SystemOne returned a non-object response")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionError("SystemOne response is missing the 'answers' object")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return SystemOneResult(
        model=str(payload.get("model") or ""),
        answers=answers,
        input_tokens=_int_or_zero(usage.get("input_tokens")),
        output_tokens=_int_or_zero(usage.get("output_tokens")),
        latency_ms=float(latency_ms),
        raw=payload,
    )


def _message_from_body(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8") if raw else "{}")
    except (ValueError, UnicodeDecodeError):
        return ""
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _backoff(attempt: int) -> float:
    return min(0.25 * (2**attempt), 2.0)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["DEFAULT_MODEL", "SystemOneClient", "SystemOneResult"]
