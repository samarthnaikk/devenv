"""Redaction and a conservative send/deny check for hosted decision providers.

Devenv is local-first. A remote decision provider must never receive secrets or
private key material. :func:`prepare_remote_state` redacts known secret shapes
and truncates; :func:`looks_sensitive` is a conservative deny signal so callers
can fall back to the heuristic instead of sending.
"""

from __future__ import annotations

import re

from core.logging_utils import redact_text

_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|access[_-]?token|refresh[_-]?token|client[_-]?secret)\b\s*[=:]\s*\S+"
)
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{40,}\b")

DEFAULT_LIMIT = 4000


def redact_state(text: str, *, limit: int = DEFAULT_LIMIT) -> str:
    """Redact common secret shapes from ``text`` and truncate for transport."""

    return redact_text(str(text or ""), limit=limit)


def looks_sensitive(text: str) -> bool:
    """Return True when ``text`` carries material we refuse to send remotely."""

    value = str(text or "")
    if not value:
        return False
    if _PRIVATE_KEY_RE.search(value):
        return True
    if _SECRET_ASSIGNMENT_RE.search(value):
        return True
    if _LONG_TOKEN_RE.search(value):
        return True
    return False


def prepare_remote_state(
    text: str,
    *,
    redact: bool = True,
    limit: int = DEFAULT_LIMIT,
    allow_sensitive: bool = False,
) -> str | None:
    """Return the state to send remotely, or None if it must not be sent.

    When ``redact`` is False the payload is only truncated. When the state looks
    sensitive and ``allow_sensitive`` is False, ``None`` is returned so the
    caller can fall back to a local/heuristic decision.
    """

    value = str(text or "")
    if not allow_sensitive and looks_sensitive(value):
        return None
    if redact:
        return redact_state(value, limit=limit)
    return value[:limit]


__all__ = [
    "DEFAULT_LIMIT",
    "looks_sensitive",
    "prepare_remote_state",
    "redact_state",
]
