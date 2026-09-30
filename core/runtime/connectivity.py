"""Internet-connectivity detection for offline auto-mode.

Devenv prefers local backends when there is no internet. Detection is a cheap,
cached DNS/HTTPS probe with explicit overrides:

- ``DEVENV_FORCE_OFFLINE=1`` — always treat as offline
- ``DEVENV_FORCE_ONLINE=1``  — always treat as online (skip the probe)
- ``DEVENV_CONNECTIVITY_TTL`` — cache seconds (default 60)
- ``DEVENV_CONNECTIVITY_HOSTS`` — comma-separated hosts to probe
"""

from __future__ import annotations

import logging
import os
import socket
import ssl
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_HOSTS = ("api.openai.com", "github.com")
DEFAULT_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 2.0

# (available, checked_at)
_CACHE: tuple[bool, float] | None = None


def _force_offline() -> bool:
    return os.getenv("DEVENV_FORCE_OFFLINE", "").strip().lower() in _TRUE_VALUES


def _force_online() -> bool:
    return os.getenv("DEVENV_FORCE_ONLINE", "").strip().lower() in _TRUE_VALUES


def _ttl_seconds() -> float:
    raw = os.getenv("DEVENV_CONNECTIVITY_TTL", "").strip()
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_TTL_SECONDS
    except ValueError:
        return DEFAULT_TTL_SECONDS


def _probe_hosts() -> tuple[str, ...]:
    raw = os.getenv("DEVENV_CONNECTIVITY_HOSTS", "").strip()
    if not raw:
        return DEFAULT_HOSTS
    hosts = tuple(host.strip() for host in raw.split(",") if host.strip())
    return hosts or DEFAULT_HOSTS


def _reachable(host: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    target = host if "://" in host else f"https://{host}"
    parsed = urlparse(target)
    hostname = parsed.hostname
    if not hostname:
        return False
    port = parsed.port or (443 if parsed.scheme != "http" else 80)
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            pass
        return True
    except (OSError, ValueError):
        pass
    # Fall back to DNS resolution, which still indicates some connectivity.
    try:
        socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        return True
    except (socket.gaierror, OSError):
        return False


def internet_available(*, refresh: bool = False, ttl: float | None = None) -> bool:
    """Return True when the machine appears to have internet access.

    Result is cached for ``DEVENV_CONNECTIVITY_TTL`` seconds. Overrides
    ``DEVENV_FORCE_OFFLINE`` / ``DEVENV_FORCE_ONLINE`` take precedence.
    """

    if _force_offline():
        return False
    if _force_online():
        return True
    global _CACHE
    now = time.monotonic()
    resolved_ttl = _ttl_seconds() if ttl is None else max(0.0, float(ttl))
    if not refresh and _CACHE is not None and (now - _CACHE[1]) < resolved_ttl:
        return _CACHE[0]
    available = any(_reachable(host) for host in _probe_hosts())
    _CACHE = (available, now)
    return available


def is_offline(*, refresh: bool = False, ttl: float | None = None) -> bool:
    return not internet_available(refresh=refresh, ttl=ttl)


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


def local_backend_priority() -> tuple[str, ...]:
    """Ordered local backends to prefer when offline."""

    raw = os.getenv("DEVENV_OFFLINE_BACKENDS", "ollama,llama_cpp").strip()
    order = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    return order or ("ollama", "llama_cpp")


__all__ = [
    "internet_available",
    "is_offline",
    "clear_cache",
    "local_backend_priority",
]
