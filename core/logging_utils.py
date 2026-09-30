"""Centralized logging configuration for Devenv.

The runtime logs to stderr by default and, when a workspace is known, also to a
rotating file under ``<workspace>/.devenv/logs/devenv.log``. A context filter
stamps each record with the active turn/session/backend/workspace so a single
user turn can be reconstructed across the ``core.*`` loggers. Set
``DEVENV_LOG_JSON=1`` for one-JSON-object-per-line output.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONTEXT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s [turn=%(turn_id)s backend=%(backend)s]: %(message)s"
)

# Context variables populated at turn boundaries. They default to "-" so the
# format never raises on records emitted outside a turn.
turn_id_var: ContextVar[str] = ContextVar("devenv_turn_id", default="-")
session_id_var: ContextVar[str] = ContextVar("devenv_session_id", default="-")
backend_var: ContextVar[str] = ContextVar("devenv_backend", default="-")
workspace_var: ContextVar[str] = ContextVar("devenv_workspace", default="-")

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def set_log_context(
    *,
    turn_id: str | None = None,
    session_id: str | None = None,
    backend: str | None = None,
    workspace: str | None = None,
) -> None:
    """Set the active log context (unset fields are left unchanged)."""

    if turn_id is not None:
        turn_id_var.set(turn_id)
    if session_id is not None:
        session_id_var.set(session_id)
    if backend is not None:
        backend_var.set(backend)
    if workspace is not None:
        workspace_var.set(workspace)


def clear_log_context() -> None:
    turn_id_var.set("-")
    session_id_var.set("-")
    backend_var.set("-")
    workspace_var.set("-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.turn_id = turn_id_var.get()
        record.session_id = session_id_var.get()
        record.backend = backend_var.get()
        record.workspace = workspace_var.get()
        return True


_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[=:]\s*\S+"),
        r"\1=<redacted>",
    ),
    (re.compile(r"(?i)bearer\s+[a-z0-9._-]+"), "Bearer <redacted>"),
    (re.compile(r"sk-[a-zA-Z0-9]{8,}"), "<redacted>"),
)


def redact_text(text: str, *, limit: int = 2000) -> str:
    """Redact common secret shapes and truncate long payloads for logs."""

    value = str(text or "")
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    if limit and len(value) > limit:
        value = f"{value[: limit - 3].rstrip()}..."
    return value


class RedactingFilter(logging.Filter):
    """Redact secrets/size from message and args before formatting."""

    def __init__(self, *, redact: bool = True, limit: int = 2000) -> None:
        super().__init__()
        self.redact = redact
        self.limit = limit

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.redact:
            return True
        try:
            # Resolve the final message first, then redact/truncate it, so the
            # formatting placeholders are never corrupted by rewriting `msg`.
            rendered = record.getMessage()
            redacted = redact_text(rendered, limit=self.limit)
            if redacted != rendered:
                record.msg = redacted
                record.args = ()
        except Exception:  # pragma: no cover - never let logging break the app
            return True
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "turn_id": getattr(record, "turn_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "backend": getattr(record, "backend", "-"),
            "workspace": getattr(record, "workspace", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _log_file_path(workspace: str | None) -> Path | None:
    explicit = os.getenv("DEVENV_LOG_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if not _env_bool("DEVENV_LOG_TO_FILE", True):
        return None
    # Only write a file when a workspace is actually known. Falling back to
    # os.getcwd() would create stray .devenv/logs directories in fixture trees
    # and library consumers that never configured a workspace.
    resolved_workspace = workspace or os.getenv("DEVENV_WORKSPACE", "").strip()
    if not resolved_workspace:
        return None
    try:
        return Path(resolved_workspace).expanduser() / ".devenv" / "logs" / "devenv.log"
    except Exception:  # pragma: no cover - defensive
        return None


def configure_logging(
    level: str | None = None,
    *,
    workspace: str | None = None,
    force: bool = False,
) -> None:
    resolved_level = (level or os.getenv("DEVENV_LOG_LEVEL") or "INFO").upper()
    levelno = getattr(logging, resolved_level, logging.INFO)
    use_json = _env_bool("DEVENV_LOG_JSON", False)
    context_filter = ContextFilter()
    redacting_filter = RedactingFilter(redact=_env_bool("DEVENV_LOG_REDACT", True))

    def _build_formatter() -> logging.Formatter:
        if use_json:
            return JsonFormatter()
        return logging.Formatter(CONTEXT_LOG_FORMAT)

    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler()
    stream.setFormatter(_build_formatter())
    stream.addFilter(context_filter)
    stream.addFilter(redacting_filter)
    handlers.append(stream)

    log_path = _log_file_path(workspace)
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            max_bytes = int(os.getenv("DEVENV_LOG_MAX_BYTES", str(2 * 1024 * 1024)))
            backups = int(os.getenv("DEVENV_LOG_BACKUPS", "3"))
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
            file_handler.setFormatter(_build_formatter())
            file_handler.addFilter(context_filter)
            file_handler.addFilter(redacting_filter)
            handlers.append(file_handler)
        except (OSError, ValueError):  # pragma: no cover - defensive
            pass

    root = logging.getLogger()
    if force or not getattr(root, "_devenv_configured", False):
        for existing in list(root.handlers):
            root.removeHandler(existing)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(levelno)
        root._devenv_configured = True  # type: ignore[attr-defined]


__all__ = [
    "LOG_FORMAT",
    "CONTEXT_LOG_FORMAT",
    "ContextFilter",
    "RedactingFilter",
    "JsonFormatter",
    "configure_logging",
    "set_log_context",
    "clear_log_context",
    "redact_text",
    "turn_id_var",
    "session_id_var",
    "backend_var",
    "workspace_var",
]
