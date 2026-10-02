"""Durable, tamper-evident audit trail for runtime decisions.

Every recorded event is appended to both:

* an append-only JSONL file under ``<workspace>/.devenv/audit/audit-YYYYMMDD.jsonl``
* a queryable ``runtime_events`` row in ``memory.db``

Each event carries ``prev_hash`` and ``hash`` (sha256 over the canonical JSON of
the event without ``hash``). The chain makes silent edits/deletions detectable
via :func:`verify_audit_chain`. Chaining and file writing are opt-out through
``DEVENV_AUDIT_CHAIN`` / ``DEVENV_AUDIT_TO_FILE``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Event types emitted by the runtime.
TURN_START = "turn.start"
TURN_END = "turn.end"
TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
POLICY_DECISION = "policy.decision"
SANDBOX_VIOLATION = "sandbox.violation"
RETRIEVAL_TRACE = "retrieval.trace"
VERIFICATION = "verification.result"
BACKEND_ERROR = "backend.error"
DECISION_RESULT = "decision.result"


def audit_enabled() -> bool:
    return os.getenv("DEVENV_AUDIT", "1").strip().lower() in _TRUE_VALUES


def _chain_enabled() -> bool:
    return os.getenv("DEVENV_AUDIT_CHAIN", "1").strip().lower() in _TRUE_VALUES


def _file_enabled() -> bool:
    return os.getenv("DEVENV_AUDIT_TO_FILE", "1").strip().lower() in _TRUE_VALUES


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_event(event_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(event_without_hash).encode("utf-8")).hexdigest()


@dataclass
class AuditRecorder:
    """Appends audit events to disk and/or the runtime_events table."""

    workspace_path: str
    store: Any = None
    turn_id: str = ""
    session_id: str = ""
    _prev_hash: str = field(default="", init=False)
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.workspace_path = str(Path(self.workspace_path).expanduser().resolve())
        if self.store is not None:
            try:
                self._prev_hash = self.store.last_runtime_event_hash()
            except Exception:  # pragma: no cover - defensive
                self._prev_hash = ""

    @property
    def enabled(self) -> bool:
        return audit_enabled()

    def _audit_dir(self) -> Path:
        return Path(self.workspace_path) / ".devenv" / "audit"

    def record(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        backend: str = "",
        model: str = "",
        turn_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        event: dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "ts": time.time(),
            "turn_id": turn_id if turn_id is not None else self.turn_id,
            "session_id": session_id if session_id is not None else self.session_id,
            "workspace": self.workspace_path,
            "event_type": str(event_type),
            "backend": str(backend or ""),
            "model": str(model or ""),
            "payload_json": _canonical(payload or {}),
        }
        if _chain_enabled():
            event["prev_hash"] = self._prev_hash
            event["hash"] = _hash_event(event)
            self._prev_hash = event["hash"]
        else:
            event["prev_hash"] = ""
            event["hash"] = ""
        self._seq += 1

        if _file_enabled():
            self._write_file(event)
        if self.store is not None:
            try:
                self.store.append_runtime_event(event)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Audit DB append failed: error=%s", exc)
        return event

    def _write_file(self, event: dict[str, Any]) -> None:
        try:
            audit_dir = self._audit_dir()
            audit_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%d")
            path = audit_dir / f"audit-{stamp}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(event) + "\n")
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Audit file append failed: error=%s", exc)

    @staticmethod
    def verify_chain(events: list[dict[str, Any]]) -> tuple[bool, str]:
        """Verify a hash chain over ``events`` in ascending sequence order."""

        previous = ""
        for event in events:
            if not event.get("hash"):
                continue
            candidate = dict(event)
            candidate.pop("hash", None)
            candidate.pop("seq", None)
            if candidate.get("prev_hash", "") != previous:
                return False, f"broken link at event {event.get('event_id')}"
            if _hash_event(candidate) != event.get("hash"):
                return False, f"hash mismatch at event {event.get('event_id')}"
            previous = event["hash"]
        return True, "ok"


def build_recorder(workspace_path: str, store: Any = None) -> AuditRecorder:
    return AuditRecorder(workspace_path=workspace_path, store=store)


def audit_file_reports(workspace_path: str) -> list[Path]:
    directory = Path(workspace_path).expanduser() / ".devenv" / "audit"
    if not directory.exists():
        return []
    return sorted(directory.glob("audit-*.jsonl"))


def read_audit_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return events


def prune_audit_files(workspace_path: str, *, retention_days: int) -> int:
    """Delete audit JSONL files older than ``retention_days``. Returns count removed."""

    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for path in audit_file_reports(workspace_path):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


__all__ = [
    "AuditRecorder",
    "build_recorder",
    "audit_enabled",
    "TURN_START",
    "TURN_END",
    "TOOL_CALL",
    "TOOL_RESULT",
    "POLICY_DECISION",
    "SANDBOX_VIOLATION",
    "RETRIEVAL_TRACE",
    "VERIFICATION",
    "BACKEND_ERROR",
    "DECISION_RESULT",
    "audit_file_reports",
    "read_audit_file",
    "prune_audit_files",
]
