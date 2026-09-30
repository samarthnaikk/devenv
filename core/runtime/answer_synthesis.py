"""Multi-session answer synthesis.

The single-call formatter (`answer_formatter.AnswerFormatter`) rewrites one
evidence bundle into one answer. When evidence spans several sessions, this
module additionally answers *each* session separately and then reconciles the
per-session answers into one cited answer.

The per-session stage surfaces session-specific detail; the reconciliation stage
merges, resolves conflicts, and attributes each claim to its source session.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_MAX_SESSIONS = 3

PER_SESSION_SYSTEM_PROMPT = """You are answering a QUESTION using EVIDENCE taken from ONE prior coding session.

Rules:
- Answer only from the EVIDENCE for this session. Do not invent facts.
- If the evidence does not contain the answer, reply exactly: NO_RELEVANT_EVIDENCE
- Keep it tight: short Markdown bullets, bold labels, exact identifiers/code verbatim.
- Do not mention chat roles, tool calls, or that you were given evidence."""

RECONCILE_SYSTEM_PROMPT = """You merge several per-session answers into ONE final answer to the QUESTION.

Rules:
- Lead with the direct answer; use short Markdown bullets and bold labels.
- Preserve exact identifiers, file names, numbers, and code values verbatim.
- Attribute each grounded fact with a short source tag like (source: <session_id>) using the ids given.
- If sessions conflict, say so explicitly and prefer the most specific evidence.
- If a requested part is missing everywhere, add one bullet "**Not found in prior sessions:** <what>".
- Never mention chat roles, tool calls, or the evidence itself."""


def multisession_enabled() -> bool:
    return os.getenv("DEVENV_MULTISESSION_ANSWERS", "1").strip().lower() in _TRUE_VALUES


def max_sessions() -> int:
    raw = os.getenv("DEVENV_MULTISESSION_MAX_SESSIONS", str(DEFAULT_MAX_SESSIONS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_SESSIONS


def group_evidence_by_session(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{session_id: {"lines": [...], "session": {...}}}``.

    Prefers an explicit ``by_session`` mapping on the bundle; otherwise groups
    the flat ``lines`` under the first session so callers still get one group.
    """

    bundle = evidence or {}
    by_session = bundle.get("by_session")
    if isinstance(by_session, dict) and by_session:
        grouped: dict[str, dict[str, Any]] = {}
        sessions_by_id = {
            str(session.get("session_id") or ""): session
            for session in (bundle.get("sessions") or [])
            if isinstance(session, dict)
        }
        for session_id, lines in by_session.items():
            grouped[str(session_id)] = {
                "lines": [str(line) for line in (lines or []) if str(line).strip()],
                "session": sessions_by_id.get(str(session_id), {"session_id": str(session_id)}),
            }
        return grouped

    lines = [str(line) for line in (bundle.get("lines") or []) if str(line).strip()]
    sessions = [session for session in (bundle.get("sessions") or []) if isinstance(session, dict)]
    if not sessions or not lines:
        return {}
    first = sessions[0]
    return {
        str(first.get("session_id") or "session"): {
            "lines": lines,
            "session": first,
        }
    }


def _format_session_evidence(session: dict[str, Any], lines: list[str]) -> str:
    title = str(session.get("title") or "")
    workspace = str(session.get("workspace_path") or "")
    descriptor = " · ".join(part for part in (title, workspace) if part)
    header = f"SESSION {session.get('session_id', '')}"
    if descriptor:
        header = f"{header} ({descriptor})"
    return "\n".join([header, *(f"- {line}" for line in lines)])


class MultiSessionAnswer:
    """Answer each top session, then reconcile into one cited answer."""

    def __init__(
        self,
        chat: Callable[[list[dict[str, str]]], str],
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self._chat = chat
        self.max_sessions = max(1, int(max_sessions))

    def synthesize(self, question: str, evidence: dict[str, Any]) -> str | None:
        if not question.strip():
            return None
        groups = group_evidence_by_session(evidence)
        if not groups:
            return None
        ranked = sorted(groups.items(), key=lambda item: -len(item[1]["lines"]))[: self.max_sessions]

        per_session: list[tuple[str, str]] = []
        for session_id, group in ranked:
            answer = self._answer_session(question, group)
            if answer and answer.strip() and answer.strip().upper() != "NO_RELEVANT_EVIDENCE":
                per_session.append((session_id, answer.strip()))

        if not per_session:
            return None
        if len(per_session) == 1:
            # Nothing to reconcile; still tag the single source.
            session_id, answer = per_session[0]
            return f"{answer}\n\n_Source: {session_id}_"
        return self._reconcile(question, per_session)

    def _answer_session(self, question: str, group: dict[str, Any]) -> str | None:
        block = _format_session_evidence(group["session"], group["lines"])
        messages = [
            {"role": "system", "content": PER_SESSION_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nEVIDENCE:\n{block}"},
        ]
        try:
            return str(self._chat(messages) or "")
        except Exception as exc:  # pragma: no cover - backend failures degrade
            logger.warning("Per-session answer failed: error=%s", exc)
            return None

    def _reconcile(self, question: str, per_session: list[tuple[str, str]]) -> str | None:
        parts = ["PER-SESSION ANSWERS:"]
        for session_id, answer in per_session:
            parts.append(f"\n[{session_id}]\n{answer}")
        parts.append("\nSOURCE SESSION IDS: " + ", ".join(session_id for session_id, _ in per_session))
        messages = [
            {"role": "system", "content": RECONCILE_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\n" + "\n".join(parts)},
        ]
        try:
            reconciled = str(self._chat(messages) or "").strip()
        except Exception as exc:  # pragma: no cover - backend failures degrade
            logger.warning("Reconciliation failed: error=%s", exc)
            return None
        if not reconciled:
            return None
        return reconciled


def build_multisession_answerer(chat: Callable[[list[dict[str, str]]], str]) -> MultiSessionAnswer:
    return MultiSessionAnswer(chat, max_sessions=max_sessions())


__all__ = [
    "MultiSessionAnswer",
    "build_multisession_answerer",
    "group_evidence_by_session",
    "multisession_enabled",
    "max_sessions",
]
