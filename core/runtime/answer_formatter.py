"""Answer formatter layer.

The formatter does no retrieval and no reasoning: it takes the user's question
plus the raw evidence bundle produced by the retrieval-selection layer and emits
a clean, structured, user-facing answer. All cleaning (role prefixes, tool-call
noise, chat turns) is the formatter model's job — never the retrieval engine's.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]

_ENABLED_VALUES = {"1", "true", "yes", "on"}

FORMATTER_SYSTEM_PROMPT = """You are a formatter, not an analyst.

You receive a QUESTION and EVIDENCE taken from prior coding sessions. The
evidence is raw: it may contain chat turns ("User asked", "Assistant reported"),
tool output, code, and irrelevant text.

Your only job is to answer the QUESTION using the EVIDENCE and present it in a
clean, structured form. Rules:
- Output ONLY the answer to the question. Do not describe your process.
- Never mention or restate chat roles ("User asked", "Assistant reported",
  "Tool output"), tool calls, file listings, or the evidence itself.
- Never include reasoning, thinking, or meta commentary.
- Use short Markdown bullet points (or numbered points) and bold labels for
  distinct facts. Lead with the direct answer.
- Preserve exact identifiers, file names, numbers, and code values verbatim.
- If the evidence does not contain a requested part, add a single bullet
  "**Not found in prior sessions:** <what was missing>" instead of inventing it.
- Be concise. Do not pad with restatements of the question."""


def formatter_enabled() -> bool:
    return os.getenv("DEVENV_ANSWER_FORMATTER", "1").strip().lower() in _ENABLED_VALUES


def _format_sessions(bundle: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for session in bundle or ():
        session_id = str(session.get("session_id") or "")
        title = str(session.get("title") or "")
        workspace = str(session.get("workspace_path") or "")
        descriptor = " · ".join(part for part in (title, workspace) if part)
        lines.append(f"- {session_id}: {descriptor}" if descriptor else f"- {session_id}")
    return "\n".join(lines)


def build_formatter_messages(
    question: str,
    evidence: dict[str, Any],
) -> list[dict[str, str]]:
    bundle = evidence or {}
    lines = [str(line) for line in (bundle.get("lines") or []) if str(line).strip()]
    sessions = bundle.get("sessions") or []
    user_parts = [
        f"QUESTION:\n{question}",
        "",
        "EVIDENCE (raw; may contain chat/tool noise you must strip):",
    ]
    if lines:
        user_parts.extend(f"- {line}" for line in lines)
    else:
        user_parts.append("(no evidence retrieved)")
    if sessions:
        user_parts.extend(["", "SOURCE SESSIONS:", _format_sessions(sessions)])
    return [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


class AnswerFormatter:
    """Session-formatting layer driven by an injectable chat callable."""

    def __init__(self, chat: ChatFn, *, model: str = "") -> None:
        self._chat = chat
        self.model = model

    def format(
        self,
        question: str,
        evidence: dict[str, Any],
    ) -> str | None:
        if not question.strip():
            return None
        lines = (evidence or {}).get("lines") or []
        if not lines:
            return None
        messages = build_formatter_messages(question, evidence)
        try:
            content = self._chat(messages)
        except Exception as exc:  # pragma: no cover - backend failures degrade
            logger.warning("Answer formatter chat failed: error=%s", exc)
            return None
        cleaned = str(content or "").strip()
        return cleaned or None


def build_answer_formatter(ai: Any) -> AnswerFormatter:
    """Build a formatter bound to the answer model (never the selector model)."""

    def chat(messages: list[dict[str, str]]) -> str:
        response = ai.chat(messages)
        return getattr(response, "content", "") or ""

    return AnswerFormatter(chat)
