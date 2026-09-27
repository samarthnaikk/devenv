"""LLM-backed session selector: project-aware listwise reranking.

The selector is intentionally transport-agnostic: it takes a ``chat`` callable
``(messages) -> str`` so it can be driven by any backend (the user-chosen model).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from .session_selection import SessionCandidate, SelectionResult

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = """You select which past coding sessions answer a developer's query.

Rules:
- Prefer candidates whose project matches the CURRENT PROJECT.
- Only choose a different project when the user's query explicitly names that
  project, or no current-project candidate contains the answer.
- For every cross-project choice, give a short reason (under 12 words).
- Select only sessions that genuinely contain the answer. If none do, return an
  empty "selected" list.
- Never invent session ids; use only the ids listed in CANDIDATES.

Respond with a single JSON object and nothing else:
{
  "ordered": ["<session_id>", ...],
  "selected": ["<session_id>", ...],
  "confidence": 0.0,
  "need_more": false,
  "refined_query": "",
  "evidence": {"<session_id>": ["<verbatim line>", ...]},
  "reasons": {"<session_id>": "<short reason>"}
}"""


def project_name(workspace_path: str | None) -> str:
    if not workspace_path:
        return "unknown"
    return Path(str(workspace_path)).name or "unknown"


def build_selector_messages(
    task: str,
    candidates: Sequence[SessionCandidate],
    *,
    workspace_path: str | None,
) -> list[dict[str, str]]:
    current_project = project_name(workspace_path)
    lines = [
        f"CURRENT PROJECT: {current_project} (path: {workspace_path or 'unknown'})",
        f"USER QUERY: {task}",
        "",
        "CANDIDATES:",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"[{index}] id={candidate.session_id} | project={project_name(candidate.workspace_path)}"
            f" | title={candidate.title} | updated={candidate.updated_at} | score={candidate.score}"
        )
        if candidate.snippet:
            lines.append(f"    snippet: {candidate.snippet}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _extract_json(raw: str) -> object:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return payload


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


def parse_selection(raw: str, candidates: Sequence[SessionCandidate]) -> SelectionResult:
    payload = _extract_json(raw)
    known = {candidate.session_id for candidate in candidates}
    if not isinstance(payload, dict):
        return SelectionResult(
            session_ids=(),
            reason="selector returned unparseable output",
            degraded=True,
        )

    def known_ids(key: str) -> list[str]:
        ordered: list[str] = []
        for value in _string_list(payload.get(key)):
            if value in known and value not in ordered:
                ordered.append(value)
        return ordered

    ordered = known_ids("ordered")
    if "selected" in payload:
        selected = known_ids("selected")
    else:
        selected = ordered

    confidence = payload.get("confidence")
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = 0.0

    evidence: dict[str, list[str]] = {}
    raw_evidence = payload.get("evidence")
    if isinstance(raw_evidence, dict):
        for session_id, lines in raw_evidence.items():
            if session_id not in known:
                continue
            cleaned = [str(line)[:300] for line in _string_list(lines)][:8]
            if cleaned:
                evidence[session_id] = cleaned

    reasons: dict[str, str] = {}
    raw_reasons = payload.get("reasons")
    if isinstance(raw_reasons, dict):
        reasons = {
            str(key): str(value)[:200]
            for key, value in raw_reasons.items()
            if str(key) in known
        }

    refined_query = payload.get("refined_query")
    refined = str(refined_query).strip() if isinstance(refined_query, str) else ""

    return SelectionResult(
        session_ids=tuple(selected),
        confidence=confidence_value,
        need_more=bool(payload.get("need_more")),
        refined_query=refined or None,
        reason="; ".join(f"{sid}: {text}" for sid, text in reasons.items())[:400],
        evidence=evidence,
    )


class LLMSessionSelector:
    """SessionSelector implementation driven by a chat callable."""

    def __init__(self, chat: ChatFn, *, model: str = "") -> None:
        self._chat = chat
        self.model = model

    def select(
        self,
        task: str,
        candidates: Sequence[SessionCandidate],
        *,
        workspace_path: str,
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(session_ids=(), reason="no candidates")
        messages = build_selector_messages(task, candidates, workspace_path=workspace_path)
        try:
            raw = self._chat(messages)
        except Exception as exc:  # pragma: no cover - backend failures degrade gracefully
            logger.warning("Session selector chat failed: error=%s", exc)
            return SelectionResult(
                session_ids=(),
                reason=f"selector error: {type(exc).__name__}",
                degraded=True,
            )
        return parse_selection(raw, candidates)
