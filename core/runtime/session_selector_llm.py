"""LLM-backed session selector: project-aware listwise reranking.

The selector is intentionally transport-agnostic: it takes a ``chat`` callable
``(messages) -> str`` so it can be driven by any backend (the user-chosen model).
"""

from __future__ import annotations

import json
import logging
import math
import random
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
- The snippet lines shown for each candidate are the passages the retriever
  matched; base your decision and evidence on them, not on the title alone.
- If the query has several parts, make sure the chosen session(s) and evidence
  cover every part; quote the exact lines for each part.
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
        ordered=tuple(ordered),
    )


class LLMSessionSelector:
    """SessionSelector implementation driven by a chat callable.

    ``permutations > 1`` runs the listwise prompt over shuffled candidate orders
    and aggregates the selections. A session must be chosen by a majority of the
    permutations to survive, which defends against listwise position bias at the
    cost of extra model calls.
    """

    def __init__(
        self,
        chat: ChatFn,
        *,
        model: str = "",
        permutations: int = 1,
        rng: random.Random | None = None,
    ) -> None:
        self._chat = chat
        self.model = model
        self._permutations = max(1, int(permutations))
        self._rng = rng or random.Random()

    def select(
        self,
        task: str,
        candidates: Sequence[SessionCandidate],
        *,
        workspace_path: str,
    ) -> SelectionResult:
        ordered_candidates = list(candidates)
        if not ordered_candidates:
            return SelectionResult(session_ids=(), reason="no candidates")
        if self._permutations <= 1:
            return self._select_single(task, ordered_candidates, workspace_path)
        return self._select_permuted(task, ordered_candidates, workspace_path)

    def _select_single(
        self,
        task: str,
        candidates: list[SessionCandidate],
        workspace_path: str,
    ) -> SelectionResult:
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

    def _select_permuted(
        self,
        task: str,
        candidates: list[SessionCandidate],
        workspace_path: str,
    ) -> SelectionResult:
        votes: dict[str, int] = {}
        ranks: dict[str, list[int]] = {}
        evidence: dict[str, list[str]] = {}
        confidences: list[float] = []
        reasons: list[str] = []
        need_more = False
        refined: str | None = None
        successes = 0

        for _ in range(self._permutations):
            order = list(candidates)
            self._rng.shuffle(order)
            messages = build_selector_messages(task, order, workspace_path=workspace_path)
            try:
                raw = self._chat(messages)
            except Exception as exc:  # pragma: no cover - backend failures
                logger.warning("Session selector permutation failed: error=%s", exc)
                continue
            result = parse_selection(raw, order)
            if result.degraded:
                continue
            successes += 1
            confidences.append(result.confidence)
            for session_id in result.session_ids:
                votes[session_id] = votes.get(session_id, 0) + 1
            for rank, session_id in enumerate(result.ordered or result.session_ids, start=1):
                ranks.setdefault(session_id, []).append(rank)
            for session_id, lines in result.evidence.items():
                evidence.setdefault(session_id, lines)
            need_more = need_more or result.need_more
            refined = refined or result.refined_query
            if result.reason:
                reasons.append(result.reason)

        if successes == 0:
            return SelectionResult(
                session_ids=(),
                reason="selector error across permutations",
                degraded=True,
            )

        threshold = max(1, math.ceil(successes / 2))
        chosen = [session_id for session_id, count in votes.items() if count >= threshold]

        def average_rank(session_id: str) -> float:
            session_ranks = ranks.get(session_id)
            if not session_ranks:
                return float(10**6)
            return sum(session_ranks) / len(session_ranks)

        chosen.sort(key=average_rank)
        confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        return SelectionResult(
            session_ids=tuple(chosen),
            confidence=confidence,
            need_more=need_more,
            refined_query=refined,
            reason="; ".join(reasons)[:400],
            evidence=evidence,
        )
