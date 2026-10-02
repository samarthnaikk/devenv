"""Typed System One question builders and answer parsers.

Each builder returns the ``questions`` mapping for the ``/v1/systemone``
contract; each parser turns the returned ``answers`` back into a small typed
result the gate can act on. Keep questions atomic and the answer space closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import DecisionError

INTENT_ROUTES = ("recall", "explain", "inspect", "mutate", "meta")
_LOCAL_KNOWLEDGE_ROUTES = frozenset({"recall", "explain"})


def build_intent_questions() -> dict[str, Any]:
    """A ``choice`` over the routes Devenv's kernel supports plus a mutation Noul."""

    return {
        "route": {
            "type": "choice",
            "instructions": "What does the user want from this turn?",
            "criteria": {
                "recall": "Answer from stored memory or prior sessions; no workspace change",
                "explain": "Explain existing code or architecture; no workspace change",
                "inspect": "Look up files, symbols, or text without changing them",
                "mutate": "Create, edit, delete files, or run commands that change the workspace",
                "meta": "Configure the agent, backend, model, permissions, or session tags",
            },
        },
        "mutates_workspace": {
            "type": "noul",
            "instructions": (
                "Will satisfying this request require writing to the workspace or running "
                "a command that changes it?"
            ),
            "criteria": {
                "true": "Requires a workspace mutation or a mutating command",
                "false": "Read-only; no workspace mutation needed",
            },
        },
    }


@dataclass(frozen=True)
class IntentResult:
    route: str
    confidence: float
    mutates_workspace: float
    reason: str

    @property
    def use_local_knowledge(self) -> bool:
        return self.route in _LOCAL_KNOWLEDGE_ROUTES and self.mutates_workspace < 0.5


def parse_intent_answers(answers: dict[str, Any]) -> IntentResult:
    if not isinstance(answers, dict):
        raise DecisionError("intent answers must be a mapping")
    route_answer = answers.get("route")
    if not isinstance(route_answer, dict):
        raise DecisionError("intent answers are missing the 'route' answer")
    route = str(route_answer.get("choice") or "").strip().lower()
    if not route:
        raise DecisionError("intent 'route' answer has no choice")
    confidence = _as_float(route_answer.get("confidence"), default=0.0)

    mutation_answer = answers.get("mutates_workspace")
    if isinstance(mutation_answer, dict):
        mutates_workspace = _as_float(mutation_answer.get("noul"), default=0.0)
    else:
        mutates_workspace = 0.0

    return IntentResult(
        route=route,
        confidence=confidence,
        mutates_workspace=mutates_workspace,
        reason=f"systemone:{route}",
    )


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "INTENT_ROUTES",
    "IntentResult",
    "build_intent_questions",
    "parse_intent_answers",
]
