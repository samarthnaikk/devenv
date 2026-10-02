"""Typed System One question builders and answer parsers.

Each builder returns the ``questions`` mapping for the ``/v1/systemone``
contract; each parser turns the returned ``answers`` back into a small typed
result the gate can act on. Keep questions atomic and the answer space closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import DecisionError, WriteDecision

INTENT_ROUTES = ("recall", "explain", "inspect", "mutate", "meta")
_LOCAL_KNOWLEDGE_ROUTES = frozenset({"recall", "explain"})

MEMORY_WRITE_INSTRUCTIONS = (
    "Does `candidate` contain a durable fact, preference, or decision that will be "
    "useful in future sessions? Ignore transient status, greetings, restatements of "
    "code already visible in the workspace, and low-value detail."
)


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


def build_memory_write_questions(candidate_texts: Sequence[str]) -> dict[str, Any]:
    """One ``noul`` per candidate, each carrying its own candidate in instructions.

    Candidates are embedded per-question so the model evaluates them in isolation
    (no cross-contamination between candidates in the same batch).
    """

    questions: dict[str, Any] = {}
    for index, text in enumerate(candidate_texts):
        questions[f"write_{index}"] = {
            "type": "noul",
            "instructions": {
                "candidate": str(text or ""),
                "question": MEMORY_WRITE_INSTRUCTIONS,
            },
            "criteria": {
                "true": "Durable and likely useful in a later session",
                "false": "Transient, redundant, or low value",
            },
        }
    return questions


def parse_memory_write_answers(answers: dict[str, Any], count: int) -> list[WriteDecision]:
    """Parse one Noul per candidate. Missing/invalid answers fail open (keep)."""

    decisions: list[WriteDecision] = []
    mapping = answers if isinstance(answers, dict) else {}
    for index in range(count):
        answer = mapping.get(f"write_{index}")
        noul = _optional_float(answer.get("noul")) if isinstance(answer, dict) else None
        if noul is None:
            decisions.append(
                WriteDecision(
                    should_write=True,
                    confidence=1.0,
                    reason="missing memory-write answer; kept",
                    source="heuristic",
                )
            )
            continue
        decisions.append(
            WriteDecision(
                should_write=noul >= 0.5,
                confidence=abs(noul - 0.5) * 2.0,
                reason=f"systemone:noul={noul:.3f}",
                source="systemone",
            )
        )
    return decisions


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "INTENT_ROUTES",
    "IntentResult",
    "MEMORY_WRITE_INSTRUCTIONS",
    "build_intent_questions",
    "build_memory_write_questions",
    "parse_intent_answers",
    "parse_memory_write_answers",
]
