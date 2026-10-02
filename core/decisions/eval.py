"""Evaluation helpers for the decision gates.

Loads JSONL corpora, runs the heuristic and/or System One providers over them,
and computes accuracy / precision / recall / F1 and expected calibration error
per gate. Used by the ``devenv-decisions`` CLI and its regression gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import GATE_INTENT, GATE_MEMORY_WRITE, DecisionConfig
from .models import DecisionError, DecisionRecord
from .provider import HeuristicDecisionProvider, build_decision_provider

DOMAINS = ("intent", "memory")


@dataclass(frozen=True)
class MetricResult:
    total: int
    correct: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    ece: float | None = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "errors": self.errors,
        }
        if self.ece is not None:
            payload["ece"] = round(self.ece, 4)
        return payload


@dataclass
class EvalOutcome:
    domain: str
    provider: str
    metrics: MetricResult
    records: list[DecisionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "provider": self.provider,
            "metrics": self.metrics.to_dict(),
        }


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            items.append(payload)
    return items


def evaluate_intent(
    corpus: list[dict[str, Any]],
    *,
    config: DecisionConfig,
    recorder=None,
) -> EvalOutcome:
    provider = _select_provider(config)
    provider_name = str(getattr(provider, "provider_name", "heuristic"))
    truths: list[bool] = []
    preds: list[bool] = []
    confidences: list[float] = []
    latencies: list[float] = []
    records: list[DecisionRecord] = []
    errors = 0

    for item in corpus:
        prompt = str(item.get("prompt", ""))
        expected = bool(item.get("expected_use_local_knowledge", False))
        started = _now()
        try:
            decision = provider.decide(prompt)
            fallback = False
            error = ""
        except DecisionError as exc:
            decision = HeuristicDecisionProvider().decide(prompt)
            fallback = True
            error = str(exc)
            errors += 1
        latencies.append(_ms(started))
        truths.append(expected)
        preds.append(bool(decision.use_local_knowledge))
        confidences.append(float(decision.confidence))
        records.append(
            DecisionRecord(
                gate=GATE_INTENT,
                domain="intent",
                provider=provider_name,
                mode="eval",
                answer={"use_local_knowledge": bool(decision.use_local_knowledge), **({"error": error} if error else {})},
                confidence=float(decision.confidence),
                reason=decision.reason,
                latency_ms=latencies[-1],
                fallback=fallback,
            )
        )

    metrics = _metrics(truths, preds, confidences, latencies, errors=errors)
    return EvalOutcome(domain="intent", provider=provider_name, metrics=metrics, records=records)


def evaluate_memory(
    corpus: list[dict[str, Any]],
    *,
    config: DecisionConfig,
    recorder=None,
) -> EvalOutcome:
    provider = _select_provider(config)
    provider_name = str(getattr(provider, "provider_name", "heuristic"))
    truths: list[bool] = []
    preds: list[bool] = []
    confidences: list[float] = []
    latencies: list[float] = []
    records: list[DecisionRecord] = []
    errors = 0

    for item in corpus:
        label = str(item.get("label", ""))
        summary = str(item.get("summary", ""))
        expected = bool(item.get("expected_write", True))
        candidate = type("Candidate", (), {"label": label, "category": str(item.get("category", "component")), "summary": summary})()
        started = _now()
        try:
            decisions = provider.judge_writes([candidate], [], context={"source": "eval"})
            decision = decisions[0]
            fallback = decision.source != "systemone"
        except DecisionError as exc:
            decision = HeuristicDecisionProvider().judge_writes([candidate], [])[0]
            fallback = True
            errors += 1
        latencies.append(_ms(started))
        truths.append(expected)
        preds.append(bool(decision.should_write))
        confidences.append(float(decision.confidence))
        records.append(
            DecisionRecord(
                gate=GATE_MEMORY_WRITE,
                domain="memory",
                provider=provider_name,
                mode="eval",
                answer={"should_write": bool(decision.should_write), "label": label},
                confidence=float(decision.confidence),
                reason=decision.reason,
                latency_ms=latencies[-1],
                fallback=fallback,
            )
        )

    metrics = _metrics(truths, preds, confidences, latencies, errors=errors)
    return EvalOutcome(domain="memory", provider=provider_name, metrics=metrics, records=records)


def evaluate(
    domain: str,
    corpus: list[dict[str, Any]],
    *,
    config: DecisionConfig,
    recorder=None,
) -> EvalOutcome:
    if domain == "intent":
        return evaluate_intent(corpus, config=config, recorder=recorder)
    if domain == "memory":
        return evaluate_memory(corpus, config=config, recorder=recorder)
    raise DecisionError(f"unknown eval domain: {domain}")


def compare_to_baseline(outcome: EvalOutcome, baseline: dict[str, Any], *, tolerance: float = 0.0) -> tuple[bool, str]:
    """Return (ok, detail). ``ok`` is False when accuracy regresses beyond tolerance."""

    baseline_metrics = (baseline or {}).get("metrics", {})
    baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0))
    current = outcome.metrics.accuracy
    if current + tolerance < baseline_accuracy:
        return False, f"{outcome.domain}: accuracy {current:.4f} < baseline {baseline_accuracy:.4f} (tolerance {tolerance})"
    return True, f"{outcome.domain}: accuracy {current:.4f} >= baseline {baseline_accuracy:.4f}"


def _select_provider(config: DecisionConfig):
    provider = build_decision_provider(config, offline=False)
    return provider


def _metrics(
    truths: list[bool],
    preds: list[bool],
    confidences: list[float],
    latencies: list[float],
    *,
    errors: int,
) -> MetricResult:
    total = len(truths)
    if total == 0:
        return MetricResult(0, 0, 0.0, 0.0, 0.0, 0.0)
    correct = sum(1 for truth, pred in zip(truths, preds) if truth == pred)
    tp = sum(1 for truth, pred in zip(truths, preds) if truth and pred)
    fp = sum(1 for truth, pred in zip(truths, preds) if not truth and pred)
    fn = sum(1 for truth, pred in zip(truths, preds) if truth and not pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return MetricResult(
        total=total,
        correct=correct,
        accuracy=correct / total,
        precision=precision,
        recall=recall,
        f1=f1,
        ece=_ece(truths, preds, confidences),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        errors=errors,
    )


def _ece(truths: list[bool], preds: list[bool], confidences: list[float], bins: int = 10) -> float:
    if not truths:
        return 0.0
    bucket_totals = [0] * bins
    bucket_correct = [0] * bins
    bucket_conf = [0.0] * bins
    for truth, pred, confidence in zip(truths, preds, confidences):
        clamped = min(max(float(confidence), 0.0), 1.0)
        index = min(int(clamped * bins), bins - 1)
        bucket_totals[index] += 1
        bucket_conf[index] += clamped
        bucket_correct[index] += 1 if truth == pred else 0
    total = len(truths)
    ece = 0.0
    for index in range(bins):
        if bucket_totals[index] == 0:
            continue
        avg_conf = bucket_conf[index] / bucket_totals[index]
        avg_acc = bucket_correct[index] / bucket_totals[index]
        ece += (bucket_totals[index] / total) * abs(avg_conf - avg_acc)
    return ece


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _now() -> float:
    import time

    return time.perf_counter()


def _ms(started: float) -> float:
    import time

    return (time.perf_counter() - started) * 1000.0


# Default corpus locations shipped with the repo.
def default_corpus_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "decisions" / "corpora"


def default_baseline_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "decisions" / "baselines"


__all__ = [
    "DOMAINS",
    "EvalOutcome",
    "MetricResult",
    "compare_to_baseline",
    "default_baseline_dir",
    "default_corpus_dir",
    "evaluate",
    "evaluate_intent",
    "evaluate_memory",
    "load_corpus",
]
