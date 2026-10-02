from __future__ import annotations

import unittest

from core.decisions.memory import DecisionConsolidationExtractor
from core.decisions.models import DecisionError, DecisionMode, DecisionRecord, WriteDecision
from core.memory.models import (
    ConsolidationEntity,
    ConsolidationExtraction,
    ConsolidationUpdate,
)


def _extraction() -> ConsolidationExtraction:
    return ConsolidationExtraction(
        detected_project="RxGPT",
        new_entities=(
            ConsolidationEntity(label="Django Auth", category="component", summary="durable"),
            ConsolidationEntity(label="Greeting", category="noise", summary="hello"),
        ),
        updates_to_existing_nodes=(ConsolidationUpdate(node_id="proj_rxgpt", append_summary="more"),),
    )


class _FakeInner:
    def __init__(self, extraction: ConsolidationExtraction) -> None:
        self.extraction = extraction
        self.calls = 0

    def extract(self, logs, existing_nodes):
        self.calls += 1
        return self.extraction


class _FakeProvider:
    provider_name = "typesafe"

    def __init__(self, decisions=None, error: Exception | None = None) -> None:
        self.decisions = decisions
        self.error = error
        self.calls = 0

    def judge_writes(self, candidates, existing_nodes, *, context=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.decisions is not None:
            return list(self.decisions)
        return [WriteDecision(True, 1.0, "ok", "systemone") for _ in candidates]


class DecisionConsolidationExtractorTest(unittest.TestCase):
    def test_off_mode_does_not_call_provider(self) -> None:
        provider = _FakeProvider()
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=provider,
            mode=DecisionMode.OFF,
        )
        result = extractor.extract([], [])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(result.new_entities), 2)

    def test_shadow_records_but_keeps_everything(self) -> None:
        decisions = [WriteDecision(True, 1.0, "keep", "systemone"), WriteDecision(False, 0.9, "drop", "systemone")]
        records: list[DecisionRecord] = []
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=_FakeProvider(decisions),
            mode=DecisionMode.SHADOW,
            recorder=records.append,
        )
        result = extractor.extract([], [])
        self.assertEqual(len(result.new_entities), 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.shadow for record in records))
        self.assertEqual(records[1].answer["label"], "Greeting")

    def test_enforce_drops_rejected_entities_only(self) -> None:
        decisions = [WriteDecision(True, 1.0, "keep", "systemone"), WriteDecision(False, 0.9, "drop", "systemone")]
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=_FakeProvider(decisions),
            mode=DecisionMode.ENFORCE,
        )
        result = extractor.extract([], [])
        self.assertEqual([entity.label for entity in result.new_entities], ["Django Auth"])
        self.assertEqual(len(result.updates_to_existing_nodes), 1)

    def test_provider_error_fails_open(self) -> None:
        records: list[DecisionRecord] = []
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=_FakeProvider(error=DecisionError("down")),
            mode=DecisionMode.ENFORCE,
            recorder=records.append,
        )
        result = extractor.extract([], [])
        self.assertEqual(len(result.new_entities), 2)
        self.assertTrue(all(record.fallback for record in records))

    def test_mismatched_decision_count_fails_open(self) -> None:
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=_FakeProvider(decisions=[WriteDecision(False, 1.0, "drop", "systemone")]),
            mode=DecisionMode.ENFORCE,
        )
        result = extractor.extract([], [])
        self.assertEqual(len(result.new_entities), 2)

    def test_non_systemone_decisions_are_marked_fallback(self) -> None:
        records: list[DecisionRecord] = []
        decisions = [WriteDecision(True, 1.0, "sensitive kept", "heuristic")] * 2
        extractor = DecisionConsolidationExtractor(
            inner=_FakeInner(_extraction()),
            provider=_FakeProvider(decisions),
            mode=DecisionMode.ENFORCE,
            recorder=records.append,
        )
        extractor.extract([], [])
        self.assertTrue(all(record.fallback for record in records))


if __name__ == "__main__":
    unittest.main()