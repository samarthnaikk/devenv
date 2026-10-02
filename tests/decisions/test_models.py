from __future__ import annotations

import unittest

from core.decisions.models import (
    DecisionMode,
    DecisionRecord,
    WriteDecision,
)


class DecisionModeTest(unittest.TestCase):
    def test_parse_known_values(self) -> None:
        self.assertIs(DecisionMode.parse("off"), DecisionMode.OFF)
        self.assertIs(DecisionMode.parse("SHADOW"), DecisionMode.SHADOW)
        self.assertIs(DecisionMode.parse(" enforce "), DecisionMode.ENFORCE)

    def test_parse_falls_back(self) -> None:
        self.assertIs(DecisionMode.parse(None), DecisionMode.OFF)
        self.assertIs(DecisionMode.parse("bogus"), DecisionMode.OFF)
        self.assertIs(DecisionMode.parse("", default=DecisionMode.SHADOW), DecisionMode.SHADOW)
        self.assertIs(DecisionMode.parse("bogus", default=DecisionMode.ENFORCE), DecisionMode.ENFORCE)


class WriteDecisionTest(unittest.TestCase):
    def test_to_dict_round_trips_fields(self) -> None:
        decision = WriteDecision(should_write=False, confidence=0.2, reason="not durable", source="typesafe")
        payload = decision.to_dict()
        self.assertFalse(payload["should_write"])
        self.assertEqual(payload["confidence"], 0.2)
        self.assertEqual(payload["reason"], "not durable")
        self.assertEqual(payload["source"], "typesafe")


class DecisionRecordTest(unittest.TestCase):
    def test_to_dict_omits_optional_when_empty(self) -> None:
        record = DecisionRecord(gate="intent", domain="routing", provider="heuristic", mode="shadow")
        payload = record.to_dict()
        self.assertEqual(payload["gate"], "intent")
        self.assertNotIn("confidence", payload)
        self.assertNotIn("state_hash", payload)

    def test_to_dict_includes_optional_when_set(self) -> None:
        record = DecisionRecord(
            gate="memory_write",
            domain="memory",
            provider="typesafe",
            mode="enforce",
            answer={"should_write": True},
            confidence=0.8,
            state_hash="abc",
        )
        payload = record.to_dict()
        self.assertEqual(payload["confidence"], 0.8)
        self.assertEqual(payload["state_hash"], "abc")
        self.assertEqual(payload["answer"], {"should_write": True})


if __name__ == "__main__":
    unittest.main()
