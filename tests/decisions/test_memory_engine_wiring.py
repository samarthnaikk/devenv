from __future__ import annotations

import tempfile
import unittest

from core.decisions.config import DecisionConfig
from core.decisions.models import DecisionRecord, WriteDecision
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex


class _GateProvider:
    provider_name = "typesafe"

    def __init__(self) -> None:
        self.calls = 0

    def judge_writes(self, candidates, existing_nodes, *, context=None):
        self.calls += 1
        return [
            WriteDecision(
                should_write=str(candidate.label) == "Keep",
                confidence=0.9,
                reason="test",
                source="systemone",
            )
            for candidate in candidates
        ]


def _seed(engine: MemoryEngine) -> None:
    engine.add_episodic_log(
        "We picked session cookies for auth and rejected JWT for now.",
        "Noted.",
        metadata={
            "memory_entities": [
                {
                    "node_id": "cmp_keep",
                    "label": "Keep",
                    "category": "decision",
                    "summary": "Auth uses session cookies; JWT was rejected.",
                },
                {
                    "node_id": "cmp_drop",
                    "label": "Drop",
                    "category": "noise",
                    "summary": "The user said hello.",
                },
            ]
        },
    )


class MemoryEngineWriteGateTest(unittest.TestCase):
    def _engine(self, *, mode: str, provider: _GateProvider, recorder=None) -> MemoryEngine:
        config = DecisionConfig.from_env(
            env={
                "DEVENV_DECISION_PROVIDER": "local",
                "DEVENV_SYSTEMONE_BASE_URL": "http://127.0.0.1:8000",
                "DEVENV_DECISION_MEMORY_WRITE": mode,
            }
        )
        return MemoryEngine(
            db_path=f"{self.tempdir.name}/memory.db",
            vector_dir=f"{self.tempdir.name}/vectors",
            embedder=HashingEmbedder(dimension=8),
            vector_index=InMemoryVectorIndex(),
            decision_config=config,
            decision_provider=provider,
            decision_recorder=recorder,
        )

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_enforce_drops_rejected_entity(self) -> None:
        provider = _GateProvider()
        engine = self._engine(mode="enforce", provider=provider)
        _seed(engine)

        result = engine.run_consolidation()

        self.assertEqual(result.created_nodes, ("cmp_keep",))
        self.assertIsNone(engine.store.get_node("cmp_drop"))
        self.assertEqual(provider.calls, 1)

    def test_shadow_keeps_everything_but_records(self) -> None:
        provider = _GateProvider()
        records: list[DecisionRecord] = []
        engine = self._engine(mode="shadow", provider=provider, recorder=records.append)
        _seed(engine)

        result = engine.run_consolidation()

        self.assertEqual(set(result.created_nodes), {"cmp_keep", "cmp_drop"})
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.shadow for record in records))
        self.assertEqual(provider.calls, 1)

    def test_off_by_default_does_not_call_provider(self) -> None:
        provider = _GateProvider()
        engine = self._engine(mode="off", provider=provider)
        _seed(engine)

        result = engine.run_consolidation()

        self.assertEqual(set(result.created_nodes), {"cmp_keep", "cmp_drop"})
        self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()