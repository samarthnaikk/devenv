from __future__ import annotations

import unittest

from core.decisions.models import DecisionError, DecisionMode, DecisionRecord
from core.decisions.router import IntentRouter, build_intent_router
from core.decisions.config import DecisionConfig
from core.runtime.local_router import LocalRouteDecision


def _decision(use_local: bool, confidence: float, reason: str) -> LocalRouteDecision:
    return LocalRouteDecision(use_local, confidence, confidence, 0.0, reason)


class _FakeHeuristic:
    threshold = 0.44

    def __init__(self, decision: LocalRouteDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, prompt: str) -> LocalRouteDecision:
        self.calls += 1
        return self.decision


class _FakeProvider:
    provider_name = "typesafe"

    def __init__(self, decision: LocalRouteDecision | None = None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls = 0

    def decide(self, prompt: str) -> LocalRouteDecision:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return self.decision


class IntentRouterTest(unittest.TestCase):
    def _router(self, *, mode, provider, recorder=None, min_confidence=0.65) -> IntentRouter:
        return IntentRouter(
            heuristic=_FakeHeuristic(_decision(True, 0.3, "heuristic")),
            provider=provider,
            mode=mode,
            provider_name="typesafe",
            min_confidence=min_confidence,
            recorder=recorder,
        )

    def test_off_mode_never_calls_provider(self) -> None:
        provider = _FakeProvider(_decision(False, 0.99, "systemone:mutate"))
        records: list[DecisionRecord] = []
        router = self._router(mode=DecisionMode.OFF, provider=provider, recorder=records.append)

        result = router.decide("how does the backend work?")

        self.assertTrue(result.use_local_knowledge)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(records, [])

    def test_shadow_records_but_returns_heuristic(self) -> None:
        provider = _FakeProvider(_decision(False, 0.99, "systemone:mutate"))
        records: list[DecisionRecord] = []
        router = self._router(mode=DecisionMode.SHADOW, provider=provider, recorder=records.append)

        result = router.decide("how does the backend work?")

        self.assertTrue(result.use_local_knowledge)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].shadow)
        self.assertEqual(records[0].gate, "intent")

    def test_enforce_uses_confident_provider_decision(self) -> None:
        provider = _FakeProvider(_decision(False, 0.91, "systemone:mutate"))
        records: list[DecisionRecord] = []
        router = self._router(mode=DecisionMode.ENFORCE, provider=provider, recorder=records.append)

        result = router.decide("create a new file")

        self.assertFalse(result.use_local_knowledge)
        self.assertEqual(records[0].fallback, False)
        self.assertFalse(records[0].shadow)

    def test_enforce_falls_back_below_threshold(self) -> None:
        provider = _FakeProvider(_decision(False, 0.4, "systemone:mutate"))
        records: list[DecisionRecord] = []
        router = self._router(mode=DecisionMode.ENFORCE, provider=provider, recorder=records.append)

        result = router.decide("create a new file")

        self.assertTrue(result.use_local_knowledge)
        self.assertTrue(records[0].fallback)
        self.assertIn("threshold", records[0].answer.get("error", ""))

    def test_provider_error_falls_back(self) -> None:
        provider = _FakeProvider(error=DecisionError("boom"))
        records: list[DecisionRecord] = []
        router = self._router(mode=DecisionMode.ENFORCE, provider=provider, recorder=records.append)

        result = router.decide("how does the backend work?")

        self.assertTrue(result.use_local_knowledge)
        self.assertTrue(records[0].fallback)
        self.assertEqual(records[0].answer.get("error"), "boom")

    def test_recorder_failure_is_swallowed(self) -> None:
        def boom(_record: DecisionRecord) -> None:
            raise RuntimeError("recorder down")

        provider = _FakeProvider(_decision(False, 0.91, "systemone:mutate"))
        router = self._router(mode=DecisionMode.ENFORCE, provider=provider, recorder=boom)
        result = router.decide("create a file")
        self.assertFalse(result.use_local_knowledge)


class BuildIntentRouterTest(unittest.TestCase):
    def test_off_config_yields_heuristic_router(self) -> None:
        router = build_intent_router(DecisionConfig.from_env(env={}))
        decision = router.decide("how does the backend work?")
        self.assertIsInstance(decision, LocalRouteDecision)

    def test_typesafe_config_selects_systemone_provider(self) -> None:
        config = DecisionConfig.from_env(
            env={
                "DEVENV_DECISION_PROVIDER": "typesafe",
                "DEVENV_DECISION_INTENT": "shadow",
                "TYPESAFE_API_KEY": "key",
            }
        )
        router = build_intent_router(config)
        self.assertEqual(router._provider_name, "typesafe")
        self.assertIsNotNone(router._provider)


if __name__ == "__main__":
    unittest.main()
