from __future__ import annotations

import unittest

from core.decisions.config import DecisionConfig
from core.decisions.models import DecisionError, DecisionUnavailable
from core.decisions.provider import SystemOneDecisionProvider, build_decision_provider
from core.decisions.systemone import SystemOneResult
from core.runtime.local_router import LocalRouteDecision


class _FakeClient:
    def __init__(self, result: SystemOneResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.states: list[str] = []

    def evaluate(self, state, questions, *, model=None):
        self.states.append(state)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _result() -> SystemOneResult:
    return SystemOneResult(
        model="jev-1.13.0",
        answers={
            "route": {"type": "choice", "choice": "recall", "confidence": 0.9},
            "mutates_workspace": {"type": "noul", "noul": 0.1},
        },
    )


def _config(**overrides: str) -> DecisionConfig:
    env = {
        "DEVENV_DECISION_PROVIDER": "local",
        "DEVENV_SYSTEMONE_BASE_URL": "http://127.0.0.1:8000",
        "DEVENV_DECISION_INTENT": "enforce",
        **overrides,
    }
    return DecisionConfig.from_env(env=env)


class SystemOneDecisionProviderTest(unittest.TestCase):
    def test_decide_maps_answers_to_route_decision(self) -> None:
        client = _FakeClient(_result())
        provider = SystemOneDecisionProvider(_config(), client=client)

        decision = provider.decide("what do you remember about this repo?")

        self.assertIsInstance(decision, LocalRouteDecision)
        self.assertTrue(decision.use_local_knowledge)
        self.assertEqual(decision.confidence, 0.9)
        self.assertEqual(decision.reason, "systemone:recall")
        self.assertEqual(len(client.states), 1)

    def test_sensitive_state_is_refused(self) -> None:
        client = _FakeClient(_result())
        provider = SystemOneDecisionProvider(_config(), client=client)

        with self.assertRaises(DecisionUnavailable):
            provider.decide("-----BEGIN RSA PRIVATE KEY-----\nMIIE")
        self.assertEqual(client.states, [])

    def test_client_error_propagates(self) -> None:
        client = _FakeClient(error=DecisionError("upstream down"))
        provider = SystemOneDecisionProvider(_config(), client=client)
        with self.assertRaises(DecisionError):
            provider.decide("how does this work?")

    def test_local_provider_is_not_blocked_by_offline_guard(self) -> None:
        client = _FakeClient(_result())
        provider = SystemOneDecisionProvider(_config(), client=client, offline=True)
        decision = provider.decide("how does this work?")
        self.assertTrue(decision.use_local_knowledge)


class BuildDecisionProviderTest(unittest.TestCase):
    def test_off_provider_is_heuristic(self) -> None:
        provider = build_decision_provider(DecisionConfig.from_env(env={}))
        self.assertEqual(getattr(provider, "provider_name", "heuristic"), "heuristic")

    def test_typesafe_provider_selected(self) -> None:
        config = DecisionConfig.from_env(env={"DEVENV_DECISION_PROVIDER": "typesafe"})
        provider = build_decision_provider(config)
        self.assertIsInstance(provider, SystemOneDecisionProvider)

    def test_remote_provider_offline_falls_back_to_heuristic(self) -> None:
        config = DecisionConfig.from_env(env={"DEVENV_DECISION_PROVIDER": "typesafe"})
        provider = build_decision_provider(config, offline=True)
        self.assertEqual(getattr(provider, "provider_name", "heuristic"), "heuristic")


if __name__ == "__main__":
    unittest.main()
