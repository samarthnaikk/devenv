from __future__ import annotations

import unittest

from core.decisions.config import DecisionConfig
from core.decisions.provider import (
    HeuristicDecisionProvider,
    MemoryWriteDecisionProvider,
    RouteDecisionProvider,
    build_decision_provider,
)
from core.runtime.local_router import LocalRouteDecision


class HeuristicDecisionProviderTest(unittest.TestCase):
    def test_route_decision_matches_local_router_contract(self) -> None:
        provider = HeuristicDecisionProvider()
        decision = provider.decide("how does the backend work and why?")
        self.assertIsInstance(decision, LocalRouteDecision)

    def test_write_passthrough_keeps_every_candidate(self) -> None:
        provider = HeuristicDecisionProvider()
        decisions = provider.judge_writes([object(), object(), object()], existing_nodes=[])
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(decision.should_write for decision in decisions))
        self.assertTrue(all(decision.source == "heuristic" for decision in decisions))

    def test_satisfies_protocols(self) -> None:
        provider = HeuristicDecisionProvider()
        self.assertIsInstance(provider, RouteDecisionProvider)
        self.assertIsInstance(provider, MemoryWriteDecisionProvider)

    def test_factory_returns_heuristic(self) -> None:
        provider = build_decision_provider(DecisionConfig.from_env(env={}))
        self.assertIsInstance(provider, HeuristicDecisionProvider)


if __name__ == "__main__":
    unittest.main()
