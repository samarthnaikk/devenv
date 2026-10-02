from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse
from core.decisions.models import DecisionRecord
from core.decisions.router import IntentRouter
from core.memory.storage import SQLiteMemoryStore
from core.runtime import DevenvKernel
from core.runtime.local_router import LocalRouteDecision


class _FakeMemory:
    def __init__(self, store: SQLiteMemoryStore | None = None) -> None:
        self.store = store

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        return None

    def retrieve_context(self, current_prompt: str, top_k: int = 5):
        return type("Result", (), {"markdown_context": ""})()

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "log-1"


class _FakeAI:
    def register_tool(self, tool) -> None:
        return None

    def chat(self, messages, memory_context=None, temperature=0.2, tool_names=None) -> AIResponse:
        return AIResponse(content="ok")


class KernelIntentWiringTest(unittest.TestCase):
    def test_default_router_is_intent_router_and_uses_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=_FakeMemory(), ai=_FakeAI())

            self.assertIsInstance(kernel.local_router, IntentRouter)
            decision = kernel.local_router.decide("how does the backend work and why?")
            self.assertIsInstance(decision, LocalRouteDecision)

    def test_decision_record_reaches_audit_trail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(str(Path(tempdir) / "memory.db"))
            kernel = DevenvKernel(tempdir, memory=_FakeMemory(store=store), ai=_FakeAI())

            kernel._record_decision(
                DecisionRecord(
                    gate="intent",
                    domain="intent",
                    provider="heuristic",
                    mode="shadow",
                    answer={"use_local_knowledge": True},
                    confidence=0.5,
                )
            )
            events = store.list_runtime_events(event_type="decision.result", limit=10)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "decision.result")


if __name__ == "__main__":
    unittest.main()
