from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from typing import Any

from core.ai.models import AIResponse
from core.runtime import DevenvKernel
from core.tools.read_file import ReadFileTool


@dataclass(frozen=True)
class FakeRetrievalResult:
    markdown_context: str


class FakeMemory:
    def __init__(self) -> None:
        self.working_memory_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        self.working_memory_calls.append((messages, active_state))

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context=f"## Retrieved Memory\n- Prompt: {current_prompt}")


class FakeAI:
    def __init__(self, responses: list[AIResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.registered_tools: list[str] = []
        self.chat_calls: list[dict[str, Any]] = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def chat(self, messages: list[dict[str, Any]], memory_context: str | None = None, temperature: float = 0.2) -> AIResponse:
        self.chat_calls.append(
            {
                "messages": messages,
                "memory_context": memory_context,
                "temperature": temperature,
            }
        )
        return self.responses.pop(0)


class DevenvKernelTest(unittest.TestCase):
    def test_register_tool_syncs_runtime_and_ai(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())

        self.assertIn("read_file", kernel.tools)
        self.assertEqual(ai.registered_tools, ["read_file"])

    def test_execute_turn_returns_direct_ai_response(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Final answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 10},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Explain the repo")

        self.assertEqual(result.final_response, "Final answer")
        self.assertEqual(result.total_usage["prompt_tokens"], 10)
        self.assertEqual(ai.chat_calls[0]["memory_context"], "## Retrieved Memory\n- Prompt: Explain the repo")


if __name__ == "__main__":
    unittest.main()
