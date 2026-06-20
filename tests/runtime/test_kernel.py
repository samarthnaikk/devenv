from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.runtime import DevenvKernel
from core.tools.list_directory import ListDirectoryTool
from core.tools.read_file import ReadFileTool


@dataclass(frozen=True)
class FakeRetrievalResult:
    markdown_context: str


class FakeMemory:
    def __init__(self) -> None:
        self.working_memory_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.logs: list[tuple[str, str, dict[str, Any] | None]] = []
        self.consolidation_runs = 0

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        self.working_memory_calls.append((messages, active_state))

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context=f"## Retrieved Memory\n- Prompt: {current_prompt}")

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.logs.append((user_prompt, agent_response, metadata))
        return "log-1"

    def run_consolidation(self):
        self.consolidation_runs += 1
        return type("Result", (), {"processed_logs": 1, "created_nodes": (), "updated_nodes": ()})()


class FailingMemory(FakeMemory):
    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        raise RuntimeError("memory offline")


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
            kernel.register_tool(ListDirectoryTool())

        self.assertIn("read_file", kernel.tools)
        self.assertIn("list_directory", kernel.tools)
        self.assertEqual(ai.registered_tools, ["read_file", "list_directory"])

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
        self.assertEqual(memory.logs[0][0], "Explain the repo")
        self.assertEqual(memory.logs[0][1], "Final answer")
        self.assertEqual(memory.logs[0][2]["workspace_path"], str(Path(tempdir).resolve()))
        self.assertEqual(memory.logs[0][2]["session_id"], kernel.session_id)
        self.assertEqual(memory.working_memory_calls[0][1]["session_id"], kernel.session_id)
        self.assertEqual(memory.consolidation_runs, 1)
        self.assertTrue(result.ai_logs)
        self.assertTrue(result.system_logs)

    def test_execute_turn_runs_registered_tool(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_1",
                            tool_name="read_file",
                            arguments={"path": "note.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 7},
                ),
                AIResponse(
                    content="I read the file.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"completion_tokens": 4},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            note_path = f"{tempdir}/note.txt"
            with open(note_path, "w", encoding="utf-8") as handle:
                handle.write("hello runtime")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            result = kernel.execute_turn("Read note.txt")

        self.assertEqual(result.final_response, "I read the file.")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "read_file")
        self.assertTrue(result.steps[0].success)
        self.assertIn("read_file completed", result.steps[0].output)
        self.assertIn("hello runtime", result.steps[0].output)
        second_call_messages = ai.chat_calls[1]["messages"]
        self.assertEqual(second_call_messages[-1]["role"], "tool")
        self.assertEqual(result.total_usage["prompt_tokens"], 7)
        self.assertEqual(result.total_usage["completion_tokens"], 4)

    def test_execute_turn_flags_sandbox_violation(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_unsafe",
                            tool_name="read_file",
                            arguments={"path": "../secrets.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content="I could not access that path.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Read ../secrets.txt")

        self.assertTrue(result.steps[0].is_sandboxed_violation)
        self.assertIn("Sandbox violation", result.steps[0].output)
        self.assertEqual(result.final_response, "I could not access that path.")

    def test_execute_turn_caps_tool_iterations(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(call_id="call_1", tool_name="missing_tool", arguments={}),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(call_id="call_2", tool_name="missing_tool", arguments={}),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Loop", max_consecutive_tools=1)

        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.final_response, "Tool execution limit reached before the request could be completed.")

    def test_execute_turn_continues_when_memory_retrieval_fails(self) -> None:
        memory = FailingMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Fallback answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 2},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Explain the repo")

        self.assertEqual(result.final_response, "Fallback answer")
        self.assertEqual(ai.chat_calls[0]["memory_context"], "")


if __name__ == "__main__":
    unittest.main()
