from __future__ import annotations

import tempfile
import unittest
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.runtime import DevenvKernel
from core.runtime.kernel import _focus_memory_context_for_direct_answers
from core.runtime.models import AgentState, PlanningMode
from core.tools.base import BaseTool, ToolResult


class FakeMemory:
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


class FakeAI:
    def __init__(self, responses: list[AIResponse]) -> None:
        self.responses = list(responses)

    def register_tool(self, tool) -> None:
        return None

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names=None,
    ) -> AIResponse:
        return self.responses.pop(0)


class FailingDiagnosticsTool(BaseTool):
    name = "run_diagnostics"
    description = "Fake diagnostics tool for testing."

    def input_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode", "tests")
        return ToolResult(success=False, output=f"FAIL {mode}", data={"mode": mode})


class PlanningKernelTest(unittest.TestCase):
    def test_parse_markdown_to_blueprint_extracts_mixed_checkbox_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = kernel._parse_markdown_to_blueprint(
                "\n".join(
                    [
                        "Plan:",
                        "- [ ] Inspect the calendar folder",
                        "* [ ] Add main.py",
                        "1. [x] Verify sample output",
                    ]
                )
            )

        self.assertEqual([task.description for task in blueprint.tasks], [
            "Inspect the calendar folder",
            "Add main.py",
            "Verify sample output",
        ])
        self.assertFalse(blueprint.tasks[0].is_completed)
        self.assertTrue(blueprint.tasks[2].is_completed)

    def test_planning_blocks_mutation_tool_calls_until_a_blueprint_exists(self) -> None:
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(call_id="call-1", tool_name="write_file", arguments={"path": "main.py", "content": "print('x')"}),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content="- [ ] Create main.py",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="Created the file plan.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            result = kernel.execute_turn("Create a main.py")

        self.assertIn("Blocked planning tool call: write_file", result.system_logs)
        self.assertEqual(result.blueprint.tasks[0].description, "Create main.py")

    def test_verification_failure_resets_state_to_planning(self) -> None:
        ai = FakeAI(
            [
                AIResponse(
                    content="- [ ] Update the backend",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="Updated the backend.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            kernel.register_tool(FailingDiagnosticsTool())
            result = kernel.execute_turn("Update the backend")

        self.assertEqual(result.state, AgentState.PLANNING.name)
        self.assertFalse(result.blueprint.verification_passed)
        self.assertIn("Verification failed; state reset to PLANNING", result.system_logs)

    def test_scaffold_request_uses_tiny_execution_scope_and_trimmed_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.tools = {
                "list_directory": object(),
                "write_file": object(),
                "edit_file": object(),
                "read_file": object(),
            }

            scope = kernel._resolve_execution_tool_scope(
                "make a frontend folder in calendar with html css and js files",
                "Create the frontend folder and starter files",
            )
            memory_context = kernel._resolve_execution_memory(
                user_prompt="make a frontend folder in calendar with html css and js files",
                task_description="Create the frontend folder and starter files",
                memory_context="## Retrieved Memory\n- Older project notes",
            )

        self.assertEqual(scope, ["edit_file", "list_directory", "write_file"])
        self.assertIn("Older project notes", memory_context)
        self.assertLessEqual(len(memory_context), 360)

    def test_requires_planning_only_for_change_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

        self.assertTrue(kernel._requires_planning("Fix the backend auth bug"))
        self.assertTrue(kernel._requires_planning("make a frontend folder in calendar"))
        self.assertTrue(kernel._requires_planning("complete frontend for calendar folder"))
        self.assertFalse(kernel._requires_planning("how does the rvidia backend work"))
        self.assertFalse(kernel._requires_planning("tell me about this project"))

    def test_should_plan_respects_explicit_planning_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

        self.assertTrue(kernel._should_plan("how does the repo work?", PlanningMode.FORCE_PLAN))
        self.assertFalse(kernel._should_plan("create a frontend folder", PlanningMode.FORCE_DIRECT))

    def test_direct_memory_focus_prefers_retrieved_memory_block(self) -> None:
        focused = _focus_memory_context_for_direct_answers(
            "## Working Memory\n- noisy\n## Retrieved Memory\n- [episode] rvidia backend uses FastAPI",
            120,
        )

        self.assertNotIn("## Working Memory", focused)
        self.assertIn("## Retrieved Memory", focused)
        self.assertIn("FastAPI", focused)


if __name__ == "__main__":
    unittest.main()
