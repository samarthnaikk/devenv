from __future__ import annotations

import tempfile
import unittest
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.runtime import DevenvKernel
from core.runtime.kernel import _focus_memory_context_for_direct_answers, _summarize_execution_note
from core.runtime.models import AgentState, CheckpointTask, ExecutionBlueprint, PlanningMode
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


class FakeWriteTool(BaseTool):
    name = "write_file"
    description = "Fake writer for execution tests."

    def input_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="wrote file", data=dict(kwargs))


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

    def test_parse_markdown_to_blueprint_extracts_step_sections_without_swallowing_code(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = kernel._parse_markdown_to_blueprint(
                "\n".join(
                    [
                        "To complete the frontend:",
                        "Step 1: HTML Structure",
                        "Create the basic HTML structure for the calendar.",
                        "```html",
                        "<div>ignored code</div>",
                        "```",
                        "### Step 2: CSS Styling",
                        "Add CSS to make the calendar visually appealing.",
                        "```css",
                        ".calendar {}",
                        "```",
                        "Step 3: JavaScript Functionality",
                        "Add JavaScript to render days and handle navigation.",
                    ]
                )
            )

        self.assertEqual(len(blueprint.tasks), 3)
        self.assertEqual(blueprint.tasks[0].description, "HTML Structure: Create the basic HTML structure for the calendar.")
        self.assertEqual(blueprint.tasks[1].description, "CSS Styling: Add CSS to make the calendar visually appealing.")
        self.assertEqual(blueprint.tasks[2].description, "JavaScript Functionality: Add JavaScript to render days and handle navigation.")

    def test_summarize_execution_note_strips_code_and_keeps_plain_objective(self) -> None:
        note = _summarize_execution_note(
            "\n".join(
                [
                    "To add JavaScript functionality for calendar interactions, I will wire up month navigation.",
                    "```javascript",
                    "const monthYear = document.getElementById('month-year');",
                    "function renderCalendar() {}",
                    "```",
                    "<div class=\"calendar\"></div>",
                ]
            )
        )

        self.assertEqual(
            note,
            "To add JavaScript functionality for calendar interactions, I will wire up month navigation.",
        )

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

    def test_force_plan_executes_one_checkpoint_per_turn(self) -> None:
        ai = FakeAI(
            [
                AIResponse(
                    content="- [ ] Create calendar folder\n- [ ] Add main.py",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="Created the calendar folder.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="Added main.py.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            first = kernel.execute_turn("Create a calendar app", planning_mode=PlanningMode.FORCE_PLAN)
            second = kernel.execute_turn(
                "Create a calendar app",
                planning_mode=PlanningMode.FORCE_PLAN,
                continue_plan=True,
            )

        self.assertEqual(first.final_response, "Created the calendar folder.")
        self.assertEqual([task.is_completed for task in first.blueprint.tasks], [True, False])
        self.assertEqual(first.state, AgentState.EXECUTING.name)
        self.assertEqual(second.final_response, "Added main.py.")
        self.assertEqual([task.is_completed for task in second.blueprint.tasks], [True, True])
        self.assertEqual(second.state, AgentState.VERIFYING.name)

    def test_auto_planning_executes_one_checkpoint_per_turn(self) -> None:
        ai = FakeAI(
            [
                AIResponse(
                    content="- [ ] Create frontend folder\n- [ ] Add styles.css",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="Created the frontend folder.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            result = kernel.execute_turn("Create a frontend folder")

        self.assertEqual(result.final_response, "Created the frontend folder.")
        self.assertEqual([task.is_completed for task in result.blueprint.tasks], [True, False])
        self.assertEqual(result.state, AgentState.EXECUTING.name)

    def test_mutation_checkpoint_requires_real_write_tool_before_completion(self) -> None:
        ai = FakeAI(
            [
                AIResponse(
                    content="- [ ] Create index.html",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content="I would create index.html with a simple layout.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call-1",
                            tool_name="write_file",
                            arguments={"path": "calendar/index.html", "content": "<h1>Calendar</h1>", "mode": "fresh"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content="Created index.html.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            kernel.register_tool(FakeWriteTool())
            result = kernel.execute_turn("Create index.html for the calendar frontend")

        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "write_file")
        self.assertIn("requires a file mutation tool before completion", "\n".join(result.system_logs))
        self.assertEqual(result.final_response, "Created index.html.")

    def test_build_execution_prompt_includes_target_path_and_checkpoint_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [x] Create frontend folder\n- [ ] Add index.html\n- [ ] Add styles.css",
                tasks=[
                    CheckpointTask(task_id=1, description="Create frontend folder", is_completed=True),
                    CheckpointTask(task_id=2, description="Add index.html"),
                    CheckpointTask(task_id=3, description="Add styles.css"),
                ],
                active_task_pointer=1,
            )
            prompt = kernel._build_execution_prompt(
                user_prompt="complete the frontend for calendar in html css js",
                checkpoint_index=2,
                total_checkpoints=3,
                task_description="Add index.html",
                blueprint=blueprint,
            )

        self.assertIn("All new files for this request must stay under: calendar/frontend", prompt)
        self.assertIn("Completed earlier: Create frontend folder", prompt)
        self.assertIn("Next after this: Add styles.css", prompt)

    def test_repair_tool_arguments_prefixes_scaffold_files_under_calendar_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "complete the frontend for calendar in html css js"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Add index.html",
                tasks=[CheckpointTask(task_id=1, description="Add index.html")],
                active_task_pointer=0,
            )
            repaired = kernel._repair_tool_arguments(
                ToolCallRequest(
                    call_id="call-1",
                    tool_name="write_file",
                    arguments={"path": "index.html", "content": "<h1>Calendar</h1>", "mode": "fresh"},
                )
            )

        self.assertEqual(repaired["path"], "calendar/frontend/index.html")

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
