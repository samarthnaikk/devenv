from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.runtime import DevenvKernel
from core.runtime.kernel import PLANNING_SYSTEM_RULE, _focus_memory_context_for_direct_answers, _summarize_execution_note
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


class CapturingDiagnosticsTool(BaseTool):
    name = "run_diagnostics"
    description = "Captures diagnostic target paths for verification tests."

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def input_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult(success=True, output="PASS", data=dict(kwargs))


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
        self.assertIn("Verification failed; appended repair checkpoint", result.system_logs)
        self.assertTrue(any(task.repair_origin_checkpoint_id == 1 for task in result.blueprint.tasks))
        repair_task = next(task for task in result.blueprint.tasks if task.repair_origin_checkpoint_id == 1)
        self.assertIn("Verification failed", repair_task.description)

    def test_verification_failure_does_not_chain_repairs_from_repair_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [x] Update backend\n- [x] Repair checkpoint 1",
                tasks=[
                    CheckpointTask(task_id=1, description="Update backend", is_completed=True),
                    CheckpointTask(task_id=2, description="Repair checkpoint 1", repair_origin_checkpoint_id=1, is_completed=True),
                ],
                active_task_pointer=1,
            )

            updated, appended = kernel._append_repair_checkpoint(blueprint, checkpoint_id=2, reason="Verification failed")

        self.assertFalse(appended)
        self.assertEqual(len(updated.tasks), 2)

    def test_planning_rule_allows_many_single_shot_checkpoints(self) -> None:
        self.assertIn("as many single-shot checkpoints as needed", PLANNING_SYSTEM_RULE)
        self.assertNotIn("at most 4 checkpoints", PLANNING_SYSTEM_RULE)

    def test_verification_scopes_diagnostics_to_checkpoint_target(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            (Path(tempdir) / "calendar" / "frontend").mkdir(parents=True)
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            diagnostics = CapturingDiagnosticsTool()
            kernel.register_tool(diagnostics)
            checkpoint = CheckpointTask(
                task_id=1,
                description="Create calendar frontend",
                target_path_hint="calendar/frontend",
                expected_artifact="frontend",
                verification_mode="frontend",
            )

            success, _trace, results = kernel._verify_active_checkpoint(
                checkpoint=checkpoint,
                final_response="Created the frontend files.",
                checkpoint_steps=[],
                system_logs=[],
            )

        self.assertTrue(success)
        self.assertEqual(len(diagnostics.calls), 2)
        self.assertTrue(all(call["target_path"].endswith("calendar/frontend") for call in diagnostics.calls))
        self.assertEqual([result.mode for result in results], ["file", "tests", "types"])

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

    def test_validate_scaffold_tool_call_rejects_empty_write_to_target_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "make frontend folder for calendar html css js"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Create html file",
                tasks=[CheckpointTask(task_id=1, description="Create html file")],
                active_task_pointer=0,
            )
            error = kernel._validate_scaffold_tool_call(
                "write_file",
                {
                    "path": str((Path(tempdir) / "calendar" / "frontend").resolve()),
                    "content": "",
                    "mode": "fresh",
                },
            )

        self.assertIn("index.html", error)

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
