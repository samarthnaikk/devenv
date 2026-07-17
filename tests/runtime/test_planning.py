from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.runtime import DevenvKernel
from core.runtime.kernel import PLANNING_SYSTEM_RULE, _focus_memory_context_for_direct_answers, _summarize_execution_note
from core.runtime.models import AgentState, CheckpointTask, ExecutionBlueprint, ExecutionMode, PlanningMode, TurnOutcome
from core.tools.base import BaseTool, ToolResult
from core.tools.inspect_symbols import InspectSymbolsTool
from core.tools.list_directory import ListDirectoryTool
from core.tools.read_file import ReadFileTool


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

    def test_direct_blueprint_pre_splits_compound_mutation_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = kernel._build_direct_blueprint(
                "Update auth.py and api.py, then verify the login flow still works"
            )

        self.assertEqual(len(blueprint.tasks), 3)
        self.assertTrue(blueprint.tasks[0].description.startswith("Inspect the files and dependencies needed for:"))
        self.assertFalse(blueprint.tasks[0].expects_mutation)
        self.assertEqual(blueprint.tasks[0].verification_mode, "chat")
        self.assertTrue(blueprint.tasks[1].expects_mutation)
        self.assertEqual(blueprint.tasks[2].verification_mode, "code")

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
        self.assertEqual(result.execution_mode, ExecutionMode.REPAIR.value)
        self.assertEqual(result.turn_outcome, TurnOutcome.VERIFICATION_FAILURE.value)
        self.assertTrue(any(task.repair_origin_checkpoint_id == 1 for task in result.blueprint.tasks))
        repair_task = next(task for task in result.blueprint.tasks if task.repair_origin_checkpoint_id == 1)
        self.assertIn("Verification failed", repair_task.description)
        self.assertGreaterEqual(repair_task.repair_attempt_count, 1)
        self.assertTrue(repair_task.requires_verification)

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

    def test_append_repair_checkpoint_stops_when_repair_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Update backend",
                tasks=[
                    CheckpointTask(
                        task_id=1,
                        description="Update backend",
                        repair_attempt_count=2,
                        max_repair_attempts=2,
                    )
                ],
                active_task_pointer=0,
            )

            updated, appended = kernel._append_repair_checkpoint(blueprint, checkpoint_id=1, reason="Verification failed")

        self.assertFalse(appended)
        self.assertEqual(len(updated.tasks), 1)

    def test_describe_repair_chain_block_reports_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Update backend",
                tasks=[
                    CheckpointTask(
                        task_id=1,
                        description="Update backend",
                        repair_attempt_count=2,
                        max_repair_attempts=2,
                    )
                ],
                active_task_pointer=0,
            )

            reason = kernel._describe_repair_chain_block(blueprint, 1)

        self.assertEqual(reason, "repair budget exhausted for checkpoint 1")

    def test_describe_repair_chain_block_reports_repair_checkpoint_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Repair checkpoint 1",
                tasks=[
                    CheckpointTask(
                        task_id=2,
                        description="Repair checkpoint 1",
                        repair_origin_checkpoint_id=1,
                    )
                ],
                active_task_pointer=0,
            )

            reason = kernel._describe_repair_chain_block(blueprint, 2)

        self.assertEqual(reason, "checkpoint 2 is already a repair checkpoint")

    def test_build_checkpoint_task_populates_execution_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            task = kernel._build_checkpoint_task(
                task_id=1,
                description="Add index.html",
                original_objective="complete the frontend for calendar in html css js",
            )

        self.assertTrue(task.expects_mutation)
        self.assertTrue(task.requires_verification)
        self.assertIn("write_file", task.allowed_tool_names)
        self.assertGreaterEqual(task.max_repair_attempts, 1)

    def test_planning_rule_allows_many_single_shot_checkpoints(self) -> None:
        self.assertIn("as many single-shot checkpoints as needed", PLANNING_SYSTEM_RULE)
        self.assertNotIn("at most 4 checkpoints", PLANNING_SYSTEM_RULE)

    def test_verification_scopes_diagnostics_to_checkpoint_target(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            frontend = Path(tempdir) / "calendar" / "frontend"
            frontend.mkdir(parents=True)
            (frontend / "index.html").write_text(
                '<link rel="stylesheet" href="styles.css" />\n<div id="calendar-grid" class="calendar-grid"></div>\n<script src="script.js"></script>\n',
                encoding="utf-8",
            )
            (frontend / "styles.css").write_text(".calendar-day { color: white; }\n", encoding="utf-8")
            (frontend / "script.js").write_text("function renderCalendar() {}\n", encoding="utf-8")
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
        self.assertEqual([result.mode for result in results], ["file", "frontend", "lint"])

    def test_verification_prefers_touched_paths_reported_by_tool_data(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            src_dir = Path(tempdir) / "src"
            src_dir.mkdir(parents=True)
            target_file = src_dir / "feature.py"
            target_file.write_text("print('ok')\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            diagnostics = CapturingDiagnosticsTool()
            kernel.register_tool(diagnostics)
            checkpoint = CheckpointTask(
                task_id=1,
                description="Update the feature module",
                expected_artifact="code",
                verification_mode="code",
            )

            success, _trace, results = kernel._verify_active_checkpoint(
                checkpoint=checkpoint,
                final_response="Updated feature module.",
                checkpoint_steps=[
                    type(
                        "Step",
                        (),
                        {
                            "arguments": {},
                            "data": {"written_paths": ["src/feature.py"]},
                        },
                    )()
                ],
                system_logs=[],
            )

        self.assertTrue(success)
        self.assertEqual(results[0].mode, "file")
        self.assertTrue(all(call["target_path"].endswith("src/feature.py") for call in diagnostics.calls))

    def test_verification_can_be_skipped_by_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            checkpoint = CheckpointTask(
                task_id=1,
                description="Summarize the backend",
                verification_mode="chat",
                requires_verification=False,
            )

            success, _trace, results = kernel._verify_active_checkpoint(
                checkpoint=checkpoint,
                final_response="Backend summarized.",
                checkpoint_steps=[],
                system_logs=[],
            )

        self.assertTrue(success)
        self.assertEqual(results[0].mode, "none")

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

    def test_can_continue_active_plan_from_follow_up_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "Build the calendar frontend in html css js"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [x] Add index.html\n- [ ] Add styles.css",
                tasks=[
                    CheckpointTask(task_id=1, description="Add index.html", is_completed=True),
                    CheckpointTask(task_id=2, description="Add styles.css"),
                ],
                active_task_pointer=1,
            )

            can_continue = kernel._can_continue_active_plan("continue with the calendar frontend plan")

        self.assertTrue(can_continue)

    def test_can_exit_active_plan_from_follow_up_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "Build the calendar frontend in html css js"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Add index.html",
                tasks=[CheckpointTask(task_id=1, description="Add index.html")],
                active_task_pointer=0,
            )

            can_continue = kernel._can_continue_active_plan("exit plan mode and just answer")

        self.assertFalse(can_continue)

    def test_force_plan_executes_remaining_checkpoints_in_same_turn(self) -> None:
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
            result = kernel.execute_turn("Create a calendar app", planning_mode=PlanningMode.FORCE_PLAN)

        self.assertEqual(result.final_response, "Added main.py.")
        self.assertEqual([task.is_completed for task in result.blueprint.tasks], [True, True])
        self.assertEqual(result.state, AgentState.VERIFYING.name)

    def test_follow_up_continue_prompt_resumes_active_plan(self) -> None:
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
            second = kernel.execute_turn("continue", planning_mode=PlanningMode.AUTO)

        self.assertEqual(first.final_response, "Added main.py.")
        self.assertEqual(second.final_response, "Nothing left to execute.")
        self.assertEqual([task.is_completed for task in second.blueprint.tasks], [True, True])

    def test_auto_planning_executes_remaining_checkpoints_in_same_turn(self) -> None:
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
                AIResponse(
                    content="Added styles.css.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            result = kernel.execute_turn("Create a frontend folder")

        self.assertEqual(result.final_response, "Added styles.css.")
        self.assertEqual([task.is_completed for task in result.blueprint.tasks], [True, True])
        self.assertEqual(result.state, AgentState.VERIFYING.name)

    def test_follow_up_instruction_updates_active_plan_instead_of_replanning(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "I want to integrate chat app to this codebase, make a plan"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Inspect backend hooks\n- [ ] Draft integration plan",
                original_objective="I want to integrate chat app to this codebase, make a plan",
                tasks=[
                    CheckpointTask(task_id=1, description="Inspect backend hooks"),
                    CheckpointTask(task_id=2, description="Draft integration plan"),
                ],
                active_task_pointer=0,
            )

            blueprint, _conversation, trace = kernel._checkpoint_creation_stage(
                user_prompt="Okay add all the files in chatapp in folder (the backend files). Integrate with frontend",
                memory_context="",
                continue_plan=False,
                local_only=False,
                planning_mode=PlanningMode.AUTO,
                steps=[],
                total_usage={},
                ai_logs=[],
                system_logs=[],
                max_consecutive_tools=5,
                tool_policy_events=[],
            )

        self.assertEqual(trace.summary, "Updated active checkpoint plan from follow-up instruction")
        self.assertEqual(blueprint.original_objective, "Okay add all the files in chatapp in folder (the backend files). Integrate with frontend")
        self.assertEqual(blueprint.tasks[0].expected_artifact, "code")

    def test_local_plan_markdown_prefers_backend_frontend_integration_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            plan = kernel._build_local_plan_markdown(
                "Okay add all the files in chatapp in folder (the backend files). Integrate with frontend"
            )

        self.assertIn("Inspect `chatapp` and confirm which backend files already exist or are still missing.", plan)
        self.assertIn("Inspect `core/runtime/web.py` to map the backend request and registration surface", plan)
        self.assertIn("Inspect `core/ai/routing.py` to map the backend routing surface", plan)
        self.assertIn("Inspect `interface/website/src/api.js` to map the frontend API helper", plan)
        self.assertIn("Inspect `interface/website/src/App.js` to map the frontend UI surface", plan)
        self.assertIn("Create `chatapp/__init__.py`", plan)
        self.assertIn("Create `chatapp/store.py`", plan)
        self.assertIn("Create `chatapp/service.py`", plan)
        self.assertIn("Create `chatapp/routes.py`", plan)
        self.assertIn("Wire `core/runtime/web.py`", plan)
        self.assertIn("Wire `core/ai/routing.py`", plan)
        self.assertIn("Connect `interface/website/src/api.js`", plan)
        self.assertIn("Connect `interface/website/src/App.js`", plan)

    def test_direct_blueprint_for_backend_frontend_integration_uses_deterministic_plan_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            prompt = "I want to integrate chat app to this codebase, add all the files in chatapp in folder (the backend files). Integrate with frontend"
            blueprint = kernel._build_direct_blueprint(prompt)

        self.assertGreaterEqual(len(blueprint.tasks), 10)
        self.assertEqual(blueprint.original_objective, prompt)
        self.assertTrue(any("Create `chatapp/__init__.py`" in task.description for task in blueprint.tasks))
        self.assertTrue(any("Connect `interface/website/src/api.js`" in task.description for task in blueprint.tasks))

    def test_checkpoint_creation_prefers_local_planning_for_backend_frontend_integration_prompt(self) -> None:
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            prompt = "I want to integrate chat app to this codebase, add all the files in chatapp in folder (the backend files). Integrate with frontend"
            blueprint, conversation, trace = kernel._checkpoint_creation_stage(
                user_prompt=prompt,
                memory_context="",
                continue_plan=False,
                local_only=False,
                planning_mode=PlanningMode.AUTO,
                steps=[],
                total_usage={},
                ai_logs=[],
                system_logs=[],
                max_consecutive_tools=5,
                tool_policy_events=[],
            )

        self.assertEqual(trace.summary, "Created ordered checkpoint blueprint")
        self.assertEqual(conversation, [])
        self.assertGreaterEqual(len(blueprint.tasks), 10)
        self.assertTrue(any("Inspect `core/runtime/web.py`" in task.description for task in blueprint.tasks))

    def test_repair_tool_arguments_defaults_list_directory_mode_to_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

            repaired = kernel._repair_tool_arguments(
                ToolCallRequest(
                    call_id="call-1",
                    tool_name="list_directory",
                    arguments={"path": "chatapp", "max_depth": 3},
                )
            )

        self.assertEqual(repaired["mode"], "recursive")

    def test_split_active_checkpoint_skips_context_only_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Inspect backend hooks",
                original_objective="Integrate chat backend with frontend",
                tasks=[
                    CheckpointTask(
                        task_id=1,
                        description="Inspect the existing backend and frontend integration points that the new chat flow must connect to.",
                        expected_artifact="code",
                    )
                ],
                active_task_pointer=0,
            )

            updated = kernel._split_active_checkpoint(blueprint, 0, reason="Execution tool limit reached before the checkpoint completed.")

        self.assertIsNone(updated)

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
                    CheckpointTask(
                        task_id=2,
                        description="Add index.html",
                        allowed_tool_names=("read_file", "write_file"),
                        expects_mutation=True,
                        requires_verification=True,
                        verification_mode="code",
                    ),
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
        self.assertIn("Allowed tools for this checkpoint: `read_file`, `write_file`", prompt)
        self.assertIn("expects a real workspace mutation", prompt)
        self.assertIn("Verification will run after completion using mode: code.", prompt)
        self.assertIn("Completed earlier: Create frontend folder", prompt)
        self.assertIn("Next after this: Add styles.css", prompt)

    def test_execution_scope_for_explicit_file_inspection_omits_list_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(InspectSymbolsTool())
            checkpoint = CheckpointTask(
                task_id=1,
                description="Inspect `core/runtime/web.py` to map the backend request surface.",
                allowed_tool_names=("list_directory", "read_file", "inspect_symbols"),
            )

            scope = kernel._resolve_execution_tool_scope(
                "Integrate the chat backend with frontend",
                checkpoint.description,
                checkpoint=checkpoint,
            )

        self.assertNotIn("list_directory", scope)
        self.assertIn("read_file", scope)
        self.assertIn("inspect_symbols", scope)

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

    def test_repair_tool_arguments_maps_markdown_init_path_to_active_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "I want to integrate chat app to this codebase, add all the files in chatapp in folder (the backend files). Integrate with frontend"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Create `chatapp/__init__.py`",
                tasks=[CheckpointTask(task_id=1, description="Create `chatapp/__init__.py` for the chat app package exports.")],
                active_task_pointer=0,
            )

            repaired = kernel._repair_tool_arguments(
                ToolCallRequest(
                    call_id="call-1",
                    tool_name="write_file",
                    arguments={"path": "interface/website/src/chatapp/**init**.py", "content": "x = 1\n", "mode": "fresh"},
                )
            )

        self.assertEqual(repaired["path"], str((Path(tempdir) / "chatapp" / "__init__.py").resolve()))

    def test_validate_scaffold_tool_call_rejects_empty_write_for_integration_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.active_plan_prompt = "I want to integrate chat app to this codebase, add all the files in chatapp in folder (the backend files). Integrate with frontend"
            kernel.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown="- [ ] Create `chatapp/store.py`",
                tasks=[CheckpointTask(task_id=1, description="Create `chatapp/store.py` for the in-memory chat session store.")],
                active_task_pointer=0,
            )

            error = kernel._validate_scaffold_tool_call(
                "write_file",
                {
                    "path": str((Path(tempdir) / "chatapp" / "store.py").resolve()),
                    "content": "",
                    "mode": "fresh",
                },
            )

        self.assertIn("requires non-empty content", error)

    def test_repair_directory_path_ignores_site_packages_backend_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            (workspace / ".venv" / "lib" / "python3.12" / "site-packages" / "sentence_transformers" / "backend").mkdir(parents=True)
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

            repaired = kernel._repair_directory_path("backend")

        self.assertIsNone(repaired)

    def test_repair_directory_path_ignores_codereference_backend_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            (workspace / "codereferences" / "codex" / "codex-rs" / "app-server-daemon" / "src" / "backend").mkdir(parents=True)
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

            repaired = kernel._repair_directory_path("backend")

        self.assertIsNone(repaired)

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
