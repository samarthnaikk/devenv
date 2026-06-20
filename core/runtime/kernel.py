from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from core.ai import AICore
from core.ai.models import AIResponse, ToolCallRequest
from core.env import load_dotenv
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.tools.base import BaseTool

from .models import AgentState, CheckpointTask, ExecutionBlueprint, PlanningMode, RuntimeTurnResult, ToolExecutionStep
from .local_router import LocalIntentRouter
from .mcp_client import MCPToolClient
from .sandbox import PathSandbox
from .state import resolve_memory_paths

logger = logging.getLogger(__name__)
MAX_EPHEMERAL_TURNS = 4
PLANNING_ALLOWED_TOOLS = frozenset({"list_directory", "read_file", "inspect_symbols"})
READ_ONLY_EXECUTION_TOOLS = frozenset(
    {"list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols", "search_text", "track_symbol"}
)
WRITE_EXECUTION_TOOLS = frozenset({"write_file", "edit_file"})
DELETE_EXECUTION_TOOLS = frozenset({"remove_file"})
SHELL_EXECUTION_TOOLS = frozenset({"run_shell", "run_diagnostics", "audit_changes"})
MEMORY_EXECUTION_TOOLS = frozenset({"manage_memory", "inspect_trace"})
PLANNING_MEMORY_CHAR_LIMIT = 900
EXECUTION_MEMORY_CHAR_LIMIT = 1400
SCAFFOLD_EXECUTION_TOOLS = frozenset({"list_directory", "write_file", "edit_file"})
SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT = 360
PLANNING_SYSTEM_RULE = (
    "Analyze the user's request and produce a sequential markdown checklist using checkbox items like '- [ ] Task'. "
    "Do not invoke modification tools during planning. Stay focused on planning until the checklist is complete. "
    "Prefer concise plans with at most 4 checkpoints."
)
EXECUTION_SYSTEM_RULE = (
    "Work only on the current checkpoint. Do not start future checkpoints. "
    "Use tools only when necessary, and stop after completing the current checkpoint."
)
DIRECT_SYSTEM_RULE = (
    "Answer the user's question directly. First use the memory context if it plausibly contains the answer. "
    "Use tools only if workspace inspection is still needed after considering memory. "
    "If you need a tool, emit a real function call and never print JSON tool snippets in plain text. "
    "Do not create a checklist or execution plan unless the user is asking you to make changes."
)
DIRECT_MEMORY_CHAR_LIMIT = 900


class DevenvKernel:
    def __init__(
        self,
        workspace_path: str,
        db_path: str = "memory.db",
        vector_dir: str = "vectors",
        *,
        memory: MemoryEngine | Any | None = None,
        ai: AICore | Any | None = None,
        tool_client: MCPToolClient | Any | None = None,
    ):
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        load_dotenv(self.workspace_path)
        self.sandbox = PathSandbox(root_path=self.workspace_path)
        resolved_db_path, resolved_vector_dir = resolve_memory_paths(db_path, vector_dir)
        self.memory = memory or _build_memory_engine(resolved_db_path, resolved_vector_dir)
        self.ai = ai or AICore()
        self.tools: dict[str, BaseTool] = {}
        self.ephemeral_history: list[dict[str, Any]] = []
        self.session_id = str(uuid.uuid4())
        self.db_path = resolved_db_path
        self.vector_dir = resolved_vector_dir
        self.state = AgentState.PLANNING
        self.active_blueprint: ExecutionBlueprint | None = None
        self.active_plan_prompt: str | None = None
        self.local_router = LocalIntentRouter()
        self.tool_client = tool_client or MCPToolClient(
            workspace_path=self.workspace_path,
            db_path=db_path,
            vector_dir=vector_dir,
        )

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        self.ai.register_tool(tool)
        logger.info("Registered tool with runtime and AI: tool=%s", tool.name)

    def close(self) -> None:
        if hasattr(self.tool_client, "close"):
            self.tool_client.close()

    def execute_turn(
        self,
        user_prompt: str,
        max_consecutive_tools: int = 5,
        planning_mode: PlanningMode = PlanningMode.AUTO,
        continue_plan: bool = False,
    ) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        self._record_working_memory(conversation)
        memory_context = self._retrieve_memory_context(user_prompt)
        logger.info("Retrieved memory context: chars=%s", len(memory_context))
        system_logs.append(f"Memory context chars: {len(memory_context)}")
        system_logs.append(f"Planning mode: {planning_mode.value}")
        system_logs.append(f"Continue plan: {continue_plan}")
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}
        should_plan = self._should_plan(user_prompt, planning_mode)
        if not should_plan:
            route_decision = self.local_router.decide(user_prompt)
            system_logs.append(
                f"Local route decision: use_local={route_decision.use_local_knowledge} confidence={route_decision.confidence:.3f}"
            )
            if route_decision.use_local_knowledge:
                self.state = AgentState.EXECUTING
                system_logs.append(f"State: {self.state.name}")
                local_response, handled_locally = self._run_local_knowledge_turn(
                    user_prompt=user_prompt,
                    memory_context=memory_context,
                    steps=steps,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                )
                if handled_locally:
                    if local_response:
                        conversation.append({"role": "assistant", "content": local_response})
                    logger.info("Finishing local runtime turn: final_response_present=%s total_steps=%s", local_response is not None, len(steps))
                    self._finalize_turn(user_prompt, local_response or "", conversation)
                    system_logs.append("Turn completed and stored in memory")
                    return RuntimeTurnResult(
                        final_response=local_response,
                        steps=steps,
                        total_usage=total_usage,
                        ai_logs=ai_logs,
                        system_logs=system_logs,
                        state=self.state.name,
                        blueprint=None,
                    )
                system_logs.append("Local knowledge path deferred to direct AI mode")

            self.state = AgentState.EXECUTING
            system_logs.append(f"State: {self.state.name}")
            direct_response = self._run_direct_turn(
                user_prompt=user_prompt,
                memory_context=memory_context,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                max_consecutive_tools=max_consecutive_tools,
            )
            if direct_response:
                conversation.append({"role": "assistant", "content": direct_response})
            logger.info("Finishing direct runtime turn: final_response_present=%s total_steps=%s", direct_response is not None, len(steps))
            self._finalize_turn(user_prompt, direct_response or "", conversation)
            system_logs.append("Turn completed and stored in memory")
            return RuntimeTurnResult(
                final_response=direct_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                state=self.state.name,
                blueprint=None,
            )

        self.state = AgentState.PLANNING
        system_logs.append(f"State: {self.state.name}")
        if continue_plan and self._can_continue_active_plan(user_prompt):
            blueprint = self.active_blueprint or self._parse_markdown_to_blueprint(user_prompt)
            system_logs.append("Resuming existing execution plan")
            planning_conversation = []
        else:
            self.active_plan_prompt = user_prompt

            planning_response, planning_conversation = self._run_planning_phase(
                user_prompt=user_prompt,
                memory_context=memory_context,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                max_consecutive_tools=max_consecutive_tools,
            )
            blueprint = self._parse_markdown_to_blueprint(planning_response or user_prompt)
            self.active_blueprint = blueprint
            system_logs.append(f"Plan checkpoints: {len(blueprint.tasks)}")

        try:
            execution_final_response, plan_complete = self._run_execution_phase(
                user_prompt=user_prompt,
                memory_context=memory_context,
                blueprint=blueprint,
                conversation=planning_conversation,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                max_consecutive_tools=max_consecutive_tools,
                planning_mode=planning_mode,
            )
        except RuntimeError as exc:
            system_logs.append(f"Execution failed: {exc}")
            logger.warning("Execution phase failed: error=%s", exc)
            self._finalize_turn(user_prompt, "", conversation)
            system_logs.append("Turn completed and stored in memory")
            return RuntimeTurnResult(
                final_response=None,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                state=self.state.name,
                blueprint=self.active_blueprint,
                error_message=str(exc),
            )
        if execution_final_response:
            conversation.append({"role": "assistant", "content": execution_final_response})

        if not plan_complete:
            logger.info("Checkpoint execution paused: awaiting next plan continuation")
            self._finalize_turn(user_prompt, execution_final_response or "", conversation)
            system_logs.append("Turn completed and stored in memory")
            return RuntimeTurnResult(
                final_response=execution_final_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                state=self.state.name,
                blueprint=self.active_blueprint,
            )

        verification_ok = self._run_verification_phase(
            blueprint=blueprint,
            steps=steps,
            system_logs=system_logs,
        )
        if not verification_ok:
            self.state = AgentState.PLANNING
            system_logs.append("Verification failed; state reset to PLANNING")
        self.active_plan_prompt = None

        logger.info("Finishing runtime turn: final_response_present=%s total_steps=%s", execution_final_response is not None, len(steps))
        self._finalize_turn(user_prompt, execution_final_response or "", conversation)
        system_logs.append("Turn completed and stored in memory")
        return RuntimeTurnResult(
            final_response=execution_final_response,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            state=self.state.name,
            blueprint=self.active_blueprint,
        )

    def _run_direct_turn(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
    ) -> str | None:
        direct_memory = _focus_memory_context_for_direct_answers(memory_context, DIRECT_MEMORY_CHAR_LIMIT)
        tool_scope = self._resolve_direct_tool_scope(user_prompt)
        system_logs.append(f"Direct memory chars sent: {len(direct_memory)}")
        system_logs.append(f"Direct tool scope size: {len(tool_scope)}")
        conversation = [
            {"role": "system", "content": DIRECT_SYSTEM_RULE},
            {"role": "user", "content": user_prompt},
        ]
        tool_iterations = 0

        while True:
            ai_response = self.ai.chat(
                messages=list(conversation),
                memory_context=direct_memory,
                tool_names=tool_scope,
            )
            _merge_usage(total_usage, ai_response.usage)
            ai_logs.append(
                f"Direct response: finish_reason={ai_response.finish_reason}, tool_calls={len(ai_response.tool_calls)}, total_tokens={ai_response.usage.get('total_tokens', 0)}"
            )
            inline_tool_call = _coerce_inline_tool_call(ai_response.content, tool_scope)
            effective_tool_calls = list(ai_response.tool_calls)
            if inline_tool_call is not None:
                effective_tool_calls = [inline_tool_call]
                ai_logs.append(f"Recovered inline tool request: {inline_tool_call.tool_name}")
                system_logs.append(f"Recovered inline tool request: {inline_tool_call.tool_name}")

            if effective_tool_calls:
                tool_call = effective_tool_calls[0]
                tool_iterations += 1
                if tool_iterations > max_consecutive_tools:
                    raise RuntimeError("Direct tool limit reached before the request could be completed.")
                if len(effective_tool_calls) > 1:
                    system_logs.append(
                        f"Direct mode deferred {len(effective_tool_calls) - 1} extra tool call(s) to preserve bounded execution"
                    )
                ai_logs.append(f"Tool requested: {tool_call.tool_name}")
                conversation.append(_assistant_tool_call_message(ai_response, [tool_call], content_override=ai_response.content if inline_tool_call is None else None))
                step = self._execute_tool_call(tool_call)
                steps.append(step)
                system_logs.append(f"Tool step {len(steps)}: {tool_call.tool_name} success={step.success}")
                conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                continue

            if ai_response.content:
                ai_logs.append("Assistant produced direct response")
            return ai_response.content

    def _run_local_knowledge_turn(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        steps: list[ToolExecutionStep],
        ai_logs: list[str],
        system_logs: list[str],
    ) -> tuple[str | None, bool]:
        ai_logs.append("Local router selected knowledge mode")
        memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
        if memory_answer is not None:
            ai_logs.append("Local knowledge answer assembled from memory")
            return memory_answer, True

        candidate_path = self._resolve_workspace_candidate(user_prompt)
        if candidate_path is None:
            ai_logs.append("Local knowledge mode found no strong memory or workspace candidate")
            return None, False

        tool_call = ToolCallRequest(
            call_id=f"local_{uuid.uuid4().hex[:10]}",
            tool_name="list_directory",
            arguments={"path": candidate_path, "mode": "recursive", "max_depth": 2},
        )
        step = self._execute_tool_call(tool_call)
        steps.append(step)
        system_logs.append(f"Tool step {len(steps)}: list_directory success={step.success}")
        if not step.success:
            ai_logs.append("Local knowledge mode could not inspect workspace candidate")
            return None, False

        ai_logs.append("Local knowledge answer assembled from workspace structure")
        return _summarize_directory_listing(candidate_path, step.output), True

    def _run_planning_phase(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        planning_memory = _trim_memory_context(memory_context, PLANNING_MEMORY_CHAR_LIMIT)
        system_logs.append(f"Planning memory chars sent: {len(planning_memory)}")
        system_logs.append("Planning tool scope size: 0")
        conversation = [
            {"role": "system", "content": PLANNING_SYSTEM_RULE},
            {"role": "user", "content": user_prompt},
        ]
        planning_message_count = 0
        while True:
            ai_response = self.ai.chat(
                messages=list(conversation),
                memory_context=planning_memory,
                tool_names=[],
            )
            _merge_usage(total_usage, ai_response.usage)
            ai_logs.append(
                f"Planning response: finish_reason={ai_response.finish_reason}, tool_calls={len(ai_response.tool_calls)}, total_tokens={ai_response.usage.get('total_tokens', 0)}"
            )
            if ai_response.tool_calls:
                for tool_call in ai_response.tool_calls:
                    if tool_call.tool_name not in PLANNING_ALLOWED_TOOLS:
                        logger.warning("Blocked non-planning tool during planning: tool=%s", tool_call.tool_name)
                        system_logs.append(f"Blocked planning tool call: {tool_call.tool_name}")
                        conversation.append(_assistant_tool_call_message(ai_response, [tool_call]))
                        conversation.append(
                            _tool_message(
                                tool_call.call_id,
                                tool_call.tool_name,
                                "Planning phase active. Emit a checkbox plan before invoking modification or non-planning tools.",
                            )
                        )
                        break
                    if len(steps) >= max_consecutive_tools:
                        raise RuntimeError("Planning tool limit reached before a blueprint could be produced.")
                    step = self._execute_tool_call(tool_call)
                    steps.append(step)
                    ai_logs.append(f"Planning tool requested: {tool_call.tool_name}")
                    system_logs.append(f"Planning tool: {tool_call.tool_name} success={step.success}")
                    conversation.append(_assistant_tool_call_message(ai_response, [tool_call]))
                    conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                planning_message_count += 1
                if planning_message_count > max_consecutive_tools:
                    raise RuntimeError("Planning exceeded the configured tool limit.")
                continue

            content = ai_response.content
            if content:
                conversation.append({"role": "assistant", "content": content})
                ai_logs.append("Planning blueprint generated")
            return content, conversation

    def _run_execution_phase(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        blueprint: ExecutionBlueprint,
        conversation: list[dict[str, Any]],
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
        planning_mode: PlanningMode,
    ) -> tuple[str | None, bool]:
        self.state = AgentState.EXECUTING
        system_logs.append(f"State: {self.state.name}")
        final_response: str | None = None
        working_blueprint = blueprint
        checkpoint_indexes = self._execution_checkpoint_indexes(working_blueprint)

        for index in checkpoint_indexes:
            task = working_blueprint.tasks[index]
            working_blueprint = _set_active_task(working_blueprint, index)
            self.active_blueprint = working_blueprint
            system_logs.append(f"Current checkpoint {index + 1}/{len(working_blueprint.tasks)}: {task.description}")
            scoped_tool_names = self._resolve_execution_tool_scope(user_prompt, task.description)
            execution_memory = self._resolve_execution_memory(
                user_prompt=user_prompt,
                task_description=task.description,
                memory_context=memory_context,
            )
            system_logs.append(f"Execution memory chars sent: {len(execution_memory)}")
            system_logs.append(f"Execution tool scope size: {len(scoped_tool_names)}")
            step_conversation = [
                {"role": "system", "content": EXECUTION_SYSTEM_RULE},
                {
                    "role": "user",
                    "content": self._build_execution_prompt(
                        user_prompt=user_prompt,
                        checkpoint_index=index + 1,
                        total_checkpoints=len(working_blueprint.tasks),
                        task_description=task.description,
                        blueprint=working_blueprint,
                    ),
                },
            ]
            tool_iterations = 0
            checkpoint_requires_mutation = self._checkpoint_requires_mutation(user_prompt, task.description) and any(
                tool_name in scoped_tool_names for tool_name in (*WRITE_EXECUTION_TOOLS, *DELETE_EXECUTION_TOOLS)
            )

            while True:
                ai_response = self.ai.chat(
                    messages=list(step_conversation),
                    memory_context=execution_memory,
                    tool_names=scoped_tool_names,
                )
                _merge_usage(total_usage, ai_response.usage)
                ai_logs.append(
                    f"Execution response: checkpoint={index + 1}, finish_reason={ai_response.finish_reason}, tool_calls={len(ai_response.tool_calls)}, total_tokens={ai_response.usage.get('total_tokens', 0)}"
                )
                if ai_response.tool_calls:
                    tool_call = ai_response.tool_calls[0]
                    tool_iterations += 1
                    if tool_iterations > max_consecutive_tools:
                        raise RuntimeError("Execution tool limit reached before the checkpoint completed.")
                    if len(ai_response.tool_calls) > 1:
                        system_logs.append(
                            f"Checkpoint {index + 1}: deferred {len(ai_response.tool_calls) - 1} extra tool call(s) to preserve single-step execution"
                        )
                    ai_logs.append(f"Tool requested: {tool_call.tool_name}")
                    step_conversation.append(_assistant_tool_call_message(ai_response, [tool_call]))
                    step = self._execute_tool_call(tool_call)
                    steps.append(step)
                    system_logs.append(f"Tool step {len(steps)}: {tool_call.tool_name} success={step.success}")
                    step_conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                    continue

                if checkpoint_requires_mutation and tool_iterations == 0:
                    ai_logs.append(f"Checkpoint requires mutation before completion: {task.description}")
                    system_logs.append(f"Checkpoint {index + 1} requires a file mutation tool before completion")
                    step_conversation.append(
                        {
                            "role": "assistant",
                            "content": ai_response.content
                            or "I described the change but did not execute it.",
                        }
                    )
                    step_conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "You have not completed this checkpoint yet. "
                                "Use a real workspace modification tool such as write_file or edit_file, "
                                "then stop after the tool succeeds."
                            ),
                        }
                    )
                    tool_iterations += 1
                    if tool_iterations > max_consecutive_tools:
                        raise RuntimeError("Execution tool limit reached before the checkpoint completed.")
                    continue

                final_response = ai_response.content or final_response
                trace_log = _summarize_execution_note(ai_response.content)
                working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                self.active_blueprint = working_blueprint
                ai_logs.append(f"Checkpoint completed: {task.description}")
                system_logs.append(f"Checkpoint {index + 1} completed")
                break

        plan_complete = _next_incomplete_task_index(working_blueprint) is None
        if not plan_complete:
            self.state = AgentState.EXECUTING
            system_logs.append("Execution paused after one checkpoint")
        return final_response, plan_complete

    def _run_verification_phase(
        self,
        *,
        blueprint: ExecutionBlueprint,
        steps: list[ToolExecutionStep],
        system_logs: list[str],
    ) -> bool:
        self.state = AgentState.VERIFYING
        system_logs.append(f"State: {self.state.name}")
        diagnostics_tool = self.tools.get("run_diagnostics")
        if diagnostics_tool is None:
            system_logs.append("Verification skipped: run_diagnostics is not registered")
            source_blueprint = self.active_blueprint or blueprint
            self.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown=source_blueprint.raw_plan_markdown,
                tasks=list(source_blueprint.tasks),
                active_task_pointer=len(source_blueprint.tasks),
                verification_passed=True,
            )
            return True

        verification_results: list[bool] = []
        for mode in ("tests", "types"):
            result = diagnostics_tool.execute(mode=mode, target_path=self.workspace_path)
            step = ToolExecutionStep(
                step_id=f"verify-{mode}",
                tool_name="run_diagnostics",
                arguments={"mode": mode, "target_path": self.workspace_path},
                output=_format_tool_output(result.output, result.data),
                success=result.success,
                is_sandboxed_violation=False,
            )
            steps.append(step)
            verification_results.append(step.success)
            system_logs.append(f"Verification {mode}: success={step.success}")

        verification_passed = all(verification_results)
        source_blueprint = self.active_blueprint or blueprint
        self.active_blueprint = ExecutionBlueprint(
            raw_plan_markdown=source_blueprint.raw_plan_markdown,
            tasks=list(source_blueprint.tasks),
            active_task_pointer=len(source_blueprint.tasks),
            verification_passed=verification_passed,
        )
        return verification_passed

    def _parse_markdown_to_blueprint(self, markdown_text: str) -> ExecutionBlueprint:
        task_pattern = re.compile(r"^\s*(?:[-*]|\d+\.)\s*\[(?P<status>[ xX])\]\s*(?P<description>.+?)\s*$")
        tasks: list[CheckpointTask] = []
        for line in markdown_text.splitlines():
            match = task_pattern.match(line)
            if not match:
                continue
            tasks.append(
                CheckpointTask(
                    task_id=len(tasks) + 1,
                    description=match.group("description").strip(),
                    is_completed=match.group("status").lower() == "x",
                )
            )

        if not tasks:
            tasks = self._parse_step_sections(markdown_text)

        if not tasks:
            fallback = markdown_text.strip() or "Handle the user request."
            tasks.append(CheckpointTask(task_id=1, description=fallback))

        return ExecutionBlueprint(raw_plan_markdown=markdown_text, tasks=tasks)

    def _parse_step_sections(self, markdown_text: str) -> list[CheckpointTask]:
        step_heading = re.compile(r"^\s*(?:#{1,6}\s*)?step\s+(?P<number>\d+)\s*:\s*(?P<title>.+?)\s*$", re.IGNORECASE)
        tasks: list[CheckpointTask] = []
        current_title: str | None = None
        detail_lines: list[str] = []
        in_code_block = False

        def flush() -> None:
            nonlocal current_title, detail_lines
            if current_title is None:
                return
            description = current_title.strip()
            detail = _summarize_step_detail(detail_lines)
            if detail:
                description = f"{description}: {detail}"
            tasks.append(CheckpointTask(task_id=len(tasks) + 1, description=description))
            current_title = None
            detail_lines = []

        for raw_line in markdown_text.splitlines():
            line = raw_line.rstrip()
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            match = step_heading.match(line)
            if match:
                flush()
                current_title = match.group("title").strip()
                continue

            if current_title is not None:
                detail_lines.append(line)

        flush()
        return tasks

    def _resolve_execution_tool_scope(self, user_prompt: str, task_description: str) -> list[str]:
        text = f"{user_prompt} {task_description}".lower()
        if self._is_scaffold_request(text):
            return sorted(name for name in SCAFFOLD_EXECUTION_TOOLS if name in self.tools)

        tool_names = set(READ_ONLY_EXECUTION_TOOLS)

        if any(token in text for token in ("create", "add", "write", "build", "generate", "html", "css", "js", "frontend", "folder", "file")):
            tool_names.update(WRITE_EXECUTION_TOOLS)
        if any(token in text for token in ("edit", "update", "modify", "change", "patch", "refactor")):
            tool_names.update(WRITE_EXECUTION_TOOLS)
        if any(token in text for token in ("delete", "remove", "cleanup")):
            tool_names.update(DELETE_EXECUTION_TOOLS)
        if any(token in text for token in ("run", "test", "verify", "diagnostic", "lint", "typecheck", "types", "shell")):
            tool_names.update(SHELL_EXECUTION_TOOLS)
        if any(token in text for token in ("memory", "recall", "trace", "history")):
            tool_names.update(MEMORY_EXECUTION_TOOLS)

        if "run_shell" in tool_names:
            tool_names.add("audit_changes")
        return sorted(name for name in tool_names if name in self.tools)

    def _resolve_execution_memory(self, *, user_prompt: str, task_description: str, memory_context: str) -> str:
        text = f"{user_prompt} {task_description}".lower()
        if self._is_scaffold_request(text):
            return _trim_memory_context(memory_context, SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT)
        return _trim_memory_context(memory_context, EXECUTION_MEMORY_CHAR_LIMIT)

    def _build_execution_prompt(
        self,
        *,
        user_prompt: str,
        checkpoint_index: int,
        total_checkpoints: int,
        task_description: str,
        blueprint: ExecutionBlueprint,
    ) -> str:
        target_path_hint = self._derive_scaffold_target_path(user_prompt, task_description)
        plan_context = self._build_checkpoint_context(blueprint, checkpoint_index - 1)
        if self._is_scaffold_request(f"{user_prompt} {task_description}".lower()):
            lines = [
                f"Goal: {user_prompt}\n"
                f"Checkpoint {checkpoint_index}/{total_checkpoints}: {task_description}",
            ]
            if target_path_hint:
                lines.append(f"All new files for this request must stay under: {target_path_hint}")
            if plan_context:
                lines.append(plan_context)
            lines.append("Complete only this checkpoint. Use the smallest valid tool call and stop after it succeeds.")
            return "\n".join(lines)
        return (
            f"Original request:\n{user_prompt}\n\n"
            f"Current checkpoint ({checkpoint_index}/{total_checkpoints}):\n- [ ] {task_description}\n\n"
            "Complete only this checkpoint, then stop."
        )

    def _is_scaffold_request(self, text: str) -> bool:
        creation_markers = ("create", "make", "add", "build", "generate")
        frontend_markers = ("html", "css", "js", "javascript", "frontend", "ui")
        non_backend_markers = ("dont connect with backend", "don't connect with backend", "no need to connect to backend")
        file_markers = ("folder", "file", "calendar")
        return (
            any(marker in text for marker in creation_markers)
            and any(marker in text for marker in frontend_markers)
            and any(marker in text for marker in file_markers)
        ) or any(marker in text for marker in non_backend_markers)

    def _requires_planning(self, user_prompt: str) -> bool:
        text = user_prompt.lower()
        if self._is_scaffold_request(text):
            return True

        change_markers = (
            "create",
            "make",
            "add",
            "build",
            "generate",
            "write",
            "edit",
            "update",
            "modify",
            "change",
            "fix",
            "refactor",
            "delete",
            "remove",
            "rename",
            "move",
            "implement",
            "patch",
            "complete",
        )
        return any(marker in text for marker in change_markers)

    def _should_plan(self, user_prompt: str, planning_mode: PlanningMode) -> bool:
        if planning_mode is PlanningMode.FORCE_PLAN:
            return True
        if planning_mode is PlanningMode.FORCE_DIRECT:
            return False
        return self._requires_planning(user_prompt)

    def _can_continue_active_plan(self, user_prompt: str) -> bool:
        if self.active_blueprint is None or self.active_plan_prompt != user_prompt:
            return False
        return _next_incomplete_task_index(self.active_blueprint) is not None

    def _execution_checkpoint_indexes(self, blueprint: ExecutionBlueprint) -> list[int]:
        start_index = _next_incomplete_task_index(blueprint)
        if start_index is None:
            return []
        return [start_index]

    def _build_checkpoint_context(self, blueprint: ExecutionBlueprint, task_index: int) -> str:
        completed = [task.description for task in blueprint.tasks[:task_index] if task.is_completed]
        remaining = [task.description for task in blueprint.tasks[task_index + 1 :] if not task.is_completed]
        lines: list[str] = []
        if completed:
            lines.append(f"Completed earlier: {completed[-1]}")
        if remaining:
            lines.append(f"Next after this: {remaining[0]}")
        return "\n".join(lines)

    def _checkpoint_requires_mutation(self, user_prompt: str, task_description: str) -> bool:
        text = f"{user_prompt} {task_description}".lower()
        mutation_markers = (
            "create",
            "make",
            "add",
            "build",
            "generate",
            "write",
            "edit",
            "update",
            "modify",
            "change",
            "fix",
            "refactor",
            "delete",
            "remove",
            "rename",
            "move",
            "implement",
            "patch",
            "html",
            "css",
            "js",
            "javascript",
            "frontend",
            "file",
            "folder",
        )
        return any(marker in text for marker in mutation_markers)

    def _resolve_direct_tool_scope(self, user_prompt: str) -> list[str]:
        text = user_prompt.lower()
        tool_names = set(READ_ONLY_EXECUTION_TOOLS)
        if any(token in text for token in ("memory", "recall", "trace", "history", "earlier", "previous")):
            tool_names.update(MEMORY_EXECUTION_TOOLS)
        return sorted(name for name in tool_names if name in self.tools)

    def _resolve_workspace_candidate(self, user_prompt: str) -> str | None:
        prompt_tokens = [token for token in re.findall(r"[a-z0-9_]+", user_prompt.lower()) if len(token) >= 3]
        try:
            entries = sorted(Path(self.workspace_path).iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return None

        names = [entry.name.lower() for entry in entries if entry.is_dir()]
        for token in prompt_tokens:
            matches = get_close_matches(token, names, n=1, cutoff=0.72)
            if matches:
                matched_name = matches[0]
                for entry in entries:
                    if entry.is_dir() and entry.name.lower() == matched_name:
                        return str(entry)
        return self.workspace_path if entries else None

    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolExecutionStep:
        logger.info("Intercepted tool call: tool=%s arguments=%s", tool_call.tool_name, tool_call.arguments)
        unsafe_argument = self.sandbox.find_unsafe_argument(tool_call.arguments)
        if unsafe_argument is not None:
            _key, value = unsafe_argument
            logger.warning("Sandbox violation detected: tool=%s path=%s", tool_call.tool_name, value)
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=self.sandbox.violation_message(value),
                success=False,
                is_sandboxed_violation=True,
            )

        tool = self.tools.get(tool_call.tool_name)
        if tool is None:
            logger.error("Requested tool is not registered: tool=%s", tool_call.tool_name)
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=f"Tool '{tool_call.tool_name}' is not registered in the runtime.",
                success=False,
                is_sandboxed_violation=False,
            )

        normalized_arguments = self.sandbox.normalize_arguments(self._repair_tool_arguments(tool_call))
        scaffold_validation_error = self._validate_scaffold_tool_call(tool_call.tool_name, normalized_arguments)
        if scaffold_validation_error is not None:
            logger.warning("Rejected scaffold tool call: tool=%s error=%s", tool_call.tool_name, scaffold_validation_error)
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=normalized_arguments,
                output=scaffold_validation_error,
                success=False,
                is_sandboxed_violation=False,
            )
        logger.info("Executing MCP tool: tool=%s normalized_arguments=%s", tool_call.tool_name, normalized_arguments)
        result = self.tool_client.call_tool(tool_call.tool_name, normalized_arguments)
        logger.info("MCP tool finished: tool=%s success=%s is_error=%s", tool_call.tool_name, result.success, result.is_error)
        return ToolExecutionStep(
            step_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            arguments=normalized_arguments,
            output=_format_tool_output(result.output, result.data),
            success=result.success and not result.is_error,
            is_sandboxed_violation=False,
        )

    def _repair_tool_arguments(self, tool_call: ToolCallRequest) -> dict[str, Any]:
        arguments = dict(tool_call.arguments)
        if "max_depth" in arguments and isinstance(arguments["max_depth"], str) and arguments["max_depth"].isdigit():
            arguments["max_depth"] = int(arguments["max_depth"])

        if tool_call.tool_name == "list_directory":
            path_value = arguments.get("path")
            if isinstance(path_value, str):
                repaired_path = self._repair_directory_path(path_value)
                if repaired_path is not None:
                    arguments["path"] = repaired_path
        if tool_call.tool_name in WRITE_EXECUTION_TOOLS | DELETE_EXECUTION_TOOLS | frozenset({"edit_file"}):
            path_value = arguments.get("path")
            if isinstance(path_value, str):
                target_path_hint = self._active_scaffold_target_path()
                if target_path_hint:
                    repaired_path = self._repair_scaffold_path(path_value, target_path_hint)
                    if repaired_path is not None:
                        arguments["path"] = repaired_path

        return arguments

    def _active_scaffold_target_path(self) -> str | None:
        prompt = self.active_plan_prompt or ""
        task_description = ""
        if self.active_blueprint and 0 <= self.active_blueprint.active_task_pointer < len(self.active_blueprint.tasks):
            task_description = self.active_blueprint.tasks[self.active_blueprint.active_task_pointer].description
        return self._derive_scaffold_target_path(prompt, task_description)

    def _derive_scaffold_target_path(self, user_prompt: str, task_description: str = "") -> str | None:
        combined = f"{user_prompt} {task_description}".strip().lower()
        if not self._is_scaffold_request(combined):
            return None
        direct_match = re.search(r"\b([a-z0-9_-]+/[a-z0-9_/-]+)\b", combined)
        if direct_match:
            return direct_match.group(1).strip("/")
        nested_match = re.search(r"\b([a-z0-9_-]+)\s+folder\s+(?:inside|in|under)\s+([a-z0-9_/-]+)\b", combined)
        if nested_match:
            child, parent = nested_match.groups()
            return f"{parent.strip('/')}/{child.strip('/')}"
        if "frontend" in combined and "calendar" in combined:
            return "calendar/frontend"
        return None

    def _repair_scaffold_path(self, requested_path: str, target_path_hint: str) -> str | None:
        candidate = Path(requested_path)
        if candidate.is_absolute():
            return None

        target = Path(target_path_hint)
        if candidate.parts[: len(target.parts)] == target.parts:
            return requested_path

        if candidate.parts and candidate.parts[0] == target.name:
            return str(target.parent / candidate).replace("\\", "/")

        if len(candidate.parts) == 1:
            return str(target / candidate.name).replace("\\", "/")

        return None

    def _validate_scaffold_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        target_path_hint = self._active_scaffold_target_path()
        if target_path_hint is None or tool_name not in WRITE_EXECUTION_TOOLS:
            return None

        path_value = arguments.get("path")
        content_value = arguments.get("content")
        if not isinstance(path_value, str):
            return None

        target_path = (Path(self.workspace_path) / target_path_hint).resolve()
        requested_path = Path(path_value).expanduser()
        if not requested_path.is_absolute():
            requested_path = (Path(self.workspace_path) / requested_path).resolve()

        if requested_path == target_path and isinstance(content_value, str) and not content_value.strip():
            return (
                f"Use write_file on a file inside {target_path_hint} such as "
                f"{target_path_hint}/index.html, not on the folder itself."
            )
        return None

    def _repair_directory_path(self, requested_path: str) -> str | None:
        candidate = Path(requested_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.workspace_path) / candidate

        if candidate.exists():
            return str(candidate.resolve())

        requested_name = candidate.name.lower()
        try:
            directories = [entry for entry in Path(self.workspace_path).iterdir() if entry.is_dir()]
        except OSError:
            return None

        names = [entry.name.lower() for entry in directories]
        matches = get_close_matches(requested_name, names, n=1, cutoff=0.5)
        if matches:
            matched_name = matches[0]
            for entry in directories:
                if entry.name.lower() == matched_name:
                    logger.info("Repaired missing directory path: requested=%s repaired=%s", requested_path, entry)
                    return str(entry.resolve())

        return None

    def _finalize_turn(self, user_prompt: str, final_response: str, conversation: list[dict[str, Any]]) -> None:
        self.ephemeral_history = _compact_conversation(conversation, max_turns=MAX_EPHEMERAL_TURNS)
        self._record_working_memory(self.ephemeral_history)
        logger.info(
            "Compacted runtime history: retained_messages=%s raw_messages=%s",
            len(self.ephemeral_history),
            len(conversation),
        )
        logger.info("Recording episodic log for completed turn")
        try:
            log_id = self.memory.add_episodic_log(
                user_prompt,
                final_response,
                metadata={
                    "workspace_path": self.workspace_path,
                    "session_id": self.session_id,
                },
            )
            logger.info("Recorded episodic log: log_id=%s", log_id)
            if hasattr(self.memory, "run_consolidation"):
                result = self.memory.run_consolidation()
                logger.info(
                    "Memory consolidation finished: processed_logs=%s created_nodes=%s updated_nodes=%s",
                    getattr(result, "processed_logs", 0),
                    len(getattr(result, "created_nodes", ())),
                    len(getattr(result, "updated_nodes", ())),
                )
        except Exception as exc:
            logger.warning("Failed to record episodic log; continuing without persisted memory: error=%s", exc)

    def _record_working_memory(self, conversation: list[dict[str, Any]]) -> None:
        try:
            compact_messages = _compact_conversation(conversation, max_turns=MAX_EPHEMERAL_TURNS)
            self.memory.record_working_memory(
                messages=compact_messages,
                active_state={
                    "workspace_path": self.workspace_path,
                    "session_id": self.session_id,
                },
            )
        except Exception as exc:
            logger.warning("Failed to record working memory; continuing: error=%s", exc)

    def _retrieve_memory_context(self, user_prompt: str) -> str:
        try:
            result = self.memory.retrieve_context(user_prompt)
            self._persist_last_retrieval_trace(getattr(result, "trace", None))
            return result.markdown_context
        except Exception as exc:
            logger.warning("Memory retrieval failed; continuing without memory context: error=%s", exc)
            return ""

    def _persist_last_retrieval_trace(self, trace: Any) -> None:
        if trace is None:
            return
        store = getattr(self.memory, "store", None)
        if store is None or not hasattr(store, "set_state"):
            return
        try:
            store.set_state("last_retrieval_trace", json.dumps(asdict(trace), sort_keys=True))
        except Exception as exc:
            logger.warning("Failed to persist retrieval trace state: error=%s", exc)


def _assistant_tool_call_message(
    ai_response: AIResponse,
    tool_calls: list[ToolCallRequest] | None = None,
    content_override: str | None = None,
) -> dict[str, Any]:
    selected_tool_calls = tool_calls or list(ai_response.tool_calls)
    return {
        "role": "assistant",
        "content": ai_response.content if content_override is None else content_override,
        "tool_calls": [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in selected_tool_calls
        ],
    }


def _tool_message(call_id: str, tool_name: str, output: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": output,
    }


def _format_tool_output(output: str, data: dict[str, Any]) -> str:
    if not data:
        return output
    return f"{output}\n{json.dumps(data, sort_keys=True)}"


def _merge_usage(total_usage: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total_usage[key] = total_usage.get(key, 0) + value


def _compact_conversation(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        retained.append({"role": role, "content": content})

    return retained[-(max_turns * 2) :]


def _build_memory_engine(db_path: str, vector_dir: str) -> MemoryEngine:
    try:
        return MemoryEngine(db_path=db_path, vector_dir=vector_dir)
    except Exception as exc:
        logger.warning("Falling back to hashing memory embedder: error=%s", exc)
        return MemoryEngine(
            db_path=db_path,
            vector_dir=vector_dir,
            embedder=HashingEmbedder(dimension=384),
        )


def _trim_memory_context(memory_context: str, char_limit: int) -> str:
    stripped = memory_context.strip()
    if len(stripped) <= char_limit:
        return stripped

    lines: list[str] = []
    current_length = 0
    for line in stripped.splitlines():
        next_length = current_length + len(line) + (1 if lines else 0)
        if next_length > char_limit:
            break
        lines.append(line)
        current_length = next_length

    if not lines:
        return stripped[:char_limit].rstrip()
    return "\n".join(lines).rstrip()


def _focus_memory_context_for_direct_answers(memory_context: str, char_limit: int) -> str:
    stripped = memory_context.strip()
    if not stripped:
        return ""

    retrieved_header = "## Retrieved Memory"
    header_index = stripped.find(retrieved_header)
    if header_index >= 0:
        focused = stripped[header_index:]
        return _trim_memory_context(focused, char_limit)
    return _trim_memory_context(stripped, char_limit)


def _coerce_inline_tool_call(content: str | None, allowed_tools: list[str]) -> ToolCallRequest | None:
    if not isinstance(content, str) or not content.strip() or not allowed_tools:
        return None

    inline_payload = _extract_json_block(content)
    if inline_payload is None:
        return None

    candidates: list[dict[str, Any]] = []
    if isinstance(inline_payload, dict):
        candidates = [inline_payload]
    elif isinstance(inline_payload, list):
        candidates = [item for item in inline_payload if isinstance(item, dict)]

    for candidate in candidates:
        tool_name = candidate.get("name")
        parameters = candidate.get("parameters")
        if not isinstance(tool_name, str) or tool_name not in allowed_tools:
            continue
        if not isinstance(parameters, dict):
            continue
        return ToolCallRequest(
            call_id=f"inline_{uuid.uuid4().hex[:10]}",
            tool_name=tool_name,
            arguments=parameters,
        )

    return None


def _extract_json_block(content: str) -> dict[str, Any] | list[Any] | None:
    stripped = content.strip()
    candidates = [stripped]
    for opener in ("\n[", "\n{"):
        index = stripped.find(opener)
        if index >= 0:
            candidates.append(stripped[index + 1 :].strip())

    for candidate in candidates:
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def _answer_from_retrieved_memory(user_prompt: str, memory_context: str) -> str | None:
    if not memory_context.strip():
        return None

    retrieved_header = "## Retrieved Memory"
    header_index = memory_context.find(retrieved_header)
    if header_index < 0:
        return None

    lines = memory_context[header_index:].splitlines()[1:]
    bullet_lines = []
    for line in lines:
        if not line.startswith("- "):
            continue
        bullet = line[2:].strip()
        bullet_lower = bullet.lower()
        if bullet_lower.startswith("prompt:") or bullet_lower.startswith("[workspace] workspace:"):
            continue
        bullet_lines.append(bullet)
    if not bullet_lines:
        return None

    prompt_tokens = {token for token in re.findall(r"[a-z0-9_]+", user_prompt.lower()) if len(token) >= 4}
    ranked: list[tuple[int, str]] = []
    for line in bullet_lines:
        line_lower = line.lower()
        overlap = sum(1 for token in prompt_tokens if token in line_lower)
        ranked.append((overlap, line))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_overlap = ranked[0][0]
    if best_overlap <= 0:
        return None

    selected = [line for overlap, line in ranked[:3] if overlap == best_overlap or overlap > 0]
    if not selected:
        return None
    cleaned = [_clean_memory_line(line) for line in selected]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return "\n\n".join(cleaned)


def _summarize_directory_listing(candidate_path: str, output: str) -> str:
    lines = output.splitlines()
    relative_paths: list[str] = []
    for line in lines:
        if '"relative_path":' in line:
            fragment = line.split('"relative_path":', 1)[1].strip().strip('",')
            if fragment:
                relative_paths.append(fragment.strip('"'))
    unique_paths = list(dict.fromkeys(relative_paths))
    if not unique_paths:
        return f"I inspected `{candidate_path}` locally, but I didn't find enough structured file evidence to summarize it yet."

    preview = ", ".join(unique_paths[:6])
    return f"I inspected `{candidate_path}` locally. Relevant paths I found: {preview}."


def _clean_memory_line(line: str) -> str:
    cleaned = line.strip()
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[1].strip()
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
    cleaned = re.sub(r"^(episodic memory|episode)\s+[a-f0-9-]+:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _set_active_task(blueprint: ExecutionBlueprint, task_index: int) -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        tasks=list(blueprint.tasks),
        active_task_pointer=task_index,
        verification_passed=blueprint.verification_passed,
    )


def _mark_checkpoint_completed(blueprint: ExecutionBlueprint, task_index: int, trace_log: str) -> ExecutionBlueprint:
    tasks: list[CheckpointTask] = []
    for index, task in enumerate(blueprint.tasks):
        if index == task_index:
            tasks.append(
                CheckpointTask(
                    task_id=task.task_id,
                    description=task.description,
                    is_completed=True,
                    execution_trace_log=trace_log,
                )
            )
        else:
            tasks.append(task)

    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        tasks=tasks,
        active_task_pointer=min(task_index + 1, len(tasks)),
        verification_passed=blueprint.verification_passed,
    )


def _next_incomplete_task_index(blueprint: ExecutionBlueprint) -> int | None:
    for index, task in enumerate(blueprint.tasks):
        if not task.is_completed:
            return index
    return None


def _summarize_step_detail(lines: list[str]) -> str:
    cleaned: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned:
                break
            continue
        if line.lower() in {"html", "css", "javascript", "js"}:
            continue
        if line.startswith("<") or line.startswith("{") or line.startswith("const ") or line.startswith("function "):
            break
        cleaned.append(re.sub(r"\s+", " ", line))
        if len(" ".join(cleaned)) >= 140:
            break

    summary = " ".join(cleaned).strip()
    summary = re.sub(r"[`*_#]+", "", summary)
    return summary[:160].rstrip(" :;,-")


def _summarize_execution_note(content: str | None) -> str:
    if not content or not content.strip():
        return "Checkpoint completed via tool execution."

    lines: list[str] = []
    in_code_block = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        if line.lower() in {"html", "css", "javascript", "js", "python"}:
            continue
        if line.startswith("<") or line.startswith("{") or line.startswith("const ") or line.startswith("function "):
            continue
        lines.append(re.sub(r"\s+", " ", line))
        if len(" ".join(lines)) >= 180:
            break

    summary = " ".join(lines).strip()
    if not summary:
        return "Checkpoint completed via tool execution."
    summary = re.sub(r"[`*_#]+", "", summary)
    return summary[:180].rstrip(" :;,-")
