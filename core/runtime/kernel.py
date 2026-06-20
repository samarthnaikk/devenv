from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.ai import AICore
from core.ai.models import AIResponse, ToolCallRequest
from core.env import load_dotenv
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.tools.base import BaseTool

from .models import AgentState, CheckpointTask, ExecutionBlueprint, RuntimeTurnResult, ToolExecutionStep
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

    def execute_turn(self, user_prompt: str, max_consecutive_tools: int = 5) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        self._record_working_memory(conversation)
        memory_context = self._retrieve_memory_context(user_prompt)
        logger.info("Retrieved memory context: chars=%s", len(memory_context))
        system_logs.append(f"Memory context chars: {len(memory_context)}")
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}
        if not self._requires_planning(user_prompt):
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

        execution_final_response = self._run_execution_phase(
            user_prompt=user_prompt,
            memory_context=memory_context,
            blueprint=blueprint,
            conversation=planning_conversation,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            max_consecutive_tools=max_consecutive_tools,
        )
        if execution_final_response:
            conversation.append({"role": "assistant", "content": execution_final_response})

        verification_ok = self._run_verification_phase(
            blueprint=blueprint,
            steps=steps,
            system_logs=system_logs,
        )
        if not verification_ok:
            self.state = AgentState.PLANNING
            system_logs.append("Verification failed; state reset to PLANNING")

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
    ) -> str | None:
        self.state = AgentState.EXECUTING
        system_logs.append(f"State: {self.state.name}")
        final_response: str | None = None
        working_blueprint = blueprint

        for index, task in enumerate(working_blueprint.tasks):
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
                    ),
                },
            ]
            tool_iterations = 0

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

                final_response = ai_response.content or final_response
                trace_log = ai_response.content or "Checkpoint completed via tool execution."
                working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                self.active_blueprint = working_blueprint
                ai_logs.append(f"Checkpoint completed: {task.description}")
                system_logs.append(f"Checkpoint {index + 1} completed")
                break

        return final_response

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
            self.active_blueprint = ExecutionBlueprint(
                raw_plan_markdown=blueprint.raw_plan_markdown,
                tasks=list(blueprint.tasks),
                active_task_pointer=len(blueprint.tasks),
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
            fallback = markdown_text.strip() or "Handle the user request."
            tasks.append(CheckpointTask(task_id=1, description=fallback))

        return ExecutionBlueprint(raw_plan_markdown=markdown_text, tasks=tasks)

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
    ) -> str:
        if self._is_scaffold_request(f"{user_prompt} {task_description}".lower()):
            return (
                f"Goal: {user_prompt}\n"
                f"Checkpoint {checkpoint_index}/{total_checkpoints}: {task_description}\n"
                "Complete only this checkpoint. Use the smallest valid tool call and stop after it succeeds."
            )
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
        )
        return any(marker in text for marker in change_markers)

    def _resolve_direct_tool_scope(self, user_prompt: str) -> list[str]:
        text = user_prompt.lower()
        tool_names = set(READ_ONLY_EXECUTION_TOOLS)
        if any(token in text for token in ("memory", "recall", "trace", "history", "earlier", "previous")):
            tool_names.update(MEMORY_EXECUTION_TOOLS)
        return sorted(name for name in tool_names if name in self.tools)

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

        normalized_arguments = self.sandbox.normalize_arguments(tool_call.arguments)
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
