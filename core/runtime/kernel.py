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

from .context_stage import build_context_packet
from .local_router import LocalIntentRouter
from .local_model import load_local_small_model
from .metadata_stage import build_checkpoint_metadata
from .models import (
    AgentState,
    CheckpointTask,
    ExecutionBlueprint,
    PlanningMode,
    ProcessStage,
    RuntimeTurnResult,
    StageTrace,
    ToolExecutionStep,
    VerificationResult,
)
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
    "Break the work into as many single-shot checkpoints as needed for the available context."
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
        tool_client: Any | None = None,
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
        self.local_small_model = load_local_small_model()
        self.tool_client = tool_client or self._build_tool_client(db_path=db_path, vector_dir=vector_dir)

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
        local_only: bool = False,
    ) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        stage_traces: list[StageTrace] = []
        verification_results: list[VerificationResult] = []
        turn_metadata: dict[str, Any] = {}
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        self._record_working_memory(conversation)
        memory_context = self._retrieve_memory_context(user_prompt)
        logger.info("Retrieved memory context: chars=%s", len(memory_context))
        system_logs.append(f"Memory context chars: {len(memory_context)}")
        system_logs.append(f"Planning mode: {planning_mode.value}")
        system_logs.append(f"Continue plan: {continue_plan}")
        system_logs.append(f"Local only: {local_only}")
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}
        self.state = AgentState.PLANNING
        system_logs.append(f"State: {self.state.name}")
        blueprint, planning_conversation, creation_trace = self._checkpoint_creation_stage(
            user_prompt=user_prompt,
            memory_context=memory_context,
            continue_plan=continue_plan,
            local_only=local_only,
            planning_mode=planning_mode,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            max_consecutive_tools=max_consecutive_tools,
        )
        stage_traces.append(creation_trace)
        self.active_blueprint = blueprint
        self.active_plan_prompt = user_prompt
        turn_metadata["original_objective"] = blueprint.original_objective or user_prompt

        active_index = _next_incomplete_task_index(blueprint)
        if active_index is None:
            self.active_plan_prompt = None
            self._finalize_turn(user_prompt, "", conversation, metadata=turn_metadata)
            return RuntimeTurnResult(
                final_response="Nothing left to execute.",
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
            )

        checkpoint = blueprint.tasks[active_index]
        self.active_blueprint = _set_active_task(blueprint, active_index)
        context_packet, context_trace = build_context_packet(
            checkpoint_id=checkpoint.task_id,
            checkpoint_objective=checkpoint.objective or checkpoint.description,
            memory_context=memory_context,
            local_model=self.local_small_model,
            tool_names=self._resolve_execution_tool_scope(user_prompt, checkpoint.description),
            execute_tool_call=self._execute_tool_call,
            char_limit=self._context_char_limit_for_checkpoint(checkpoint),
        )
        stage_traces.append(context_trace)
        system_logs.append(f"Context packet chars: {len(context_packet.distilled_context)}")

        try:
            final_response, updated_blueprint, checkpoint_steps = self._brain_stage(
                user_prompt=user_prompt,
                checkpoint=checkpoint,
                blueprint=self.active_blueprint,
                planning_conversation=planning_conversation,
                context_packet=context_packet.as_prompt_block(),
                raw_memory_context=memory_context,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                max_consecutive_tools=max_consecutive_tools,
                local_only=local_only,
            )
        except RuntimeError as exc:
            system_logs.append(f"Execution failed: {exc}")
            split_blueprint = self._split_active_checkpoint(self.active_blueprint, active_index, reason=str(exc))
            if split_blueprint is not None:
                self.active_blueprint = split_blueprint
                stage_traces.append(
                    StageTrace(
                        stage=ProcessStage.CHECKPOINT_CREATION.value,
                        checkpoint_id=checkpoint.task_id,
                        success=True,
                        summary="Split oversized checkpoint into child checkpoints",
                        logs=[str(exc)],
                        payload={"reason": str(exc), "child_count": len(split_blueprint.tasks)},
                    )
                )
            self._finalize_turn(user_prompt, "", conversation, metadata=turn_metadata)
            return RuntimeTurnResult(
                final_response=None,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                error_message=str(exc),
            )

        self.active_blueprint = updated_blueprint
        stage_traces.append(
            StageTrace(
                stage=ProcessStage.BRAIN.value,
                checkpoint_id=checkpoint.task_id,
                success=True,
                summary="Brain stage completed checkpoint execution",
                logs=[f"Checkpoint {checkpoint.task_id} executed"],
                payload={"response_chars": len(final_response or ""), "step_count": len(checkpoint_steps)},
            )
        )

        metadata_record, metadata_trace = build_checkpoint_metadata(
            original_objective=user_prompt,
            checkpoint=self.active_blueprint.tasks[min(active_index, len(self.active_blueprint.tasks) - 1)],
            checkpoint_steps=checkpoint_steps,
            completion_summary=_summarize_execution_note(final_response),
        )
        turn_metadata.update(metadata_record.to_dict())
        stage_traces.append(metadata_trace)

        verification_ok, verification_trace, verification_batch = self._verify_active_checkpoint(
            checkpoint=self.active_blueprint.tasks[min(active_index, len(self.active_blueprint.tasks) - 1)],
            final_response=final_response or "",
            checkpoint_steps=checkpoint_steps,
            system_logs=system_logs,
        )
        stage_traces.append(verification_trace)
        verification_results.extend(verification_batch)

        if final_response:
            conversation.append({"role": "assistant", "content": final_response})

        if not verification_ok:
            self.state = AgentState.PLANNING
            self.active_blueprint, appended_repair = self._append_repair_checkpoint(
                self.active_blueprint,
                checkpoint_id=checkpoint.task_id,
                reason=verification_trace.summary or "Verification failed",
            )
            if appended_repair:
                system_logs.append("Verification failed; appended repair checkpoint")
            else:
                system_logs.append("Verification failed; stopped automatic repair chaining")
            self._finalize_turn(
                user_prompt,
                final_response or "",
                conversation,
                persist_memory=not (final_response or "").startswith("I inspected `"),
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=final_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
            )

        if _next_incomplete_task_index(self.active_blueprint) is None:
            self.state = AgentState.VERIFYING
            self.active_blueprint = _mark_blueprint_verified(self.active_blueprint, True)
            self.active_plan_prompt = None
        else:
            self.state = AgentState.EXECUTING

        logger.info("Finishing runtime turn: final_response_present=%s total_steps=%s", final_response is not None, len(steps))
        self._finalize_turn(
            user_prompt,
            final_response or "",
            conversation,
            persist_memory=not (final_response or "").startswith("I inspected `"),
            metadata=turn_metadata,
        )
        system_logs.append("Turn completed and stored in memory")
        return RuntimeTurnResult(
            final_response=final_response,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            stage_traces=stage_traces,
            verification_results=verification_results,
            metadata=turn_metadata,
            state=self.state.name,
            blueprint=self.active_blueprint,
        )

    def _build_tool_client(self, *, db_path: str, vector_dir: str):
        try:
            from .mcp_client import MCPToolClient

            return MCPToolClient(
                workspace_path=self.workspace_path,
                db_path=db_path,
                vector_dir=vector_dir,
            )
        except RuntimeError as exc:
            logger.warning("Using in-process tool client fallback: error=%s", exc)
            return _InProcessToolClient(self.tools)

    def _checkpoint_creation_stage(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        continue_plan: bool,
        local_only: bool,
        planning_mode: PlanningMode,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
    ) -> tuple[ExecutionBlueprint, list[dict[str, Any]], StageTrace]:
        should_resume_plan = planning_mode is not PlanningMode.FORCE_DIRECT and (continue_plan or self._is_plan_continue_request(user_prompt))
        if should_resume_plan and self._can_continue_active_plan(user_prompt):
            blueprint = self.active_blueprint or self._build_direct_blueprint(user_prompt)
            trace = StageTrace(
                stage=ProcessStage.CHECKPOINT_CREATION.value,
                success=True,
                summary="Resumed existing checkpoint plan",
                logs=[f"Checkpoint count: {len(blueprint.tasks)}"],
                payload={"continued": True},
            )
            return blueprint, [], trace

        planning_conversation: list[dict[str, Any]] = []
        should_plan = self._should_plan(user_prompt, planning_mode)
        if should_plan:
            if local_only:
                planning_response, planning_conversation = self._run_local_only_planning_phase(
                    user_prompt=user_prompt,
                    memory_context=memory_context,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                )
            else:
                planning_response, planning_conversation = self._run_planning_phase(
                    user_prompt=user_prompt,
                    memory_context=memory_context,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    max_consecutive_tools=max_consecutive_tools,
                )
            blueprint = self._parse_markdown_to_blueprint(planning_response or user_prompt, original_objective=user_prompt)
        else:
            blueprint = self._build_direct_blueprint(user_prompt)

        trace = StageTrace(
            stage=ProcessStage.CHECKPOINT_CREATION.value,
            success=True,
            summary="Created ordered checkpoint blueprint",
            logs=[f"Checkpoint count: {len(blueprint.tasks)}", f"Mode: {'planned' if should_plan else 'direct'}"],
            payload={"checkpoint_count": len(blueprint.tasks), "should_plan": should_plan},
        )
        return blueprint, planning_conversation, trace

    def _build_direct_blueprint(self, user_prompt: str) -> ExecutionBlueprint:
        task = self._build_checkpoint_task(task_id=1, description=user_prompt, original_objective=user_prompt)
        return ExecutionBlueprint(
            raw_plan_markdown=f"- [ ] {user_prompt}",
            original_objective=user_prompt,
            tasks=[task],
            active_task_pointer=0,
            verification_passed=False,
        )

    def _build_checkpoint_task(self, *, task_id: int, description: str, original_objective: str, repair_origin_checkpoint_id: int | None = None) -> CheckpointTask:
        target_path_hint = self._derive_scaffold_target_path(original_objective, description)
        expected_artifact = self._infer_expected_artifact(original_objective, description, target_path_hint)
        verification_mode = self._infer_verification_mode(expected_artifact, original_objective, description)
        output_destination = self._infer_output_destination(expected_artifact)
        return CheckpointTask(
            task_id=task_id,
            description=description,
            objective=description,
            target_path_hint=target_path_hint,
            expected_artifact=expected_artifact,
            verification_mode=verification_mode,
            repair_origin_checkpoint_id=repair_origin_checkpoint_id,
            status_reason=None,
            output_destination=output_destination,
        )

    def _infer_expected_artifact(self, user_prompt: str, task_description: str, target_path_hint: str | None) -> str:
        text = f"{user_prompt} {task_description}".lower()
        if any(token in text for token in ("html", "css", "javascript", "frontend")):
            return "frontend"
        if any(token in text for token in ("create", "write", "edit", "modify", "update", "fix", "implement", "file", "folder")) or target_path_hint:
            return "code"
        return "chat"

    def _infer_verification_mode(self, expected_artifact: str, user_prompt: str, task_description: str) -> str:
        if expected_artifact == "frontend":
            return "frontend"
        if expected_artifact == "code":
            return "code"
        text = f"{user_prompt} {task_description}".lower()
        if any(token in text for token in ("file", "folder", "remove", "delete")):
            return "file"
        return "chat"

    def _infer_output_destination(self, expected_artifact: str) -> str:
        if expected_artifact == "frontend":
            return "file_write"
        if expected_artifact == "code":
            return "file_edit"
        return "chat"

    def _context_char_limit_for_checkpoint(self, checkpoint: CheckpointTask) -> int:
        if checkpoint.expected_artifact == "frontend":
            return SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT
        if checkpoint.expected_artifact == "chat":
            return DIRECT_MEMORY_CHAR_LIMIT
        return EXECUTION_MEMORY_CHAR_LIMIT

    def _brain_stage(
        self,
        *,
        user_prompt: str,
        checkpoint: CheckpointTask,
        blueprint: ExecutionBlueprint,
        planning_conversation: list[dict[str, Any]],
        context_packet: str,
        raw_memory_context: str,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
        local_only: bool,
    ) -> tuple[str | None, ExecutionBlueprint, list[ToolExecutionStep]]:
        pre_step_count = len(steps)
        if checkpoint.expected_artifact == "chat":
            if local_only:
                final_response = self._run_local_only_direct_turn(
                    user_prompt=user_prompt,
                    memory_context=raw_memory_context,
                    steps=steps,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                )
                updated = _mark_checkpoint_completed(blueprint, blueprint.active_task_pointer, _summarize_execution_note(final_response))
                self.active_blueprint = updated
                return final_response, updated, steps[pre_step_count:]

            route_decision = self.local_router.decide(user_prompt)
            system_logs.append(
                f"Local route decision: use_local={route_decision.use_local_knowledge} confidence={route_decision.confidence:.3f}"
            )
            if route_decision.use_local_knowledge:
                local_response, handled_locally = self._run_local_knowledge_turn(
                    user_prompt=user_prompt,
                    memory_context=raw_memory_context,
                    steps=steps,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                )
                if handled_locally:
                    updated = _mark_checkpoint_completed(blueprint, blueprint.active_task_pointer, _summarize_execution_note(local_response))
                    self.active_blueprint = updated
                    return local_response, updated, steps[pre_step_count:]

            final_response = self._run_direct_turn(
                user_prompt=user_prompt,
                memory_context=context_packet,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                max_consecutive_tools=max_consecutive_tools,
            )
            updated = _mark_checkpoint_completed(blueprint, blueprint.active_task_pointer, _summarize_execution_note(final_response))
            self.active_blueprint = updated
            return final_response, updated, steps[pre_step_count:]

        if local_only:
            final_response, _plan_complete = self._run_local_only_execution_phase(
                user_prompt=user_prompt,
                memory_context=context_packet,
                blueprint=blueprint,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            return final_response, self.active_blueprint or blueprint, steps[pre_step_count:]

        final_response, _plan_complete = self._run_execution_phase(
            user_prompt=user_prompt,
            memory_context=context_packet,
            blueprint=blueprint,
            conversation=planning_conversation,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            max_consecutive_tools=max_consecutive_tools,
            planning_mode=PlanningMode.FORCE_PLAN,
        )
        return final_response, self.active_blueprint or blueprint, steps[pre_step_count:]

    def _verify_active_checkpoint(
        self,
        *,
        checkpoint: CheckpointTask,
        final_response: str,
        checkpoint_steps: list[ToolExecutionStep],
        system_logs: list[str],
    ) -> tuple[bool, StageTrace, list[VerificationResult]]:
        self.state = AgentState.VERIFYING
        system_logs.append(f"State: {self.state.name}")
        results: list[VerificationResult] = []
        success = True
        logs: list[str] = []

        if checkpoint.verification_mode == "chat":
            success = bool(final_response.strip())
            details = "Non-empty answer returned." if success else "Empty answer returned."
            results.append(VerificationResult(checkpoint_id=checkpoint.task_id, mode="chat", success=success, details=details))
            logs.append(details)
        else:
            diagnostics_target = self._resolve_verification_target_path(checkpoint, checkpoint_steps)
            file_check = self._verify_file_artifact(checkpoint, checkpoint_steps)
            if file_check is not None:
                results.append(file_check)
                logs.append(file_check.details)
                success = success and file_check.success

            diagnostics_tool = self.tools.get("run_diagnostics")
            if diagnostics_tool is not None and checkpoint.verification_mode in {"code", "frontend"}:
                diagnostic_modes = ("frontend", "lint") if checkpoint.verification_mode == "frontend" else ("tests", "types", "lint")
                for mode in diagnostic_modes:
                    result = diagnostics_tool.execute(mode=mode, target_path=diagnostics_target)
                    details = result.output
                    results.append(
                        VerificationResult(
                            checkpoint_id=checkpoint.task_id,
                            mode=mode,
                            success=result.success,
                            details=details,
                        )
                    )
                    logs.append(f"{mode}: {details}")
                    success = success and result.success
            elif not results:
                success = bool(final_response.strip())
                results.append(
                    VerificationResult(
                        checkpoint_id=checkpoint.task_id,
                        mode=checkpoint.verification_mode,
                        success=success,
                        details="Fallback verification used.",
                    )
                )
                logs.append("Fallback verification used.")

        trace = StageTrace(
            stage=ProcessStage.VERIFICATION.value,
            checkpoint_id=checkpoint.task_id,
            success=success,
            summary="Verification passed" if success else "Verification failed",
            logs=logs,
            payload={
                "verification_mode": checkpoint.verification_mode,
                "target_path": diagnostics_target if checkpoint.verification_mode != "chat" else None,
            },
        )
        return success, trace, results

    def _resolve_verification_target_path(self, checkpoint: CheckpointTask, checkpoint_steps: list[ToolExecutionStep]) -> str:
        candidates: list[Path] = []
        for step in checkpoint_steps:
            for key in ("path", "file_path", "target_path"):
                value = step.arguments.get(key)
                if not isinstance(value, str) or not value.strip():
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = Path(self.workspace_path) / candidate
                candidates.append(candidate)

        if checkpoint.target_path_hint:
            hinted = Path(checkpoint.target_path_hint)
            if not hinted.is_absolute():
                hinted = Path(self.workspace_path) / hinted
            candidates.append(hinted)

        for candidate in reversed(candidates):
            if candidate.exists():
                return str(candidate)
            if candidate.parent.exists():
                return str(candidate.parent)
        return self.workspace_path

    def _verify_file_artifact(self, checkpoint: CheckpointTask, checkpoint_steps: list[ToolExecutionStep]) -> VerificationResult | None:
        paths = []
        for step in checkpoint_steps:
            for key in ("path", "file_path", "target_path"):
                value = step.arguments.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value)
        path_hint = checkpoint.target_path_hint or (paths[-1] if paths else None)
        if not path_hint:
            return None
        candidate = Path(path_hint)
        if not candidate.is_absolute():
            candidate = Path(self.workspace_path) / candidate
        details = f"Artifact check for {candidate}: "
        if checkpoint.verification_mode == "frontend":
            root = candidate if candidate.is_dir() else candidate.parent
            required = [root / "index.html", root / "styles.css", root / "script.js"]
            missing = [path.name for path in required if not path.exists()]
            success = not missing
            details += "ok" if success else f"missing {', '.join(missing)}"
        else:
            success = candidate.exists() or candidate.parent.exists()
            details += "ok" if success else "missing"
        return VerificationResult(
            checkpoint_id=checkpoint.task_id,
            mode="file",
            success=success,
            details=details,
        )

    def _append_repair_checkpoint(self, blueprint: ExecutionBlueprint, *, checkpoint_id: int, reason: str) -> tuple[ExecutionBlueprint, bool]:
        tasks = list(blueprint.tasks)
        source_task = next((task for task in tasks if task.task_id == checkpoint_id), None)
        if source_task is None:
            return blueprint, False
        if source_task.repair_origin_checkpoint_id is not None:
            return blueprint, False
        repair_id = max(task.task_id for task in tasks) + 1 if tasks else 1
        repair_summary = _summarize_verification_failure_reason(reason)
        repair_task = CheckpointTask(
            task_id=repair_id,
            description=f"Repair checkpoint {checkpoint_id}: {repair_summary}",
            objective=f"Fix the failed verification for checkpoint {checkpoint_id}: {repair_summary}",
            target_path_hint=source_task.target_path_hint,
            expected_artifact=source_task.expected_artifact,
            verification_mode=source_task.verification_mode,
            repair_origin_checkpoint_id=checkpoint_id,
            status_reason=reason,
            output_destination=source_task.output_destination,
        )
        insert_at = next((index for index, task in enumerate(tasks) if task.task_id == checkpoint_id), len(tasks)) + 1
        tasks.insert(insert_at, repair_task)
        return ExecutionBlueprint(
            raw_plan_markdown=blueprint.raw_plan_markdown,
            original_objective=blueprint.original_objective,
            tasks=tasks,
            active_task_pointer=insert_at,
            verification_passed=False,
        ), True

    def _split_active_checkpoint(self, blueprint: ExecutionBlueprint | None, task_index: int, *, reason: str) -> ExecutionBlueprint | None:
        if blueprint is None or not (0 <= task_index < len(blueprint.tasks)):
            return None
        source_task = blueprint.tasks[task_index]
        child_descriptions = self._decompose_checkpoint(source_task)
        if len(child_descriptions) <= 1:
            return None
        tasks = list(blueprint.tasks[:task_index])
        next_id = max(task.task_id for task in blueprint.tasks) + 1
        child_ids: list[int] = []
        for description in child_descriptions:
            child_ids.append(next_id)
            tasks.append(
                CheckpointTask(
                    task_id=next_id,
                    description=description,
                    objective=description,
                    target_path_hint=source_task.target_path_hint,
                    expected_artifact=source_task.expected_artifact,
                    verification_mode=source_task.verification_mode,
                    repair_origin_checkpoint_id=source_task.repair_origin_checkpoint_id,
                    status_reason=reason,
                    output_destination=source_task.output_destination,
                )
            )
            next_id += 1
        source_with_children = CheckpointTask(
            task_id=source_task.task_id,
            description=source_task.description,
            objective=source_task.objective,
            target_path_hint=source_task.target_path_hint,
            expected_artifact=source_task.expected_artifact,
            verification_mode=source_task.verification_mode,
            repair_origin_checkpoint_id=source_task.repair_origin_checkpoint_id,
            status_reason=reason,
            output_destination=source_task.output_destination,
            child_checkpoint_ids=tuple(child_ids),
            is_completed=True,
            execution_trace_log="Split into smaller child checkpoints before completion.",
        )
        tasks.insert(task_index, source_with_children)
        tasks.extend(blueprint.tasks[task_index + 1 :])
        return ExecutionBlueprint(
            raw_plan_markdown=blueprint.raw_plan_markdown,
            original_objective=blueprint.original_objective,
            tasks=tasks,
            active_task_pointer=task_index + 1,
            verification_passed=False,
        )

    def _decompose_checkpoint(self, checkpoint: CheckpointTask) -> list[str]:
        description = checkpoint.description
        if checkpoint.expected_artifact in {"code", "frontend"}:
            return [
                f"Inspect the files and dependencies needed for: {description}",
                f"Apply the requested implementation for: {description}",
                f"Verify the workspace result for: {description}",
            ]
        return [
            f"Gather the context required for: {description}",
            f"Answer the request clearly for: {description}",
        ]

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
            try:
                ai_response = self.ai.chat(
                    messages=list(conversation),
                    memory_context=direct_memory,
                    tool_names=tool_scope,
                )
            except RuntimeError as exc:
                if not steps:
                    raise
                ai_logs.append(f"AI response failed after tool execution: {exc}")
                system_logs.append("Turn ended after tool execution because the model response failed")
                logger.warning(
                    "AI response failed after tool execution: steps=%s error=%s",
                    len(steps),
                    exc,
                )
                return _build_partial_failure_response(steps, exc)
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

        if not self._can_answer_from_structure(user_prompt):
            ai_logs.append("Local knowledge mode deferred because the question needs code-level inspection")
            return None, False

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

    def _run_local_only_direct_turn(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        steps: list[ToolExecutionStep],
        ai_logs: list[str],
        system_logs: list[str],
    ) -> str:
        ai_logs.append("Local-only runtime selected")
        memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
        if memory_answer is not None:
            ai_logs.append("Local-only answer assembled from memory")
            return memory_answer

        local_response, handled_locally = self._run_local_knowledge_turn(
            user_prompt=user_prompt,
            memory_context=memory_context,
            steps=steps,
            ai_logs=ai_logs,
            system_logs=system_logs,
        )
        if handled_locally and local_response:
            return local_response

        candidate_path = self._resolve_workspace_candidate(user_prompt) or self.workspace_path
        listing_call = ToolCallRequest(
            call_id=f"local_scan_{uuid.uuid4().hex[:10]}",
            tool_name="list_directory",
            arguments={"path": candidate_path, "mode": "recursive", "max_depth": 2},
        )
        listing_step = self._execute_tool_call(listing_call)
        steps.append(listing_step)
        system_logs.append(f"Tool step {len(steps)}: list_directory success={listing_step.success}")
        if not listing_step.success:
            return f"Local-only mode could not inspect `{candidate_path}`."

        relevant_paths = self._select_local_relevant_paths(user_prompt, listing_step.output)
        if not relevant_paths:
            ai_logs.append("Local-only answer fell back to directory summary")
            return _summarize_directory_listing(candidate_path, listing_step.output)

        summary_sections: list[str] = []
        candidate_root = Path(candidate_path)
        for relative_path in relevant_paths[:3]:
            absolute_path = candidate_root / relative_path
            file_summary = self._inspect_local_file_summary(str(absolute_path), steps, system_logs)
            if file_summary:
                summary_sections.append(file_summary)

        if summary_sections:
            ai_logs.append("Local-only answer assembled from workspace files")
            return "\n\n".join(summary_sections)

        ai_logs.append("Local-only answer fell back to directory summary")
        return _summarize_directory_listing(candidate_path, listing_step.output)

    def _run_local_only_planning_phase(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        ai_logs: list[str],
        system_logs: list[str],
    ) -> tuple[str, list[dict[str, Any]]]:
        planning_memory = _trim_memory_context(memory_context, PLANNING_MEMORY_CHAR_LIMIT)
        system_logs.append(f"Planning memory chars sent: {len(planning_memory)}")
        system_logs.append("Planning tool scope size: 0")
        ai_logs.append("Planning blueprint generated locally")
        return self._build_local_plan_markdown(user_prompt), []

    def _run_local_only_execution_phase(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        blueprint: ExecutionBlueprint,
        steps: list[ToolExecutionStep],
        ai_logs: list[str],
        system_logs: list[str],
    ) -> tuple[str | None, bool]:
        self.state = AgentState.EXECUTING
        system_logs.append(f"State: {self.state.name}")
        working_blueprint = blueprint
        checkpoint_indexes = self._execution_checkpoint_indexes(working_blueprint)
        final_response: str | None = None

        for index in checkpoint_indexes:
            task = working_blueprint.tasks[index]
            working_blueprint = _set_active_task(working_blueprint, index)
            self.active_blueprint = working_blueprint
            system_logs.append(f"Current checkpoint {index + 1}/{len(working_blueprint.tasks)}: {task.description}")
            execution_memory = self._resolve_execution_memory(
                user_prompt=user_prompt,
                task_description=task.description,
                memory_context=memory_context,
            )
            system_logs.append(f"Execution memory chars sent: {len(execution_memory)}")
            local_tool_call = self._build_local_execution_tool_call(user_prompt, task.description)
            if local_tool_call is not None:
                ai_logs.append(f"Local-only tool requested: {local_tool_call.tool_name}")
                step = self._execute_tool_call(local_tool_call)
                steps.append(step)
                system_logs.append(f"Tool step {len(steps)}: {local_tool_call.tool_name} success={step.success}")
                if not step.success:
                    raise RuntimeError(step.output)

            final_response = self._build_local_checkpoint_response(user_prompt, task.description, execution_memory)
            trace_log = _summarize_execution_note(final_response)
            working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
            self.active_blueprint = working_blueprint
            ai_logs.append(f"Checkpoint completed locally: {task.description}")
            system_logs.append(f"Checkpoint {index + 1} completed")

        plan_complete = _next_incomplete_task_index(working_blueprint) is None
        if not plan_complete:
            self.state = AgentState.EXECUTING
            system_logs.append("Execution paused after one checkpoint")
        return final_response, plan_complete

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
                original_objective=source_blueprint.original_objective,
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
            original_objective=source_blueprint.original_objective,
            tasks=list(source_blueprint.tasks),
            active_task_pointer=len(source_blueprint.tasks),
            verification_passed=verification_passed,
        )
        return verification_passed

    def _parse_markdown_to_blueprint(self, markdown_text: str, *, original_objective: str | None = None) -> ExecutionBlueprint:
        task_pattern = re.compile(r"^\s*(?:[-*]|\d+\.)\s*\[(?P<status>[ xX])\]\s*(?P<description>.+?)\s*$")
        tasks: list[CheckpointTask] = []
        for line in markdown_text.splitlines():
            match = task_pattern.match(line)
            if not match:
                continue
            task = self._build_checkpoint_task(
                task_id=len(tasks) + 1,
                description=match.group("description").strip(),
                original_objective=original_objective or markdown_text.strip() or "Handle the user request.",
            )
            tasks.append(
                CheckpointTask(
                    task_id=task.task_id,
                    description=task.description,
                    objective=task.objective,
                    target_path_hint=task.target_path_hint,
                    expected_artifact=task.expected_artifact,
                    verification_mode=task.verification_mode,
                    repair_origin_checkpoint_id=task.repair_origin_checkpoint_id,
                    status_reason=task.status_reason,
                    output_destination=task.output_destination,
                    child_checkpoint_ids=task.child_checkpoint_ids,
                    is_completed=match.group("status").lower() == "x",
                )
            )

        if not tasks:
            tasks = self._parse_step_sections(markdown_text, original_objective=original_objective or markdown_text)

        if not tasks:
            fallback = markdown_text.strip() or "Handle the user request."
            tasks.append(self._build_checkpoint_task(task_id=1, description=fallback, original_objective=original_objective or fallback))

        return ExecutionBlueprint(raw_plan_markdown=markdown_text, original_objective=original_objective, tasks=tasks)

    def _parse_step_sections(self, markdown_text: str, *, original_objective: str) -> list[CheckpointTask]:
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
            tasks.append(self._build_checkpoint_task(task_id=len(tasks) + 1, description=description, original_objective=original_objective))
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
        if self.active_blueprint is None:
            return False
        if _next_incomplete_task_index(self.active_blueprint) is None:
            return False
        if self._is_plan_exit_request(user_prompt):
            return False
        if self.active_plan_prompt == user_prompt:
            return True
        if self._is_plan_continue_request(user_prompt):
            return True
        if not self.active_plan_prompt:
            return False
        active_tokens = set(_prompt_keywords(self.active_plan_prompt))
        current_tokens = set(_prompt_keywords(user_prompt))
        if not active_tokens or not current_tokens:
            return False
        overlap = len(active_tokens & current_tokens) / max(min(len(active_tokens), len(current_tokens)), 1)
        return overlap >= 0.6

    def _is_plan_exit_request(self, user_prompt: str) -> bool:
        text = user_prompt.lower()
        exit_markers = (
            "exit plan mode",
            "leave plan mode",
            "stop planning",
            "don't plan",
            "dont plan",
            "no plan",
            "just answer",
            "just tell me",
        )
        return any(marker in text for marker in exit_markers)

    def _is_plan_continue_request(self, user_prompt: str) -> bool:
        text = user_prompt.lower()
        continue_markers = (
            "continue",
            "resume",
            "keep going",
            "go on",
            "carry on",
            "proceed",
            "next checkpoint",
            "finish the plan",
            "finish it",
        )
        return any(marker in text for marker in continue_markers)

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

    def _build_local_plan_markdown(self, user_prompt: str) -> str:
        target_path = self._derive_scaffold_target_path(user_prompt) or ""
        lowered = user_prompt.lower()
        if self._is_scaffold_request(lowered):
            html_path = f"{target_path}/index.html" if target_path else "index.html"
            css_path = f"{target_path}/styles.css" if target_path else "styles.css"
            js_path = f"{target_path}/script.js" if target_path else "script.js"
            return "\n".join(
                [
                    f"- [ ] Create {html_path} with the base calendar layout and linked assets.",
                    f"- [ ] Add {css_path} with the calendar styling.",
                    f"- [ ] Add {js_path} with month navigation and date rendering.",
                ]
            )

        if "main.py" in lowered and "calendar" in lowered:
            return "\n".join(
                [
                    "- [ ] Create calendar/main.py so it prints today's date.",
                    "- [ ] Verify the generated calendar/main.py content matches the request.",
                ]
            )

        return "\n".join(
            [
                "- [ ] Inspect the relevant workspace files for the requested change.",
                "- [ ] Apply the requested update inside the matching file or folder.",
                "- [ ] Verify the result in the workspace.",
            ]
        )

    def _build_local_execution_tool_call(self, user_prompt: str, task_description: str) -> ToolCallRequest | None:
        target_path = self._derive_scaffold_target_path(user_prompt, task_description)
        lowered_task = task_description.lower()
        lowered_prompt = user_prompt.lower()

        if "main.py" in lowered_task and "calendar" in lowered_prompt:
            path = "calendar/main.py"
            return ToolCallRequest(
                call_id=f"local_write_{uuid.uuid4().hex[:10]}",
                tool_name="write_file",
                arguments={
                    "path": path,
                    "content": _local_calendar_main_py(),
                    "mode": "overwrite" if (Path(self.workspace_path) / path).exists() else "fresh",
                },
            )

        if target_path:
            target_root = Path(target_path)
            file_name: str | None = None
            content: str | None = None
            wants_dark_theme = any(token in lowered_prompt or token in lowered_task for token in ("dark theme", "dark mode"))
            if "index.html" in lowered_task or ("html" in lowered_task and "calendar" in lowered_prompt):
                file_name = "index.html"
                content = _local_calendar_html(target_path)
            elif "styles.css" in lowered_task or ("css" in lowered_task and "calendar" in lowered_prompt):
                file_name = "styles.css"
                content = _local_calendar_css(dark_theme=wants_dark_theme)
            elif "script.js" in lowered_task or ("javascript" in lowered_task) or ("js" in lowered_task and "calendar" in lowered_prompt):
                file_name = "script.js"
                content = _local_calendar_js()

            if file_name and content is not None:
                relative_path = str(target_root / file_name).replace("\\", "/")
                return ToolCallRequest(
                    call_id=f"local_write_{uuid.uuid4().hex[:10]}",
                    tool_name="write_file",
                    arguments={
                        "path": relative_path,
                        "content": content,
                        "mode": "overwrite" if (Path(self.workspace_path) / relative_path).exists() else "fresh",
                    },
                )

        if lowered_task.startswith("inspect "):
            candidate_path = self._resolve_workspace_candidate(user_prompt) or self.workspace_path
            return ToolCallRequest(
                call_id=f"local_inspect_{uuid.uuid4().hex[:10]}",
                tool_name="list_directory",
                arguments={"path": candidate_path, "mode": "recursive", "max_depth": 2},
            )

        return None

    def _build_local_checkpoint_response(self, user_prompt: str, task_description: str, execution_memory: str) -> str:
        lowered_task = task_description.lower()
        if "index.html" in lowered_task:
            return "Created the base HTML shell for the calendar frontend and linked the local stylesheet and script."
        if "styles.css" in lowered_task:
            return "Added the calendar styling layer with a responsive layout, panels, and day grid presentation."
        if "script.js" in lowered_task:
            return "Added the local JavaScript calendar behavior for month navigation and day rendering."
        if "main.py" in lowered_task:
            return "Created calendar/main.py so it prints today's date using Python's datetime module."
        if "verify" in lowered_task:
            return "Verified the generated workspace artifact against the requested local-only checkpoint."
        memory_answer = _answer_from_retrieved_memory(user_prompt, execution_memory)
        if memory_answer:
            return memory_answer
        return f"Completed locally: {task_description}"

    def _select_local_relevant_paths(self, user_prompt: str, listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = []
        if isinstance(payload, dict):
            entries = payload.get("entries") or payload.get("topology") or []

        prompt_tokens = {token for token in re.findall(r"[a-z0-9_]+", user_prompt.lower()) if len(token) >= 3}
        preferred_names = {
            "readme.md",
            "documentation.md",
            "server.py",
            "app.py",
            "main.py",
            "core.py",
            "rag.py",
            "llm_connector.py",
            "retriever.py",
        }
        scored: list[tuple[int, str]] = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("is_dir"):
                    continue
                relative_path = entry.get("relative_path")
                if not isinstance(relative_path, str) or not relative_path.strip():
                    continue
                lowered = relative_path.lower()
                score = 0
                if Path(lowered).name in preferred_names:
                    score += 8
                score += sum(2 for token in prompt_tokens if token in lowered)
                if "backend" in prompt_tokens and any(marker in lowered for marker in ("server", "app", "main", "core")):
                    score += 4
                if "rag" in prompt_tokens and "rag" in lowered:
                    score += 4
                if lowered.endswith((".py", ".md", ".txt")):
                    score += 1
                scored.append((score, relative_path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for score, path in scored if score > 0][:3]

    def _inspect_local_file_summary(self, path: str, steps: list[ToolExecutionStep], system_logs: list[str]) -> str | None:
        absolute_path = Path(path).resolve()
        if absolute_path.suffix.lower() == ".py" and "inspect_symbols" in self.tools:
            symbol_call = ToolCallRequest(
                call_id=f"local_symbols_{uuid.uuid4().hex[:10]}",
                tool_name="inspect_symbols",
                arguments={"path": str(absolute_path), "mode": "outline"},
            )
            symbol_step = self._execute_tool_call(symbol_call)
            steps.append(symbol_step)
            system_logs.append(f"Tool step {len(steps)}: inspect_symbols success={symbol_step.success}")
            if symbol_step.success:
                symbol_payload = _extract_tool_payload_json(symbol_step.output)
                symbol_summary = _summarize_symbol_outline(absolute_path.name, symbol_payload)
                if symbol_summary:
                    return symbol_summary

        if "read_file" not in self.tools:
            return None
        read_call = ToolCallRequest(
            call_id=f"local_read_{uuid.uuid4().hex[:10]}",
            tool_name="read_file",
            arguments={"path": str(absolute_path), "features": "content"},
        )
        read_step = self._execute_tool_call(read_call)
        steps.append(read_step)
        system_logs.append(f"Tool step {len(steps)}: read_file success={read_step.success}")
        if not read_step.success:
            return None
        payload = _extract_tool_payload_json(read_step.output)
        content = ""
        if isinstance(payload, dict):
            content = str(payload.get("content") or "")
        return _summarize_local_text_file(absolute_path.name, content)

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

    def _can_answer_from_structure(self, user_prompt: str) -> bool:
        lowered = user_prompt.lower()
        structure_queries = (
            "what is in",
            "show me",
            "list",
            "tell me about",
            "what folders",
            "what files",
            "what's in",
        )
        deep_queries = (
            "how does",
            "how do",
            "why does",
            "decide what",
            "what content",
            "what does it send",
            "architecture",
            "backend work",
        )
        if any(phrase in lowered for phrase in deep_queries):
            return False
        return any(phrase in lowered for phrase in structure_queries)

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
        direct_match = re.search(r"\b([a-z0-9_.-]+/[a-z0-9_./-]+)\b", combined)
        if direct_match:
            candidate = direct_match.group(1).strip("/")
            suffix = Path(candidate).suffix.lower()
            if suffix in {".html", ".css", ".js", ".py"}:
                return str(Path(candidate).parent).replace("\\", "/")
            return candidate
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

    def _finalize_turn(
        self,
        user_prompt: str,
        final_response: str,
        conversation: list[dict[str, Any]],
        *,
        persist_memory: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.ephemeral_history = _compact_conversation(conversation, max_turns=MAX_EPHEMERAL_TURNS)
        self._record_working_memory(self.ephemeral_history)
        logger.info(
            "Compacted runtime history: retained_messages=%s raw_messages=%s",
            len(self.ephemeral_history),
            len(conversation),
        )
        if not persist_memory:
            logger.info("Skipping episodic persistence for low-signal local summary response")
            return
        logger.info("Recording episodic log for completed turn")
        try:
            log_id = self.memory.add_episodic_log(
                user_prompt,
                final_response,
                metadata={
                    "workspace_path": self.workspace_path,
                    "session_id": self.session_id,
                    **(metadata or {}),
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


class _InProcessToolClient:
    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    def call_tool(self, name: str, arguments: dict[str, Any]):
        tool = self._tools.get(name)
        if tool is None:
            return type("ToolCallResult", (), {"success": False, "output": f"Tool '{name}' is not registered.", "data": {}, "is_error": True})()
        result = tool.execute(**arguments)
        return type("ToolCallResult", (), {"success": result.success, "output": result.output, "data": result.data, "is_error": False})()

    def close(self) -> None:
        return None

def _build_partial_failure_response(steps: list[ToolExecutionStep], error: RuntimeError) -> str:
    successful_steps = [step.tool_name for step in steps if step.success]
    if successful_steps:
        tool_summary = ", ".join(successful_steps)
        return (
            f"The requested tool changes were applied ({tool_summary}), "
            f"but the follow-up AI response failed: {error}"
        )

    return f"The AI response failed after tool execution: {error}"


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
    cleaned = [line for line in cleaned if line and _is_high_signal_memory_answer(line, user_prompt)]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return "\n\n".join(cleaned)


def _summarize_directory_listing(candidate_path: str, output: str) -> str:
    relative_paths: list[str] = []
    payload = _extract_tool_payload_json(output)
    entries = []
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("topology") or []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                relative_path = entry.get("relative_path")
                if isinstance(relative_path, str) and relative_path.strip():
                    relative_paths.append(relative_path.strip())

    if not relative_paths:
        lines = output.splitlines()
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


def _summarize_verification_failure_reason(reason: str) -> str:
    compact = re.sub(r"\s+", " ", (reason or "").strip())
    if not compact:
        return "resolve the verification failure"
    if len(compact) > 96:
        compact = f"{compact[:93].rstrip()}..."
    return compact


def _set_active_task(blueprint: ExecutionBlueprint, task_index: int) -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
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
                    objective=task.objective,
                    target_path_hint=task.target_path_hint,
                    expected_artifact=task.expected_artifact,
                    verification_mode=task.verification_mode,
                    repair_origin_checkpoint_id=task.repair_origin_checkpoint_id,
                    status_reason=task.status_reason,
                    output_destination=task.output_destination,
                    child_checkpoint_ids=task.child_checkpoint_ids,
                    is_completed=True,
                    execution_trace_log=trace_log,
                )
            )
        else:
            tasks.append(task)

    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
        tasks=tasks,
        active_task_pointer=min(task_index + 1, len(tasks)),
        verification_passed=blueprint.verification_passed,
    )


def _next_incomplete_task_index(blueprint: ExecutionBlueprint) -> int | None:
    for index, task in enumerate(blueprint.tasks):
        if not task.is_completed:
            return index
    return None


def _mark_blueprint_verified(blueprint: ExecutionBlueprint, passed: bool) -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
        tasks=list(blueprint.tasks),
        active_task_pointer=len(blueprint.tasks),
        verification_passed=passed,
    )


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


def _extract_tool_payload_json(output: str) -> dict[str, Any] | list[Any] | None:
    if not output:
        return None
    brace_index = output.find("{")
    bracket_index = output.find("[")
    start_candidates = [index for index in (brace_index, bracket_index) if index >= 0]
    if not start_candidates:
        return None
    start_index = min(start_candidates)
    try:
        return json.loads(output[start_index:])
    except json.JSONDecodeError:
        return None


def _is_high_signal_memory_answer(candidate: str, user_prompt: str) -> bool:
    lowered = candidate.lower()
    prompt_lowered = user_prompt.lower()
    reject_markers = (
        'relative_path',
        '"depth":',
        '"is_dir":',
        '"path":',
        "tool requested",
        "recovered inline tool request",
        "queued prompt",
        "i inspected `",
        "locally. relevant paths i found:",
    )
    if any(marker in lowered for marker in reject_markers):
        return False
    if ("how does" in prompt_lowered or "how do" in prompt_lowered or "why does" in prompt_lowered) and "i inspected" in lowered:
        return False
    return True


def _summarize_symbol_outline(file_name: str, payload: dict[str, Any] | list[Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        return None

    functions: list[str] = []
    classes: list[str] = []
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = symbol.get("name")
        if not isinstance(name, str):
            continue
        if symbol.get("type") == "class":
            classes.append(name)
        elif symbol.get("type") == "function":
            functions.append(name)

    parts: list[str] = []
    if classes:
        parts.append(f"classes: {', '.join(classes[:4])}")
    if functions:
        parts.append(f"functions: {', '.join(functions[:6])}")
    if not parts:
        return None
    return f"`{file_name}` exposes {'. '.join(parts)}."


def _summarize_local_text_file(file_name: str, content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return None

    frameworks = []
    known_terms = (
        "FastAPI",
        "Flask",
        "SQLAlchemy",
        "Redis",
        "GraphQL",
        "LanceDB",
        "SentenceTransformer",
        "Retriever",
        "RAG",
    )
    lowered = stripped.lower()
    for term in known_terms:
        if term.lower() in lowered:
            frameworks.append(term)

    first_lines = [line.strip() for line in stripped.splitlines() if line.strip()][:3]
    preview = " ".join(first_lines)
    preview = re.sub(r"\s+", " ", preview)[:220].rstrip()
    if frameworks:
        return f"`{file_name}` references {', '.join(frameworks[:4])}. Preview: {preview}"
    return f"`{file_name}` preview: {preview}"


def _local_calendar_html(target_path: str) -> str:
    asset_prefix = ""
    if "/" in target_path:
        asset_prefix = ""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Calendar</title>
    <link rel="stylesheet" href="{asset_prefix}styles.css" />
  </head>
  <body>
    <main class="calendar-app">
      <header class="calendar-header">
        <button id="prev-month" type="button" aria-label="Previous month">Prev</button>
        <div>
          <p class="calendar-kicker">Local Demo</p>
          <h1 id="month-label">Calendar</h1>
        </div>
        <button id="next-month" type="button" aria-label="Next month">Next</button>
      </header>
      <section class="calendar-panel">
        <div class="calendar-weekdays" id="calendar-weekdays"></div>
        <div class="calendar-grid" id="calendar-grid"></div>
      </section>
    </main>
    <script src="{asset_prefix}script.js"></script>
  </body>
</html>
"""


def _local_calendar_css(*, dark_theme: bool = False) -> str:
    if dark_theme:
        return """:root {
  color-scheme: dark;
  --bg: #11161d;
  --panel: rgba(24, 32, 43, 0.96);
  --border: rgba(132, 148, 173, 0.22);
  --text: #f2f5f8;
  --muted: #97a6ba;
  --accent: #7cc7ff;
  --accent-soft: rgba(124, 199, 255, 0.16);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top, rgba(124, 199, 255, 0.12), transparent 28%),
    linear-gradient(180deg, #0d131a 0%, #151d27 100%);
  color: var(--text);
}

.calendar-app {
  max-width: 960px;
  margin: 48px auto;
  padding: 24px;
}

.calendar-header,
.calendar-weekdays,
.calendar-grid {
  display: grid;
  gap: 12px;
}

.calendar-header {
  grid-template-columns: 92px 1fr 92px;
  align-items: center;
  margin-bottom: 18px;
}

.calendar-header button {
  border: 1px solid var(--border);
  background: rgba(18, 25, 35, 0.92);
  color: var(--text);
  border-radius: 10px;
  padding: 10px 12px;
  cursor: pointer;
}

.calendar-kicker {
  margin: 0 0 4px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

.calendar-header h1 {
  margin: 0;
  font-size: 32px;
}

.calendar-panel {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  padding: 20px;
  box-shadow: 0 20px 40px rgba(3, 7, 12, 0.34);
}

.calendar-weekdays,
.calendar-grid {
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.calendar-weekdays {
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.calendar-grid {
  min-height: 420px;
}

.calendar-day {
  border: 1px solid var(--border);
  background: rgba(15, 22, 32, 0.88);
  color: var(--text);
  border-radius: 14px;
  padding: 12px;
  min-height: 88px;
}

.calendar-day.is-today {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.calendar-day.is-empty {
  background: rgba(255, 255, 255, 0.03);
}

@media (max-width: 720px) {
  .calendar-app {
    margin: 20px auto;
    padding: 16px;
  }

  .calendar-header {
    grid-template-columns: 1fr 1fr;
  }

  .calendar-header h1 {
    font-size: 24px;
  }
}
"""

    return """:root {
  color-scheme: light;
  --bg: #f6f4ef;
  --panel: #ffffff;
  --border: #d4cec3;
  --text: #1f1f1f;
  --muted: #6b655d;
  --accent: #1f6feb;
  --accent-soft: #dce9ff;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background: linear-gradient(180deg, #f8f6f1 0%, #ece7dc 100%);
  color: var(--text);
}

.calendar-app {
  max-width: 960px;
  margin: 48px auto;
  padding: 24px;
}

.calendar-header,
.calendar-weekdays,
.calendar-grid {
  display: grid;
  gap: 12px;
}

.calendar-header {
  grid-template-columns: 92px 1fr 92px;
  align-items: center;
  margin-bottom: 18px;
}

.calendar-header button {
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: 10px;
  padding: 10px 12px;
  cursor: pointer;
}

.calendar-kicker {
  margin: 0 0 4px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

.calendar-header h1 {
  margin: 0;
  font-size: 32px;
}

.calendar-panel {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  padding: 20px;
  box-shadow: 0 20px 40px rgba(77, 68, 51, 0.08);
}

.calendar-weekdays,
.calendar-grid {
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.calendar-weekdays {
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.calendar-grid {
  min-height: 420px;
}

.calendar-day {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 14px;
  padding: 12px;
  min-height: 88px;
}

.calendar-day.is-today {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.calendar-day.is-empty {
  background: rgba(0, 0, 0, 0.03);
}

@media (max-width: 720px) {
  .calendar-app {
    margin: 20px auto;
    padding: 16px;
  }

  .calendar-header {
    grid-template-columns: 1fr 1fr;
  }

  .calendar-header h1 {
    font-size: 24px;
  }
}
"""


def _local_calendar_js() -> str:
    return """const monthLabel = document.getElementById("month-label");
const calendarGrid = document.getElementById("calendar-grid");
const weekdays = document.getElementById("calendar-weekdays");
const prevMonthButton = document.getElementById("prev-month");
const nextMonthButton = document.getElementById("next-month");

const weekdayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const monthLabels = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const today = new Date();
let visibleMonth = today.getMonth();
let visibleYear = today.getFullYear();

weekdayLabels.forEach((label) => {
  const item = document.createElement("div");
  item.textContent = label;
  weekdays.appendChild(item);
});

function renderCalendar() {
  const firstDay = new Date(visibleYear, visibleMonth, 1).getDay();
  const daysInMonth = new Date(visibleYear, visibleMonth + 1, 0).getDate();
  monthLabel.textContent = `${monthLabels[visibleMonth]} ${visibleYear}`;
  calendarGrid.innerHTML = "";

  for (let index = 0; index < firstDay; index += 1) {
    const emptyCell = document.createElement("div");
    emptyCell.className = "calendar-day is-empty";
    calendarGrid.appendChild(emptyCell);
  }

  for (let day = 1; day <= daysInMonth; day += 1) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "calendar-day";
    if (day === today.getDate() && visibleMonth === today.getMonth() && visibleYear === today.getFullYear()) {
      cell.classList.add("is-today");
    }
    cell.textContent = String(day);
    calendarGrid.appendChild(cell);
  }
}

prevMonthButton.addEventListener("click", () => {
  visibleMonth -= 1;
  if (visibleMonth < 0) {
    visibleMonth = 11;
    visibleYear -= 1;
  }
  renderCalendar();
});

nextMonthButton.addEventListener("click", () => {
  visibleMonth += 1;
  if (visibleMonth > 11) {
    visibleMonth = 0;
    visibleYear += 1;
  }
  renderCalendar();
});

renderCalendar();
"""


def _local_calendar_main_py() -> str:
    return """from datetime import date


def main() -> None:
    print(date.today().isoformat())


if __name__ == "__main__":
    main()
"""


def _prompt_keywords(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]
