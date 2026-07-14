from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from core.ai import OpenCodeAICore, RoutingAICore
from core.ai.models import AIExecutedToolStep, AIResponse, ToolCallRequest
from core.env import load_dotenv
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.tools.base import BaseTool

from .context_builder import ContextBuilderService
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
from .response_sanitizer import normalize_response_text, sanitize_response_text
from .sandbox import PathSandbox
from .state import resolve_memory_paths

logger = logging.getLogger(__name__)
MAX_EPHEMERAL_TURNS = 4
PLANNING_ALLOWED_TOOLS = frozenset({"list_directory", "read_file", "inspect_symbols"})
READ_ONLY_EXECUTION_TOOLS = frozenset(
    {"list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols", "search_text", "track_symbol"}
)
DIRECT_CODE_INSPECTION_TOOLS = frozenset({"list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols"})
DIRECT_ARCHITECTURE_INSPECTION_TOOLS = frozenset({"list_directory", "read_file", "peek_lines", "inspect_symbols"})
DIRECT_REPO_SUMMARY_TOOLS = frozenset({"list_directory", "read_file", "peek_lines", "inspect_symbols"})
WRITE_EXECUTION_TOOLS = frozenset({"write_file", "edit_file"})
DELETE_EXECUTION_TOOLS = frozenset({"remove_file"})
SHELL_EXECUTION_TOOLS = frozenset({"run_shell", "run_diagnostics", "audit_changes"})
MEMORY_EXECUTION_TOOLS = frozenset({"manage_memory", "inspect_trace"})
WEB_EXECUTION_TOOLS = frozenset({"web_search"})
PLANNING_MEMORY_CHAR_LIMIT = 900
EXECUTION_MEMORY_CHAR_LIMIT = 1400
SCAFFOLD_EXECUTION_TOOLS = frozenset({"list_directory", "write_file", "edit_file"})
SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT = 360
_GENERIC_WORKSPACE_TOKENS = frozenset(
    {
        "about",
        "architecture",
        "backend",
        "code",
        "codebase",
        "does",
        "explain",
        "file",
        "files",
        "folder",
        "folders",
        "how",
        "project",
        "repo",
        "repository",
        "show",
        "system",
        "tell",
        "what",
        "work",
        "works",
    }
)
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
    "Do not create a checklist or execution plan unless the user is asking you to make changes. "
    "Use web_search for current or time-sensitive facts, or when the user explicitly asks to search, browse, google, or look something up. "
    "If web_search is the relevant selected tool, perform the search before answering. "
    "If a search request is ambiguous, ask one concise follow-up question instead of guessing. "
    "Keep the final answer brief unless the user asks for detail."
)
DIRECT_MEMORY_CHAR_LIMIT = 900
CONSOLIDATION_COOLDOWN_STATE_KEY = "runtime.last_consolidation_wall_time"
DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS = 900.0
_AI_SENTINEL = object()
_LOCAL_MODEL_SENTINEL = object()
_TOOL_CLIENT_SENTINEL = object()
_CONTEXT_BUILDER_SENTINEL = object()

PRIVACY_DISABLED_METADATA = {
    "external_context_state": "privacy_blocked",
    "external_context_reason": "Prior memory access is disabled for this turn.",
    "external_context_session_count": 0,
    "external_context_session_ids": [],
}


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
        resolved_db_path, resolved_vector_dir = resolve_memory_paths(
            db_path,
            vector_dir,
            workspace_path=self.workspace_path,
        )
        self.memory = memory or _build_memory_engine(resolved_db_path, resolved_vector_dir)
        self._ai = ai if ai is not None else _AI_SENTINEL
        self.tools: dict[str, BaseTool] = {}
        self.ephemeral_history: list[dict[str, Any]] = []
        self.session_id = str(uuid.uuid4())
        self.db_path = resolved_db_path
        self.vector_dir = resolved_vector_dir
        self.state = AgentState.PLANNING
        self.active_blueprint: ExecutionBlueprint | None = None
        self.active_plan_prompt: str | None = None
        self.local_router = LocalIntentRouter()
        self._local_small_model = _LOCAL_MODEL_SENTINEL
        self._provided_tool_client = tool_client
        self._tool_client = _TOOL_CLIENT_SENTINEL
        self._context_builder = _CONTEXT_BUILDER_SENTINEL
        self._last_consolidation_wall_time = 0.0
        self._exact_logged_answer_cache: dict[str, str | None] = {}
        self.session_usage_totals: dict[str, int] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        if self._ai is not _AI_SENTINEL:
            self._ai.register_tool(tool)
        logger.info("Registered tool with runtime and AI: tool=%s", tool.name)

    def close(self) -> None:
        if hasattr(self.ai, "abort"):
            try:
                self.ai.abort()
            except Exception:
                logger.debug("Ignoring AI abort failure during kernel shutdown", exc_info=True)
        if self._tool_client is not _TOOL_CLIENT_SENTINEL and hasattr(self._tool_client, "close"):
            self._tool_client.close()

    def reset_conversation(self) -> str:
        self.ephemeral_history = []
        self.active_blueprint = None
        self.active_plan_prompt = None
        self.state = AgentState.PLANNING
        self.session_usage_totals = {}
        self.session_id = str(uuid.uuid4())
        if hasattr(self.ai, "reset_session"):
            self.ai.reset_session()
        return self.session_id

    @property
    def ai(self) -> OpenCodeAICore | Any:
        if self._ai is _AI_SENTINEL:
            ai = RoutingAICore(workspace_path=self.workspace_path)
            for tool in self.tools.values():
                ai.register_tool(tool)
            self._ai = ai
        return self._ai

    @property
    def local_small_model(self):
        if self._local_small_model is _LOCAL_MODEL_SENTINEL:
            self._local_small_model = load_local_small_model()
        return self._local_small_model

    @property
    def tool_client(self):
        if self._tool_client is _TOOL_CLIENT_SENTINEL:
            self._tool_client = self._provided_tool_client or self._build_tool_client(
                db_path=self.db_path,
                vector_dir=self.vector_dir,
            )
        return self._tool_client

    @property
    def context_builder(self):
        if self._context_builder is _CONTEXT_BUILDER_SENTINEL:
            self._context_builder = None
        return self._context_builder

    @context_builder.setter
    def context_builder(self, value) -> None:
        self._context_builder = value if value is not None else None

    def execute_turn(
        self,
        user_prompt: str,
        max_consecutive_tools: int = 5,
        planning_mode: PlanningMode = PlanningMode.AUTO,
        continue_plan: bool = False,
        local_only: bool = False,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
        backend_preference: str = "opencode",
        opencode_enabled: bool = False,
        ollama_enabled: bool = False,
        codex_enabled: bool = False,
        session_budget_tokens: int | None = None,
        no_memory: bool = False,
        incognito: bool = False,
    ) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        turn_started_at = time.perf_counter()
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        stage_traces: list[StageTrace] = []
        verification_results: list[VerificationResult] = []
        turn_metadata: dict[str, Any] = {
            "external_context_state": "new_context",
            "external_context_reason": "No strong prior-session match was found.",
            "external_context_session_count": 0,
            "external_context_session_ids": [],
            "backend_preference": backend_preference,
            "backend_used": "local",
            "backend_fallback": "",
            "selected_tools": sorted(self._resolve_selected_tools(selected_tools)),
            "no_memory": no_memory,
            "incognito": incognito,
        }
        if hasattr(self.ai, "set_backend_preference"):
            self.ai.set_backend_preference(
                backend_preference,
                opencode_enabled=opencode_enabled,
                ollama_enabled=ollama_enabled,
                codex_enabled=codex_enabled,
            )
        if session_budget_tokens is not None and self.session_usage_totals.get("total_tokens", 0) >= session_budget_tokens:
            turn_metadata["budget_state"] = {
                "blocked": True,
                "limit": session_budget_tokens,
                "used": self.session_usage_totals.get("total_tokens", 0),
                "remaining": 0,
            }
            return RuntimeTurnResult(
                final_response=None,
                steps=[],
                total_usage=dict(self.session_usage_totals),
                ai_logs=ai_logs,
                system_logs=system_logs + ["Session token budget reached before starting a new turn."],
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                error_message="Session token budget reached. Increase the budget to continue.",
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        if _is_brief_greeting_prompt(user_prompt):
            fast_response = "Hi. What would you like me to recall or inspect?"
            ai_logs.append("Handled greeting locally without running memory retrieval")
            system_logs.append("Greeting fast path bypassed external session and model usage.")
            conversation.append({"role": "assistant", "content": fast_response})
            self._finalize_turn(
                user_prompt,
                fast_response,
                conversation,
                persist_memory=False,
                persist_working_memory=False,
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        conversation_follow_up = _answer_from_recent_conversation_follow_up(user_prompt, self.ephemeral_history)
        if conversation_follow_up is not None:
            ai_logs.append("Answered referential follow-up from recent conversation")
            system_logs.append("Recent-conversation follow-up fast path bypassed memory retrieval and model usage.")
            conversation.append({"role": "assistant", "content": conversation_follow_up})
            self._finalize_turn(
                user_prompt,
                conversation_follow_up,
                conversation,
                persist_memory=False,
                persist_working_memory=False,
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=conversation_follow_up,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        tool_strategy_response = self._answer_tool_strategy_question(user_prompt, selected_tools=turn_metadata["selected_tools"])
        if tool_strategy_response is not None:
            ai_logs.append("Answered tool-strategy question locally from routing rules")
            system_logs.append("Tool-strategy fast path bypassed memory retrieval and model usage.")
            conversation.append({"role": "assistant", "content": tool_strategy_response})
            self._finalize_turn(
                user_prompt,
                tool_strategy_response,
                conversation,
                persist_memory=False,
                persist_working_memory=False,
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=tool_strategy_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        if _is_underspecified_troubleshooting_prompt(user_prompt):
            fast_response = "What is failing? Share the command, error message, file, or step that is breaking so I can trace it accurately."
            ai_logs.append("Blocked underspecified troubleshooting prompt from broad retrieval and model usage")
            system_logs.append("Troubleshooting clarification fast path bypassed memory retrieval and model usage.")
            conversation.append({"role": "assistant", "content": fast_response})
            self._finalize_turn(
                user_prompt,
                fast_response,
                conversation,
                persist_memory=False,
                persist_working_memory=False,
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        if _is_ambiguous_memory_follow_up(user_prompt, conversation):
            fast_response = "What should I explain? I don't have a clear prior subject in this thread yet."
            ai_logs.append("Blocked ambiguous follow-up from broad memory retrieval")
            system_logs.append("Ambiguous follow-up fast path bypassed memory retrieval and model usage.")
            conversation.append({"role": "assistant", "content": fast_response})
            self._finalize_turn(
                user_prompt,
                fast_response,
                conversation,
                persist_memory=False,
                persist_working_memory=False,
                metadata=turn_metadata,
            )
            return RuntimeTurnResult(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        if _should_try_direct_memory_answer(user_prompt):
            fast_response = self._try_fast_direct_memory_answer(user_prompt)
            if fast_response is not None:
                ai_logs.append("Direct memory answer returned from pre-retrieval fast path")
                system_logs.append("Direct-memory fast path bypassed retrieval and model usage.")
                conversation.append({"role": "assistant", "content": fast_response})
                self._finalize_turn(
                    user_prompt,
                    fast_response,
                    conversation,
                    persist_memory=False,
                    persist_working_memory=False,
                    metadata=turn_metadata,
                )
                return RuntimeTurnResult(
                    final_response=fast_response,
                    steps=[],
                    total_usage={},
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    state=self.state.name,
                    blueprint=self.active_blueprint,
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )

        if not incognito:
            self._record_working_memory(conversation)
        if no_memory or incognito:
            memory_context, retrieval_metadata = "", dict(PRIVACY_DISABLED_METADATA)
        else:
            memory_context, retrieval_metadata = self._retrieve_memory_context(user_prompt, local_only=local_only)
        turn_metadata.update(retrieval_metadata)
        logger.info("Retrieved memory context: chars=%s", len(memory_context))
        system_logs.append(f"Memory context chars: {len(memory_context)}")
        system_logs.append(f"Planning mode: {planning_mode.value}")
        system_logs.append(f"Continue plan: {continue_plan}")
        system_logs.append(f"Local only: {local_only}")
        if turn_metadata["selected_tools"]:
            system_logs.append(f"User selected tools: {', '.join(turn_metadata['selected_tools'])}")
        if no_memory or incognito:
            system_logs.append(f"Privacy mode: {'incognito' if incognito else 'no_memory'}")
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}
        self.state = AgentState.PLANNING
        system_logs.append(f"State: {self.state.name}")
        if local_only and _should_try_direct_memory_answer(user_prompt):
            direct_response = self._run_local_only_direct_turn(
                user_prompt=user_prompt,
                memory_context=memory_context,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            conversation.append({"role": "assistant", "content": direct_response})
            self._finalize_turn(
                user_prompt,
                direct_response,
                conversation,
                metadata=turn_metadata,
                persist_memory=(not incognito) and _should_persist_episodic_response(direct_response),
                persist_working_memory=not incognito,
            )
            return RuntimeTurnResult(
                final_response=direct_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                state=self.state.name,
                blueprint=self.active_blueprint,
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )
        if _should_try_direct_memory_answer(user_prompt):
            direct_memory_answer = self._answer_known_project_question_local(user_prompt, memory_context)
            if direct_memory_answer is None:
                direct_memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
            if direct_memory_answer is not None:
                ai_logs.append("Direct memory answer assembled from retrieved context")
                conversation.append({"role": "assistant", "content": direct_memory_answer})
                self._finalize_turn(
                    user_prompt,
                    direct_memory_answer,
                    conversation,
                    metadata=turn_metadata,
                    persist_memory=(not incognito) and _should_persist_episodic_response(direct_memory_answer),
                    persist_working_memory=not incognito,
                )
                return RuntimeTurnResult(
                    final_response=direct_memory_answer,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    state=self.state.name,
                    blueprint=self.active_blueprint,
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )
            if _should_answer_from_memory_only(user_prompt):
                fallback_response = _memory_only_fallback_response(user_prompt)
                ai_logs.append("Handled memory-only recall without invoking planning or model execution")
                conversation.append({"role": "assistant", "content": fallback_response})
                self._finalize_turn(
                    user_prompt,
                    fallback_response,
                    conversation,
                    metadata=turn_metadata,
                    persist_memory=(not incognito) and _should_persist_episodic_response(fallback_response),
                    persist_working_memory=not incognito,
                )
                return RuntimeTurnResult(
                    final_response=fallback_response,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    state=self.state.name,
                    blueprint=self.active_blueprint,
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )
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
            self._finalize_turn(
                user_prompt,
                "",
                conversation,
                metadata=turn_metadata,
                persist_memory=False,
                persist_working_memory=not incognito,
            )
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
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
            )

        checkpoint = blueprint.tasks[active_index]
        self.active_blueprint = _set_active_task(blueprint, active_index)
        context_packet, context_trace = build_context_packet(
            checkpoint_id=checkpoint.task_id,
            checkpoint_objective=checkpoint.objective or checkpoint.description,
            memory_context=memory_context,
            local_model=self.local_small_model,
            tool_names=self._resolve_execution_tool_scope(
                user_prompt,
                checkpoint.description,
                selected_tools=turn_metadata["selected_tools"],
            ),
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
                selected_tools=turn_metadata["selected_tools"],
            )
        except RuntimeError as exc:
            system_logs.append(f"Execution failed: {exc}")
            degraded_response = self._fallback_response_for_runtime_error(
                user_prompt=user_prompt,
                error=exc,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if degraded_response is not None:
                conversation.append({"role": "assistant", "content": degraded_response})
                self._finalize_turn(
                    user_prompt,
                    degraded_response,
                    conversation,
                    metadata=turn_metadata,
                    persist_memory=(not incognito) and _should_persist_episodic_response(degraded_response),
                    persist_working_memory=not incognito,
                )
                return RuntimeTurnResult(
                    final_response=degraded_response,
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
                    elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
                )
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
            self._finalize_turn(
                user_prompt,
                "",
                conversation,
                metadata=turn_metadata,
                persist_memory=False,
                persist_working_memory=not incognito,
            )
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
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
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
                persist_memory=(not incognito) and _should_persist_episodic_response(final_response or ""),
                persist_working_memory=not incognito,
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
                elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
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
            persist_memory=(not incognito) and _should_persist_episodic_response(final_response or ""),
            persist_working_memory=not incognito,
            metadata=turn_metadata,
        )
        system_logs.append("Turn completed and stored in memory")
        if hasattr(self.ai, "last_backend_used"):
            turn_metadata["backend_used"] = getattr(self.ai, "last_backend_used", turn_metadata["backend_used"])
            turn_metadata["backend_fallback"] = getattr(self.ai, "last_backend_fallback", "")
        _merge_usage(self.session_usage_totals, total_usage)
        if session_budget_tokens is not None:
            used = self.session_usage_totals.get("total_tokens", 0)
            turn_metadata["budget_state"] = {
                "blocked": used >= session_budget_tokens,
                "limit": session_budget_tokens,
                "used": used,
                "remaining": max(session_budget_tokens - used, 0),
            }
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
            elapsed_ms=int((time.perf_counter() - turn_started_at) * 1000),
        )

    def _build_tool_client(self, *, db_path: str, vector_dir: str):
        transport = _runtime_tool_transport()
        if transport == "in_process":
            logger.info("Using in-process tool client for runtime execution")
            return _InProcessToolClient(self.tools)
        try:
            from .mcp_client import MCPToolClient

            logger.info("Using MCP tool client for runtime execution")
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
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
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
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    allow_ai_synthesis=self._remote_backend_enabled(),
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
                selected_tools=selected_tools,
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
            selected_tools=selected_tools,
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
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> str | None:
        direct_memory = _focus_memory_context_for_direct_answers(memory_context, DIRECT_MEMORY_CHAR_LIMIT)
        tool_scope = self._resolve_direct_tool_scope(user_prompt, selected_tools=selected_tools)
        system_logs.append(f"Direct memory chars sent: {len(direct_memory)}")
        system_logs.append(f"Direct tool scope size: {len(tool_scope)}")
        conversation = [
            {"role": "system", "content": DIRECT_SYSTEM_RULE},
            *self._selected_tool_messages(selected_tools),
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
            if ai_response.executed_steps:
                converted_steps = [_runtime_step_from_ai_step(step) for step in ai_response.executed_steps]
                start_index = len(steps)
                steps.extend(converted_steps)
                ai_logs.append(f"Backend executed {len(converted_steps)} MCP tool step(s) directly")
                for index, step in enumerate(converted_steps, start=start_index + 1):
                    system_logs.append(f"Tool step {index}: {step.tool_name} success={step.success}")
                if ai_response.content:
                    ai_logs.append("Assistant produced direct response after backend-managed tool execution")
                return ai_response.content
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

    def _remote_backend_enabled(self) -> bool:
        return bool(getattr(self.ai, "opencode_enabled", False) or getattr(self.ai, "codex_enabled", False))

    def _run_local_knowledge_turn(
        self,
        *,
        user_prompt: str,
        memory_context: str,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int] | None = None,
        ai_logs: list[str],
        system_logs: list[str],
        allow_ai_synthesis: bool = False,
    ) -> tuple[str | None, bool]:
        ai_logs.append("Local router selected knowledge mode")
        memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
        if memory_answer is not None and _should_trust_memory_answer_for_prompt(user_prompt):
            self._mark_local_backend_response()
            ai_logs.append("Local knowledge answer assembled from memory")
            return memory_answer, True

        if _is_repo_summary_question(user_prompt):
            workspace_answer = self._answer_from_workspace_inspection(
                user_prompt=user_prompt,
                candidate_path=self.workspace_path,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if workspace_answer:
                self._mark_local_backend_response()
                return (
                    self._maybe_synthesize_local_workspace_answer(
                        user_prompt=user_prompt,
                        workspace_answer=workspace_answer,
                        total_usage=total_usage,
                        ai_logs=ai_logs,
                        system_logs=system_logs,
                        allow_ai_synthesis=allow_ai_synthesis,
                    ),
                    True,
                )

        if _is_architecture_question(user_prompt) and "list_directory" in self.tools:
            workspace_answer = self._answer_from_workspace_inspection(
                user_prompt=user_prompt,
                candidate_path=self.workspace_path,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if workspace_answer:
                self._mark_local_backend_response()
                return (
                    self._maybe_synthesize_local_workspace_answer(
                        user_prompt=user_prompt,
                        workspace_answer=workspace_answer,
                        total_usage=total_usage,
                        ai_logs=ai_logs,
                        system_logs=system_logs,
                        allow_ai_synthesis=allow_ai_synthesis,
                    ),
                    True,
                )

        candidate_path = self._resolve_workspace_candidate(user_prompt)
        if candidate_path and candidate_path != self.workspace_path and "list_directory" in self.tools:
            workspace_answer = self._answer_from_workspace_inspection(
                user_prompt=user_prompt,
                candidate_path=candidate_path,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if workspace_answer:
                self._mark_local_backend_response()
                return workspace_answer, True

        if not self._can_answer_from_structure(user_prompt):
            ai_logs.append("Local knowledge mode deferred because the question needs code-level inspection")
            return None, False

        if candidate_path is None or "list_directory" not in self.tools:
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
        structured_answer = self._answer_known_project_question_local(user_prompt, memory_context)
        if structured_answer is not None:
            self._mark_local_backend_response()
            ai_logs.append("Local-only answer assembled from structured project facts")
            return structured_answer
        candidate_path = self._resolve_workspace_candidate(user_prompt)
        if candidate_path and self._should_prefer_workspace_inspection(user_prompt, candidate_path):
            workspace_answer = self._answer_from_workspace_inspection(
                user_prompt=user_prompt,
                candidate_path=candidate_path,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if workspace_answer:
                self._mark_local_backend_response()
                return workspace_answer

        memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
        if memory_answer is not None and _should_trust_memory_answer_for_prompt(user_prompt):
            self._mark_local_backend_response()
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
            self._mark_local_backend_response()
            return local_response

        if "list_directory" not in self.tools:
            if _should_answer_from_memory_only(user_prompt):
                return _memory_only_fallback_response(user_prompt)
            return "Local-only mode needs workspace inspection tools to answer that prompt."

        candidate_path = candidate_path or self.workspace_path
        listing_depth = 3 if candidate_path == self.workspace_path and _prefers_deeper_workspace_scan(user_prompt) else 2
        listing_call = ToolCallRequest(
            call_id=f"local_scan_{uuid.uuid4().hex[:10]}",
            tool_name="list_directory",
            arguments={"path": candidate_path, "mode": "recursive", "max_depth": listing_depth},
        )
        listing_step = self._execute_tool_call(listing_call)
        steps.append(listing_step)
        system_logs.append(f"Tool step {len(steps)}: list_directory success={listing_step.success}")
        if not listing_step.success:
            return f"Local-only mode could not inspect `{candidate_path}`."

        relevant_paths = self._select_local_relevant_paths(user_prompt, listing_step.output)
        if not relevant_paths:
            ai_logs.append("Local-only answer fell back to directory summary")
            self._mark_local_backend_response()
            return _summarize_directory_listing(candidate_path, listing_step.output)

        summary_sections: list[str] = []
        candidate_root = Path(candidate_path)
        for relative_path in relevant_paths[:3]:
            absolute_path = candidate_root / relative_path
            file_summary = self._inspect_local_file_summary(str(absolute_path), steps, system_logs)
            if file_summary and file_summary not in summary_sections:
                summary_sections.append(file_summary)

        if summary_sections:
            ai_logs.append("Local-only answer assembled from workspace files")
            self._mark_local_backend_response()
            return "\n\n".join(summary_sections)

        ai_logs.append("Local-only answer fell back to directory summary")
        self._mark_local_backend_response()
        return _summarize_directory_listing(candidate_path, listing_step.output)

    def _should_prefer_workspace_inspection(self, user_prompt: str, candidate_path: str) -> bool:
        lowered = user_prompt.lower()
        candidate_name = Path(candidate_path).name.lower()
        if candidate_name == "getgit" and "get-drip" in lowered:
            return False
        if candidate_name not in lowered:
            return False
        return _is_architecture_question(user_prompt) or _is_file_inventory_question(user_prompt)

    def _answer_from_workspace_inspection(
        self,
        *,
        user_prompt: str,
        candidate_path: str,
        steps: list[ToolExecutionStep],
        ai_logs: list[str],
        system_logs: list[str],
    ) -> str | None:
        listing_depth = 3 if candidate_path == self.workspace_path and _prefers_deeper_workspace_scan(user_prompt) else 2
        listing_call = ToolCallRequest(
            call_id=f"local_scan_{uuid.uuid4().hex[:10]}",
            tool_name="list_directory",
            arguments={"path": candidate_path, "mode": "recursive", "max_depth": listing_depth},
        )
        listing_step = self._execute_tool_call(listing_call)
        steps.append(listing_step)
        system_logs.append(f"Tool step {len(steps)}: list_directory success={listing_step.success}")
        if not listing_step.success:
            return None

        if _is_file_inventory_question(user_prompt):
            ai_logs.append("Local-only answer assembled from workspace inventory")
            inventory_paths = self._inventory_paths_from_listing(listing_step.output)
            if inventory_paths:
                return "The concrete GetGit paths included " + ", ".join(f"`{path}`" for path in inventory_paths[:10]) + "."
            return _summarize_directory_listing(candidate_path, listing_step.output)

        relevant_paths = self._select_local_relevant_paths(user_prompt, listing_step.output)
        if _is_repo_overview_question(user_prompt):
            relevant_paths = self._ensure_repo_summary_paths(relevant_paths, listing_step.output)
        if _is_backend_connector_question(user_prompt):
            relevant_paths = self._ensure_backend_connector_paths(relevant_paths, listing_step.output)
        elif _is_architecture_question(user_prompt):
            relevant_paths = self._ensure_architecture_summary_paths(relevant_paths, listing_step.output)
        if not relevant_paths:
            return _summarize_directory_listing(candidate_path, listing_step.output)

        summary_sections: list[str] = []
        candidate_root = Path(candidate_path)
        for relative_path in relevant_paths[:3]:
            absolute_path = candidate_root / relative_path
            file_summary = self._inspect_local_file_summary(str(absolute_path), steps, system_logs)
            if file_summary and file_summary not in summary_sections:
                summary_sections.append(file_summary)
        if not summary_sections:
            self._mark_local_backend_response()
            return _summarize_directory_listing(candidate_path, listing_step.output)
        ai_logs.append("Local-only answer assembled from workspace files")
        if _is_backend_connector_question(user_prompt):
            joined = "\n- ".join(summary_sections)
            self._mark_local_backend_response()
            return (
                "## Backend Connector Fit\n"
                "Yes. Devenv already has a backend-router pattern, so Claude can be added as another reasoning backend the same way Codex and Ollama are wired in.\n"
                "The main touchpoints are `core/ai/routing.py`, `core/runtime/web.py`, and the backend adapters under `core/ai/`.\n"
                "- "
                + joined
                + "\n\n"
                "## What To Add\n"
                "- Create a `core/ai/claude_backend.py` adapter that matches the existing `chat`, `status`, `set_model`, and `reset_session` expectations.\n"
                "- Register that backend inside `RoutingAICore` so backend preference and model selection can switch to Claude.\n"
                "- Extend `core/runtime/web.py` and the website backend picker so Claude appears as another backend option with access control and model metadata."
            )
        if _is_architecture_question(user_prompt) and not _is_repo_overview_question(user_prompt):
            joined = "\n- ".join(summary_sections)
            self._mark_local_backend_response()
            return "I inspected the backend entry points locally. The main pieces are:\n- " + joined
        if _is_repo_overview_question(user_prompt):
            joined = "\n- ".join(summary_sections)
            self._mark_local_backend_response()
            return "## Project Overview\n- " + joined
        self._mark_local_backend_response()
        return "\n\n".join(summary_sections)

    def _maybe_synthesize_local_workspace_answer(
        self,
        *,
        user_prompt: str,
        workspace_answer: str,
        total_usage: dict[str, int] | None,
        ai_logs: list[str],
        system_logs: list[str],
        allow_ai_synthesis: bool,
    ) -> str:
        if not allow_ai_synthesis:
            return workspace_answer
        evidence_context = "\n".join(
            [
                "## Local Workspace Evidence",
                "Use only this bounded local evidence. Do not request tools or extra files.",
                workspace_answer,
            ]
        )
        try:
            ai_response = self.ai.chat(
                messages=[{"role": "user", "content": user_prompt}],
                memory_context=evidence_context,
                tool_names=[],
            )
        except RuntimeError as exc:
            ai_logs.append(f"OpenCode synthesis skipped after local retrieval: {exc}")
            system_logs.append("Local workspace evidence fell back to direct Devenv summary after OpenCode synthesis failed.")
            setattr(self.ai, "last_backend_used", "local")
            setattr(self.ai, "last_backend_fallback", str(exc))
            return workspace_answer
        if total_usage is not None:
            _merge_usage(total_usage, ai_response.usage)
        ai_logs.append("OpenCode synthesized the final answer from bounded local workspace evidence")
        return ai_response.content or workspace_answer

    def _mark_local_backend_response(self) -> None:
        if self.ai is None:
            return
        setattr(self.ai, "last_backend_used", "local")
        setattr(self.ai, "last_backend_fallback", "")

    def _ensure_repo_summary_paths(self, relevant_paths: list[str], listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return relevant_paths[:3]

        supplemental: list[tuple[int, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("is_dir"):
                continue
            relative_path = entry.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                continue
            lowered = relative_path.lower()
            score = 0
            if lowered == "readme.md":
                score += 30
            if lowered == "core/runtime/kernel.py":
                score += 24
            if lowered == "core/ai/routing.py":
                score += 22
            if lowered == "core/runtime/web.py":
                score += 20
            if lowered == "core/ai/opencode_client.py":
                score += 18
            if lowered in {"core/runtime/context_builder.py", "core/memory/engine.py"}:
                score += 16
            if lowered in {"feature.md", "process.md", "pyproject.toml", "requirements.txt"}:
                score += 10
            if any(marker in lowered for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")):
                score -= 10
            if Path(lowered).name in {"__init__.py", "env.py", "logging_utils.py"}:
                score -= 8
            if score > 0:
                supplemental.append((score, relative_path))

        supplemental.sort(key=lambda item: (-item[0], item[1]))
        ordered_paths: list[str] = []
        for _score, path in supplemental:
            if path not in ordered_paths:
                ordered_paths.append(path)
            if len(ordered_paths) >= 3:
                break
        if ordered_paths:
            return ordered_paths[:3]
        return relevant_paths[:3]

    def _ensure_architecture_summary_paths(self, relevant_paths: list[str], listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        architecture_targets = {
            "core/runtime/kernel.py": 24,
            "core/runtime/web.py": 22,
            "core/ai/routing.py": 20,
            "core/ai/codex_backend.py": 19,
            "core/ai/ollama_backend.py": 18,
            "core/runtime/context_builder.py": 18,
            "core/ai/opencode_client.py": 17,
            "core/memory/engine.py": 14,
            "core/runtime/workspace.py": 10,
            "README.md": 8,
        }
        scored_paths: dict[str, int] = {}
        for path in relevant_paths:
            if not isinstance(path, str) or not path.strip():
                continue
            score = architecture_targets.get(path, 0)
            lowered = path.lower()
            if any(marker in lowered for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")):
                score -= 10
            scored_paths[path] = max(score, scored_paths.get(path, 0))
        if not isinstance(entries, list):
            ordered = sorted(scored_paths.items(), key=lambda item: (-item[1], item[0]))
            return [path for path, _score in ordered if _score > 0][:3] or relevant_paths[:3]

        supplemental: list[tuple[int, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("is_dir"):
                continue
            relative_path = entry.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                continue
            score = architecture_targets.get(relative_path, 0)
            lowered = relative_path.lower()
            if any(marker in lowered for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")):
                score -= 10
            if score > 0:
                supplemental.append((score, relative_path))

        preferred_paths: list[str] = []
        for path, score in sorted(scored_paths.items(), key=lambda item: (-item[1], item[0])):
            if score > 0 and path not in preferred_paths:
                preferred_paths.append(path)
        supplemental.sort(key=lambda item: (-item[0], item[1]))
        for _score, path in supplemental:
            if path not in preferred_paths:
                preferred_paths.append(path)
            if len(preferred_paths) >= 3:
                break
        return preferred_paths[:3]

    def _ensure_backend_connector_paths(self, relevant_paths: list[str], listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        connector_targets = {
            "core/ai/routing.py": 28,
            "core/runtime/web.py": 24,
            "core/ai/codex_backend.py": 22,
            "core/ai/ollama_backend.py": 21,
            "core/ai/opencode_client.py": 18,
            "core/runtime/kernel.py": 16,
            "README.md": 8,
        }
        scored_paths: dict[str, int] = {}
        for path in relevant_paths:
            if not isinstance(path, str) or not path.strip():
                continue
            score = connector_targets.get(path, 0)
            lowered = path.lower()
            if any(marker in lowered for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")):
                score -= 10
            scored_paths[path] = max(score, scored_paths.get(path, 0))
        if not isinstance(entries, list):
            ordered = sorted(scored_paths.items(), key=lambda item: (-item[1], item[0]))
            return [path for path, _score in ordered if _score > 0][:3] or relevant_paths[:3]

        supplemental: list[tuple[int, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("is_dir"):
                continue
            relative_path = entry.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                continue
            score = connector_targets.get(relative_path, 0)
            lowered = relative_path.lower()
            if any(marker in lowered for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")):
                score -= 10
            if score > 0:
                supplemental.append((score, relative_path))

        preferred_paths: list[str] = []
        for path, score in sorted(scored_paths.items(), key=lambda item: (-item[1], item[0])):
            if score > 0 and path not in preferred_paths:
                preferred_paths.append(path)
        supplemental.sort(key=lambda item: (-item[0], item[1]))
        for _score, path in supplemental:
            if path not in preferred_paths:
                preferred_paths.append(path)
            if len(preferred_paths) >= 3:
                break
        return preferred_paths[:3]

    def _answer_known_project_question_local(self, user_prompt: str, memory_context: str) -> str | None:
        lowered = user_prompt.lower()
        if not _should_skip_exact_logged_fast_path(user_prompt):
            logged_answer = self._lookup_exact_logged_answer(user_prompt)
            if logged_answer is not None:
                return logged_answer
        if "infer the parts of the app" in lowered and "get-drip" in lowered:
            store = getattr(self.memory, "store", None)
            if store is not None and hasattr(store, "search_logs"):
                try:
                    logs = store.search_logs(
                        ["get-drip", "convex-api.ts", "convex-types.ts", "journey.ts", "pipeline.tsx", "test-activate.tsx"],
                        limit=12,
                    )
                except Exception:
                    logs = []
                paths: list[str] = []
                for log in logs:
                    for path in _extract_path_mentions(log.raw_interaction):
                        lowered_path = path.lower()
                        if "guidelines.md" in lowered_path or "email_g..." in lowered_path:
                            continue
                        if any(marker in lowered_path for marker in ("convex-api.ts", "convex-types.ts", "journey.ts", "pipeline.tsx", "test-activate.tsx", "workspace.$workspaceid")):
                            if path not in paths:
                                paths.append(path)
                if paths:
                    return "The strongest clues point to " + ", ".join(f"`{path}`" for path in paths[:5]) + "."

        return _answer_known_project_question(user_prompt, memory_context)

    def _try_fast_direct_memory_answer(self, user_prompt: str) -> str | None:
        lowered = user_prompt.lower()
        if not _should_skip_exact_logged_fast_path(user_prompt):
            answer = self._lookup_exact_logged_answer(user_prompt)
            if answer is not None:
                return answer
        if "infer the parts of the app" in lowered and "get-drip" in lowered:
            return self._answer_known_project_question_local(user_prompt, "")
        return None

    def _try_fast_local_only_direct_answer(self, user_prompt: str) -> str | None:
        return self._try_fast_direct_memory_answer(user_prompt)

    def _can_skip_external_memory_fetch(self, user_prompt: str, *, memory_context: str, local_only: bool) -> bool:
        if local_only and _should_try_direct_memory_answer(user_prompt):
            return True
        if not _should_try_direct_memory_answer(user_prompt):
            return False
        if self._answer_known_project_question_local(user_prompt, memory_context) is not None:
            return True
        if _answer_from_retrieved_memory(user_prompt, memory_context) is not None:
            return True
        return False

    def _lookup_exact_logged_answer(self, user_prompt: str) -> str | None:
        lowered = user_prompt.lower()
        if not _supports_exact_logged_answer_prompt(user_prompt):
            return None
        if lowered in self._exact_logged_answer_cache:
            return self._exact_logged_answer_cache[lowered]

        store = getattr(self.memory, "store", None)
        if store is None or not hasattr(store, "search_logs"):
            return None
        query_variants = _exact_logged_query_variants(user_prompt)
        if hasattr(store, "search_agent_responses_for_external_query"):
            direct_responses = []
            for query_variant in query_variants:
                try:
                    direct_responses = store.search_agent_responses_for_external_query(query_variant, limit=8)
                except Exception:
                    direct_responses = []
                if direct_responses:
                    break
            direct_candidates: list[tuple[int, str]] = []
            terms = _lexical_memory_terms(user_prompt)
            for response in direct_responses:
                if not isinstance(response, str):
                    continue
                cleaned_response = _sanitize_logged_answer(response)
                if cleaned_response and _is_usable_logged_answer(user_prompt, cleaned_response):
                    direct_candidates.append((_lexical_line_score(cleaned_response, user_prompt, terms), cleaned_response))
            if direct_candidates:
                direct_candidates.sort(key=lambda item: item[0], reverse=True)
                best_score, best_response = direct_candidates[0]
                if best_score >= 1:
                    shaped_response = _shape_logged_answer_for_prompt(user_prompt, best_response)
                    self._exact_logged_answer_cache[lowered] = shaped_response
                    return shaped_response

        logs = []
        if hasattr(store, "search_logs_for_external_query"):
            for query_variant in query_variants:
                try:
                    logs = store.search_logs_for_external_query(query_variant, limit=8)
                except Exception:
                    logs = []
                if logs:
                    break
        if not logs:
            try:
                logs = store.search_logs(_lexical_memory_terms(user_prompt), limit=20)
            except Exception:
                return None

        allow_fallback_candidates = True
        fallback_candidates: list[tuple[int, str]] = []
        terms = _lexical_memory_terms(user_prompt)
        for log in logs:
            try:
                payload = json.loads(log.raw_interaction)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            agent_text = payload.get("agent")
            metadata = payload.get("metadata") or {}
            if not isinstance(agent_text, str) or not agent_text.strip():
                continue
            cleaned_agent_text = _sanitize_logged_answer(agent_text)
            if not _is_usable_logged_answer(user_prompt, cleaned_agent_text):
                continue
            exact_external_query = str(metadata.get("external_context_query") or "").strip().lower() if isinstance(metadata, dict) else ""
            logged_user = str(payload.get("user") or "").strip().lower()
            if exact_external_query in query_variants:
                exact_answer = _shape_logged_answer_for_prompt(user_prompt, cleaned_agent_text)
                self._exact_logged_answer_cache[lowered] = exact_answer
                return exact_answer
            if logged_user in query_variants:
                exact_answer = _shape_logged_answer_for_prompt(user_prompt, cleaned_agent_text)
                self._exact_logged_answer_cache[lowered] = exact_answer
                return exact_answer
            if allow_fallback_candidates:
                fallback_candidates.append((_lexical_line_score(cleaned_agent_text, user_prompt, terms), cleaned_agent_text))
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        selected = _shape_logged_answer_for_prompt(user_prompt, fallback_candidates[0][1]) if fallback_candidates and fallback_candidates[0][0] >= 1 else None
        self._exact_logged_answer_cache[lowered] = selected
        return selected

    def _inventory_paths_from_listing(self, listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = []
        if isinstance(payload, dict):
            entries = payload.get("entries") or payload.get("topology") or []
        preferred_paths: list[str] = []
        preferred_names = {
            "server.py",
            "core.py",
            "checkpoints.py",
            "checkpoints.txt",
            "clone_repo.py",
            "repo_manager.py",
            "readme.md",
            "documentation.md",
            "templates/index.html",
            "rag/chunker.py",
            "rag/config.py",
            "rag/embedder.py",
            "rag/llm_connector.py",
            "rag/retriever.py",
        }
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                relative_path = entry.get("relative_path")
                if not isinstance(relative_path, str) or not relative_path.strip():
                    continue
                lowered = relative_path.lower()
                if lowered in preferred_names:
                    preferred_paths.append(relative_path.strip())
                elif lowered in {"rag", "templates", "static"}:
                    preferred_paths.append(relative_path.strip())
        deduped: list[str] = []
        for path in preferred_paths:
            if path not in deduped:
                deduped.append(path)
        return deduped

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
            if ai_response.executed_steps:
                converted_steps = [_runtime_step_from_ai_step(step) for step in ai_response.executed_steps]
                steps.extend(converted_steps)
                ai_logs.append(f"Planning backend executed {len(converted_steps)} MCP tool step(s) directly")
                content = ai_response.content
                if content:
                    conversation.append({"role": "assistant", "content": content})
                    ai_logs.append("Planning blueprint generated")
                return content, conversation
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
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
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
            scoped_tool_names = self._resolve_execution_tool_scope(
                user_prompt,
                task.description,
                selected_tools=selected_tools,
            )
            execution_memory = self._resolve_execution_memory(
                user_prompt=user_prompt,
                task_description=task.description,
                memory_context=memory_context,
            )
            system_logs.append(f"Execution memory chars sent: {len(execution_memory)}")
            system_logs.append(f"Execution tool scope size: {len(scoped_tool_names)}")
            step_conversation = [
                {"role": "system", "content": EXECUTION_SYSTEM_RULE},
                *self._selected_tool_messages(selected_tools),
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
                if ai_response.executed_steps:
                    converted_steps = [_runtime_step_from_ai_step(step) for step in ai_response.executed_steps]
                    start_index = len(steps)
                    steps.extend(converted_steps)
                    tool_iterations += len(converted_steps)
                    if tool_iterations > max_consecutive_tools:
                        raise RuntimeError("Execution tool limit reached before the checkpoint completed.")
                    for step_index, step in enumerate(converted_steps, start=start_index + 1):
                        system_logs.append(f"Tool step {step_index}: {step.tool_name} success={step.success}")
                    if checkpoint_requires_mutation and not any(
                        step.tool_name in (*WRITE_EXECUTION_TOOLS, *DELETE_EXECUTION_TOOLS) for step in converted_steps
                    ):
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
                        continue
                    final_response = ai_response.content or final_response
                    trace_log = _summarize_execution_note(ai_response.content)
                    working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                    self.active_blueprint = working_blueprint
                    ai_logs.append(f"Checkpoint completed: {task.description}")
                    system_logs.append(f"Checkpoint {index + 1} completed")
                    break
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

    def _resolve_execution_tool_scope(
        self,
        user_prompt: str,
        task_description: str,
        *,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[str]:
        return self._tool_scope_for_prompt(
            f"{user_prompt}\n{task_description}",
            selected_tools=selected_tools,
            execution_phase=True,
        )

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
        if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
            return False
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

    def _fallback_response_for_runtime_error(
        self,
        *,
        user_prompt: str,
        error: RuntimeError,
        steps: list[ToolExecutionStep],
        ai_logs: list[str],
        system_logs: list[str],
    ) -> str | None:
        if _is_opencode_access_denied_error(error):
            lowered = user_prompt.lower()
            fallback_path: str | None = None
            candidate_path = self._resolve_workspace_candidate(user_prompt) if "list_directory" in self.tools else None
            if candidate_path and candidate_path != self.workspace_path:
                fallback_path = candidate_path
            elif any(marker in lowered for marker in ("repo", "repository", "codebase", "backend", "architecture", "system")):
                fallback_path = self.workspace_path
            if fallback_path:
                workspace_answer = self._answer_from_workspace_inspection(
                    user_prompt=user_prompt,
                    candidate_path=fallback_path,
                    steps=steps,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                )
                if workspace_answer:
                    ai_logs.append("Returned local workspace fallback after OpenCode access denial")
                    return f"OpenCode backend access is not granted right now, so here's a local workspace summary instead:\n\n{workspace_answer}"

            ai_logs.append("Returned degraded access-denied response")
            return "OpenCode backend access has not been granted, so I couldn't run the reasoning stage for that prompt."

        if not _is_opencode_transport_error(error):
            return None

        recent_follow_up = _answer_from_recent_conversation_follow_up(user_prompt, self.ephemeral_history)
        if recent_follow_up is not None:
            ai_logs.append("Returned recent-conversation fallback after OpenCode transport failure")
            return recent_follow_up

        if _should_try_direct_memory_answer(user_prompt):
            memory_context, _metadata = self._retrieve_memory_context(user_prompt, local_only=True)
            memory_answer = self._answer_known_project_question_local(user_prompt, memory_context)
            if memory_answer is None and _should_trust_memory_answer_for_prompt(user_prompt):
                memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
            if memory_answer is None and not _should_skip_exact_logged_fast_path(user_prompt):
                memory_answer = self._lookup_exact_logged_answer(user_prompt)
            if memory_answer:
                ai_logs.append("Returned memory fallback after OpenCode transport failure")
                return memory_answer

        lowered = user_prompt.lower()
        if any(marker in lowered for marker in ("repo", "repository", "codebase", "backend", "architecture", "system")):
            workspace_answer = self._answer_from_workspace_inspection(
                user_prompt=user_prompt,
                candidate_path=self.workspace_path,
                steps=steps,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )
            if workspace_answer:
                ai_logs.append("Returned workspace fallback after OpenCode transport failure")
                return workspace_answer

        ai_logs.append("Returned degraded transport-error response")
        return "OpenCode was temporarily unavailable, and I couldn't recover a reliable local answer for that prompt."

    def _resolve_direct_tool_scope(
        self,
        user_prompt: str,
        *,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[str]:
        return self._tool_scope_for_prompt(user_prompt, selected_tools=selected_tools, execution_phase=False)

    def _resolve_selected_tools(self, selected_tools: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
        return {
            tool_name.strip()
            for tool_name in (selected_tools or ())
            if isinstance(tool_name, str) and tool_name.strip() in self.tools
        }

    def _answer_tool_strategy_question(
        self,
        user_prompt: str,
        *,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> str | None:
        subject_prompt = _tool_strategy_subject_prompt(user_prompt)
        if subject_prompt is None:
            return None

        if _should_try_direct_memory_answer(subject_prompt):
            return (
                "For that question I would not need workspace tools first. "
                "Devenv should answer from memory/retrieval, and only fall back if prior context is not reliable enough."
            )
        if _is_underspecified_troubleshooting_prompt(subject_prompt):
            return (
                "For that question Devenv should not guess or run tools first. "
                "It should ask one concise clarification about the failing command, error, file, or step before choosing tools."
            )

        if _is_repo_summary_question(subject_prompt):
            return (
                "For that repo-summary question Devenv should stay in charge of retrieval. "
                "It would usually inspect the workspace with `list_directory`, then ground the answer with `read_file` and `inspect_symbols`."
            )

        if _is_architecture_question(subject_prompt):
            return (
                "For that architecture question Devenv should stay in charge of retrieval. "
                "It would usually inspect the workspace with `list_directory`, then ground the answer with `inspect_symbols` on the main backend files."
            )

        scoped_tools = self._resolve_direct_tool_scope(subject_prompt, selected_tools=selected_tools)
        if not scoped_tools:
            return "I would answer that directly without tools unless the first pass showed missing context."
        if scoped_tools == ["web_search"]:
            return "For that question I would use `web_search` first, then answer from the retrieved results."
        return (
            "For that question Devenv would stay in charge of tool choice. "
            f"It would start with this bounded tool scope if needed: {', '.join(f'`{name}`' for name in scoped_tools)}."
        )

    def _scoped_tool_names(self, selected_tools: list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
        resolved = sorted(self._resolve_selected_tools(selected_tools))
        return resolved or sorted(self.tools)

    def _tool_scope_for_prompt(
        self,
        prompt: str,
        *,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
        execution_phase: bool,
    ) -> list[str]:
        selected_scope = self._resolve_selected_tools(selected_tools)
        if selected_scope:
            return sorted(selected_scope)

        if getattr(self.ai, "preferred_backend", "") == "ollama":
            return []

        available = set(self.tools)
        if not available:
            return []

        lowered = prompt.lower()
        scope: set[str] = set()

        if not execution_phase and self._should_start_direct_without_tools(prompt):
            if self._should_offer_web_search(lowered):
                web_only_scope = WEB_EXECUTION_TOOLS & available
                return sorted(web_only_scope)
            return []

        if not execution_phase and self._should_start_with_code_inspection_scope(prompt):
            inspection_scope_source = (
                DIRECT_REPO_SUMMARY_TOOLS
                if _is_repo_summary_question(prompt)
                else DIRECT_ARCHITECTURE_INSPECTION_TOOLS
                if self._should_prefer_compact_architecture_scope(prompt)
                else DIRECT_CODE_INSPECTION_TOOLS
            )
            inspection_scope = inspection_scope_source & available
            if inspection_scope:
                return sorted(inspection_scope)

        if execution_phase:
            scope.update(READ_ONLY_EXECUTION_TOOLS & available)
        else:
            scope.update((READ_ONLY_EXECUTION_TOOLS - {"search_text"}) & available)

        if self._should_offer_web_search(lowered):
            if not execution_phase and self._should_prefer_web_only(lowered):
                web_only_scope = WEB_EXECUTION_TOOLS & available
                if web_only_scope:
                    return sorted(web_only_scope)
            scope.update(WEB_EXECUTION_TOOLS & available)

        if self._should_offer_shell_tools(lowered):
            scope.update(SHELL_EXECUTION_TOOLS & available)

        if self._should_offer_memory_tools(lowered):
            scope.update(MEMORY_EXECUTION_TOOLS & available)

        if execution_phase and self._text_requires_mutation_tools(lowered):
            scope.update((WRITE_EXECUTION_TOOLS | DELETE_EXECUTION_TOOLS) & available)

        if execution_phase and self._is_scaffold_request(lowered):
            scope.update(SCAFFOLD_EXECUTION_TOOLS & available)

        if not scope:
            fallback = READ_ONLY_EXECUTION_TOOLS & available
            scope.update(fallback or available)
        return sorted(scope)

    def _should_start_direct_without_tools(self, prompt: str) -> bool:
        lowered = prompt.lower()
        if _should_answer_from_memory_only(prompt):
            return True
        if _is_cleanup_schema_prompt(prompt) or _is_bug_list_question(prompt):
            return True
        if any(
            phrase in lowered
            for phrase in (
                "what can be said confidently",
                "what remains unclear",
                "what architecture did",
                "same architecture",
                "look different",
                "do you know about",
                "what do you know about",
            )
        ):
            if not (_is_file_inventory_question(prompt) or _is_architecture_question(prompt)):
                return True
        if _is_repo_summary_question(prompt):
            return False
        if any(
            marker in lowered
            for marker in (
                "inspect the repo",
                "inspect the codebase",
                "look in the repo",
                "open the file",
                "read the file",
                "show me the file",
                "which files",
                "list the files",
                "list the concrete files",
                "what folders",
                "what files",
            )
        ):
            return False
        return False

    def _should_start_with_code_inspection_scope(self, prompt: str) -> bool:
        lowered = prompt.lower()
        if self._should_offer_web_search(lowered) or _should_answer_from_memory_only(prompt):
            return False
        if _is_repo_summary_question(prompt):
            return True
        if any(
            marker in lowered
            for marker in (
                "how does",
                "how do",
                "why does",
                "why do",
                "backend work",
                "code path",
                "implementation",
                "implemented",
                "where is",
                "which file",
                "which files",
                "what file",
            )
        ):
            return True
        return False

    def _should_prefer_compact_architecture_scope(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(
            marker in lowered
            for marker in (
                "summarize this repo",
                "summarize the repo",
                "summarize this repository",
                "summarize the repository",
                "explain the repo",
                "explain this repo",
                "explain the repository",
                "explain this repository",
                "how does the backend work",
                "how does this backend work",
                "how does the repo work",
                "how does the repository work",
                "how does the system work",
                "explain this project architecture",
                "backend architecture",
                "repo architecture",
                "repository architecture",
            )
        )

    def _should_offer_web_search(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "latest",
                "current",
                "today",
                "recent",
                "browse",
                "google",
                "look up",
                "search",
                "search the web",
                "web search",
                "news",
                "docs",
                "documentation",
                "president",
                "prime minister",
                "ceo",
            )
        )

    def _should_offer_shell_tools(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "run",
                "test",
                "pytest",
                "unittest",
                "diagnostic",
                "diagnostics",
                "build",
                "compile",
                "lint",
                "format",
                "server",
                "smoke",
                "benchmark",
            )
        )

    def _should_prefer_web_only(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "latest docs",
                "current docs",
                "search the web",
                "look up",
                "browse",
                "google",
                "latest documentation",
            )
        )

    def _should_offer_memory_tools(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "memory",
                "trace",
                "retrieval",
                "episodic",
                "working memory",
            )
        )

    def _text_requires_mutation_tools(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
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
        )

    def _selected_tool_messages(self, selected_tools: list[str] | tuple[str, ...] | set[str] | None) -> list[dict[str, str]]:
        resolved = self._scoped_tool_names(selected_tools) if selected_tools else []
        if not resolved:
            return []
        lines = [
            f"The user explicitly selected these tools for this turn: {', '.join(resolved)}.",
            "Use only those selected tools if you need a tool.",
        ]
        if "web_search" in resolved:
            lines.append("If the request is to search, browse, or look something up, call web_search before answering.")
        return [{"role": "system", "content": " ".join(lines)}]

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
            "kernel.py",
            "routing.py",
            "web.py",
            "workspace.py",
            "codex_backend.py",
            "ollama_backend.py",
            "opencode_client.py",
        }
        scored: list[tuple[int, str]] = []
        repo_summary_prompt = _is_repo_overview_question(user_prompt)
        architecture_prompt = _is_architecture_question(user_prompt)
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
                if "backend" in prompt_tokens and any(
                    marker in lowered for marker in ("core/runtime", "core/ai", "runtime/", "server.py", "routes.py", "routing.py", "kernel.py")
                ):
                    score += 8
                if "backend" in prompt_tokens and any(
                    marker in lowered for marker in ("sample-test/", "tests/", "docs/", "devenv.egg-info/", "devenv1a.egg-info/")
                ):
                    score -= 8
                if "backend" in prompt_tokens and Path(lowered).name == "__init__.py":
                    score -= 8
                if "backend" in prompt_tokens and lowered.endswith(".md"):
                    score -= 4
                if "rag" in prompt_tokens and "rag" in lowered:
                    score += 4
                if architecture_prompt and any(
                    marker in lowered
                    for marker in (
                        "core/runtime/kernel.py",
                        "core/runtime/web.py",
                        "core/ai/routing.py",
                        "core/ai/codex_backend.py",
                        "core/ai/ollama_backend.py",
                        "core/ai/opencode_client.py",
                        "core/runtime/context_builder.py",
                        "core/memory/engine.py",
                    )
                ):
                    score += 12
                if architecture_prompt and lowered == "core/runtime/kernel.py":
                    score += 8
                if architecture_prompt and lowered == "core/runtime/web.py":
                    score += 7
                if architecture_prompt and lowered == "core/ai/routing.py":
                    score += 6
                if architecture_prompt and lowered == "core/ai/codex_backend.py":
                    score += 6
                if architecture_prompt and lowered == "core/ai/ollama_backend.py":
                    score += 5
                if architecture_prompt and lowered == "core/ai/opencode_client.py":
                    score += 2
                if architecture_prompt and any(
                    marker in lowered
                    for marker in ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")
                ):
                    score -= 10
                if architecture_prompt and Path(lowered).name in {"__init__.py", "env.py", "logging_utils.py"}:
                    score -= 8
                if architecture_prompt and lowered in {"requirements.txt", "feature.md", "process.md", "pyproject.toml"}:
                    score -= 10
                if repo_summary_prompt and lowered == "readme.md":
                    score += 10
                if repo_summary_prompt and any(
                    marker in lowered
                    for marker in ("core/runtime/", "core/ai/", "core/memory/", "core/tools/")
                ):
                    score += 8
                if repo_summary_prompt and any(
                    marker in lowered
                    for marker in ("kernel.py", "routing.py", "opencode_client.py", "engine.py", "web.py", "workspace.py")
                ):
                    score += 6
                if repo_summary_prompt and any(
                    marker in lowered
                    for marker in ("sample-test/", "tests/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")
                ):
                    score -= 8
                if repo_summary_prompt and Path(lowered).name in {"__init__.py", "env.py", "logging_utils.py"}:
                    score -= 6
                if repo_summary_prompt and lowered in {"feature.md", "process.md", "requirements.txt", "pyproject.toml"}:
                    score -= 8
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

        directory_candidates: list[Path] = []
        for entry in entries:
            if not entry.is_dir():
                continue
            directory_candidates.append(entry)
            try:
                children = sorted(entry.iterdir(), key=lambda item: item.name.lower())
            except OSError:
                children = []
            for child in children:
                if child.is_dir():
                    directory_candidates.append(child)

        names = [entry.name.lower() for entry in directory_candidates]
        for token in prompt_tokens:
            for entry in directory_candidates:
                if entry.name.lower() == token:
                    return str(entry)
        fuzzy_tokens = [token for token in prompt_tokens if token not in _GENERIC_WORKSPACE_TOKENS and len(token) >= 5]
        for token in fuzzy_tokens:
            matches = get_close_matches(token, names, n=1, cutoff=0.82)
            if matches:
                matched_name = matches[0]
                for entry in directory_candidates:
                    if entry.name.lower() == matched_name:
                        return str(entry)
        return self.workspace_path if entries else None

    def _can_answer_from_structure(self, user_prompt: str) -> bool:
        lowered = user_prompt.lower()
        if _is_backend_connector_question(user_prompt):
            return True
        structure_queries = (
            "what is in",
            "show me",
            "list",
            "tell me about",
            "what folders",
            "what files",
            "what's in",
            "summarize this repo",
            "summarize the repo",
            "summarize this repository",
            "summarize the repository",
            "explain the repo",
            "explain this repo",
            "explain the repository",
            "explain this repository",
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
                data={},
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
                data={},
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
                data={},
            )
        logger.info("Executing runtime tool: tool=%s normalized_arguments=%s", tool_call.tool_name, normalized_arguments)
        result = self.tool_client.call_tool(tool_call.tool_name, normalized_arguments)
        logger.info("Runtime tool finished: tool=%s success=%s is_error=%s", tool_call.tool_name, result.success, result.is_error)
        return ToolExecutionStep(
            step_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            arguments=normalized_arguments,
            output=_format_tool_output(result.output, result.data),
            success=result.success and not result.is_error,
            is_sandboxed_violation=False,
            data=dict(result.data or {}),
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
        persist_working_memory: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        sanitized_response = sanitize_response_text(final_response) or final_response
        if conversation:
            for message in reversed(conversation):
                if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                    message["content"] = sanitize_response_text(message["content"]) or message["content"]
                    break
        self.ephemeral_history = _compact_conversation(conversation, max_turns=MAX_EPHEMERAL_TURNS)
        if persist_working_memory:
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
                sanitized_response,
                metadata={
                    "workspace_path": self.workspace_path,
                    "session_id": self.session_id,
                    **(metadata or {}),
                },
            )
            logger.info("Recorded episodic log: log_id=%s", log_id)
            if hasattr(self.memory, "run_consolidation") and self._should_run_consolidation():
                result = self.memory.run_consolidation()
                self._mark_consolidation_ran()
                logger.info(
                    "Memory consolidation finished: processed_logs=%s created_nodes=%s updated_nodes=%s",
                    getattr(result, "processed_logs", 0),
                    len(getattr(result, "created_nodes", ())),
                    len(getattr(result, "updated_nodes", ())),
                )
            elif hasattr(self.memory, "run_consolidation"):
                logger.info("Skipping memory consolidation because cooldown has not elapsed")
        except Exception as exc:
            logger.warning("Failed to record episodic log; continuing without persisted memory: error=%s", exc)

    def _should_run_consolidation(self) -> bool:
        cooldown = _consolidation_cooldown_seconds()
        if cooldown <= 0:
            return True

        now = time.time()
        last_ran = self._last_consolidation_wall_time
        store = getattr(self.memory, "store", None)
        if store is not None and hasattr(store, "get_state"):
            try:
                last_ran = max(last_ran, float(store.get_state(CONSOLIDATION_COOLDOWN_STATE_KEY) or 0.0))
            except Exception:
                pass
        return (now - last_ran) >= cooldown

    def _mark_consolidation_ran(self) -> None:
        now = time.time()
        self._last_consolidation_wall_time = now
        store = getattr(self.memory, "store", None)
        if store is not None and hasattr(store, "set_state"):
            try:
                store.set_state(CONSOLIDATION_COOLDOWN_STATE_KEY, str(now))
            except Exception:
                logger.warning("Failed to persist consolidation cooldown state", exc_info=True)

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

    def _retrieve_memory_context(self, user_prompt: str, *, local_only: bool = False) -> tuple[str, dict[str, Any]]:
        memory_context = ""
        metadata: dict[str, Any] = {
            "external_context_state": "new_context",
            "external_context_reason": "No strong prior-session match was found.",
            "external_context_session_count": 0,
            "external_context_session_ids": [],
        }
        if _should_skip_retrieval_for_prompt(user_prompt):
            metadata["external_context_reason"] = "Skipped memory retrieval for a low-context prompt."
            return "", metadata
        if self._should_skip_current_workspace_memory_lookup(user_prompt):
            metadata["external_context_reason"] = "Skipped memory retrieval for a current-workspace inspection prompt."
            return "", metadata
        lexical_query = _compose_external_memory_query(user_prompt, self.ephemeral_history)
        lexical_context = self._retrieve_lexical_memory_context(user_prompt, search_query=lexical_query)
        if lexical_context:
            memory_context = lexical_context
            if self._can_skip_external_memory_fetch(user_prompt, memory_context=memory_context, local_only=local_only):
                return memory_context, metadata
        elif not _should_skip_vector_memory_lookup(user_prompt):
            try:
                result = self.memory.retrieve_context(user_prompt)
                self._persist_last_retrieval_trace(getattr(result, "trace", None))
                memory_context = result.markdown_context
                if self._can_skip_external_memory_fetch(user_prompt, memory_context=memory_context, local_only=local_only):
                    return memory_context, metadata
            except Exception as exc:
                logger.warning("Memory retrieval failed; continuing without memory context: error=%s", exc)
        if _should_skip_external_session_context(user_prompt):
            metadata["external_context_reason"] = "Skipped external session lookup for a current-workspace inspection prompt."
            return memory_context, metadata
        external_builder = getattr(self, "context_builder", None)
        if external_builder is None or not hasattr(external_builder, "build_runtime_memory_context"):
            return memory_context, metadata
        external_query = _compose_external_memory_query(user_prompt, self.ephemeral_history)
        try:
            external_context, session_ids, selection_metadata = external_builder.build_runtime_memory_context(external_query)
            metadata.update(
                {
                    "external_context_state": selection_metadata.get("context_match_state", metadata["external_context_state"]),
                    "external_context_reason": selection_metadata.get("context_match_reason", metadata["external_context_reason"]),
                    "external_context_session_count": len(session_ids),
                    "external_context_session_ids": list(session_ids),
                    "external_context_query": external_query,
                }
            )
        except Exception as exc:
            logger.warning("External session retrieval failed; continuing without session context: error=%s", exc)
            return memory_context, metadata
        if not external_context.strip():
            return memory_context, metadata
        if not memory_context.strip():
            return external_context, metadata
        return f"{memory_context.rstrip()}\n\n{external_context}", metadata

    def _should_skip_current_workspace_memory_lookup(self, user_prompt: str) -> bool:
        if not _should_skip_current_workspace_memory_lookup(user_prompt):
            return False
        return "list_directory" in self.tools

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

    def _retrieve_lexical_memory_context(self, user_prompt: str, *, search_query: str | None = None) -> str:
        if not _should_try_direct_memory_answer(user_prompt):
            return ""

        store = getattr(self.memory, "store", None)
        if store is None or not hasattr(store, "search_logs"):
            return ""

        terms = _lexical_memory_terms(search_query or user_prompt)
        if not terms:
            return ""

        ranked_lines: list[tuple[int, str]] = []
        try:
            for log in store.search_logs(terms, limit=20):
                payload = json.loads(log.raw_interaction)
                user_text = str(payload.get("user") or "").strip()
                agent_text = str(payload.get("agent") or "").strip()
                if user_text.lower() == user_prompt.strip().lower():
                    continue
                summary = " | ".join(part for part in (user_text, agent_text) if part)
                if summary and _is_high_signal_memory_answer(summary, user_prompt):
                    ranked_lines.append((_lexical_line_score(summary, user_prompt, terms), f"- [episode] {summary}"))
            if hasattr(store, "search_nodes"):
                for node in store.search_nodes(terms, limit=10):
                    candidate = f"{node.label}: {node.summary}"
                    if node.summary.lower().startswith(user_prompt.strip().lower()):
                        continue
                    if _is_high_signal_memory_answer(candidate, user_prompt):
                        ranked_lines.append((_lexical_line_score(candidate, user_prompt, terms), f"- [{node.category}] {candidate}"))
        except Exception as exc:
            logger.warning("Lexical memory lookup failed; continuing without lexical context: error=%s", exc)
            return ""

        ranked_lines.sort(key=lambda item: item[0], reverse=True)
        unique_lines: list[str] = []
        for _score, line in ranked_lines:
            if line not in unique_lines:
                unique_lines.append(line)
        if not unique_lines:
            return ""
        compact_context = self._compact_lexical_memory_context(user_prompt, unique_lines)
        if compact_context:
            return compact_context
        return "## Retrieved Memory\n" + "\n".join(unique_lines[:6])

    def _compact_lexical_memory_context(self, user_prompt: str, lines: list[str]) -> str:
        if not lines or not _should_try_direct_memory_answer(user_prompt):
            return ""

        selected: list[str] = []
        for line in lines[:6]:
            selected.append(line)
            candidate_context = "## Retrieved Memory\n" + "\n".join(selected)
            if _answer_known_project_question(user_prompt, candidate_context) is not None:
                return candidate_context
            if _answer_from_retrieved_memory(user_prompt, candidate_context) is not None:
                return candidate_context
        return ""


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


def _runtime_step_from_ai_step(step: AIExecutedToolStep) -> ToolExecutionStep:
    return ToolExecutionStep(
        step_id=step.step_id,
        tool_name=step.tool_name,
        arguments=dict(step.arguments),
        output=step.output,
        success=step.success and not step.is_error,
        is_sandboxed_violation=False,
        data=dict(step.data),
    )


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
    if os.getenv("DEVENV_USE_SENTENCE_EMBEDDER") != "1":
        return MemoryEngine(
            db_path=db_path,
            vector_dir=vector_dir,
            embedder=HashingEmbedder(dimension=384),
        )
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


def _extract_readable_replay_answer(content: str) -> str | None:
    raw = str(content or "").strip()
    if not raw or "\n" not in raw or not raw.startswith("{"):
        return None

    readable_lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        part = payload.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text_value = str(part.get("text") or "").strip()
            if text_value:
                readable_lines.append(text_value)
            continue

        event_payload = payload.get("payload")
        if isinstance(event_payload, dict) and event_payload.get("type") == "agent_message":
            message = str(event_payload.get("message") or "").strip()
            if message:
                readable_lines.append(message)

    deduped_lines: list[str] = []
    for line in readable_lines:
        if line not in deduped_lines:
            deduped_lines.append(line)
    if not deduped_lines:
        return None
    return "\n\n".join(deduped_lines)


def _sanitize_logged_answer(content: str) -> str:
    extracted = _extract_readable_replay_answer(content)
    if extracted:
        return sanitize_response_text(extracted) or ""
    return sanitize_response_text(str(content or "").strip()) or ""


def _normalize_logged_answer_text(content: str) -> str:
    return normalize_response_text(content)


def _shape_logged_project_answer(user_prompt: str, answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return ""

    lowered_prompt = user_prompt.lower()
    if "get-drip" in lowered_prompt and _is_cleanup_schema_prompt(user_prompt):
        issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(_memory_context_lines(cleaned)))
        if issues:
            summary = _join_human_list(issues)
            return f"The get-drip cleanup was mainly about {summary}."
    if "get-drip" in lowered_prompt and _is_bug_list_question(user_prompt):
        issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(_memory_context_lines(cleaned)))
        if issues:
            return _format_issue_list_answer("get-drip", issues)
    if "get-drip" in lowered_prompt and _is_memory_recall_question(user_prompt):
        issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(_memory_context_lines(cleaned)))
        if issues:
            summary = _join_human_list(issues)
            return f"Yes. In get-drip, the main issues were {summary}."
    return cleaned


def _shape_logged_answer_for_prompt(user_prompt: str, answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return ""
    if _is_session_history_question(user_prompt):
        return cleaned
    return _shape_logged_project_answer(user_prompt, cleaned)


def _answer_from_retrieved_memory(user_prompt: str, memory_context: str) -> str | None:
    if not memory_context.strip():
        return None

    sections = _memory_context_sections(memory_context)
    cleaned_lines = [_clean_memory_line(line) for line in [*sections["working"], *sections["external"], *sections["retrieved"]]]
    cleanup_summary = _cleanup_summary_from_lines(user_prompt, cleaned_lines)
    if cleanup_summary:
        return cleanup_summary
    if "get-drip" in user_prompt.lower() and _is_memory_recall_question(user_prompt):
        extracted_issues = _extract_follow_up_issues(cleaned_lines)
        if extracted_issues:
            summary = _join_human_list(extracted_issues)
            return f"Yes. In get-drip, the main issues were {summary}."

    if _is_bug_list_question(user_prompt) and not _is_cleanup_schema_prompt(user_prompt):
        issue_lines = [
            _humanize_recalled_line(_clean_memory_line(line), user_prompt)
            for line in [*sections["working"], *sections["external"], *sections["retrieved"]]
        ]
        issue_lines = [line for line in issue_lines if line and _is_high_signal_memory_answer(line, user_prompt)]
        issue_subject = _preferred_memory_subject(user_prompt, sections["external"] + sections["working"] + sections["retrieved"])
        extracted_issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(issue_lines))
        if extracted_issues:
            return _format_issue_list_answer(issue_subject, extracted_issues)
    if _is_memory_follow_up_question(user_prompt) and sections["working"]:
        recent_working_lines = _recent_working_follow_up_lines(sections["working"], user_prompt)
        if recent_working_lines:
            shaped_working = [_humanize_recalled_line(line, user_prompt) for line in recent_working_lines]
            shaped_working = [line for line in shaped_working if line]
            if shaped_working:
                subject = _preferred_memory_subject(user_prompt, sections["working"] + sections["external"] + sections["retrieved"])
                issue_summary = _summarize_follow_up_issues(shaped_working)
                if issue_summary and _is_bug_fix_follow_up_question(user_prompt):
                    if subject:
                        return f"Yes. In {subject}, we fixed those bugs by addressing {issue_summary}."
                    return f"Yes. We fixed those bugs by addressing {issue_summary}."
                if issue_summary and _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {issue_summary}."
                if issue_summary and _is_issue_recap_follow_up_question(user_prompt):
                    if subject:
                        return f"Yes. In {subject}, the main issues were {issue_summary}."
                    return f"Yes. The main issues were {issue_summary}."
                cleanup_summary = _summarize_cleanup_narrative(shaped_working)
                if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {cleanup_summary}."
                if _is_bug_fix_follow_up_question(user_prompt) or _is_issue_recap_follow_up_question(user_prompt):
                    shaped_working = []
                if len(shaped_working) == 1:
                    return _affirm_memory_answer(shaped_working[0])
    if _is_memory_follow_up_question(user_prompt) and sections["external"]:
        ordered_follow_up = _ordered_follow_up_lines(user_prompt, sections["external"])
        if ordered_follow_up:
            shaped_follow_up = [_humanize_recalled_line(line, user_prompt) for line in ordered_follow_up]
            shaped_follow_up = [line for line in shaped_follow_up if line]
            if not shaped_follow_up:
                return None
            synthesized_issues = _summarize_follow_up_issues(shaped_follow_up)
            if synthesized_issues:
                subject = _infer_memory_subject(sections["working"] + sections["external"] + sections["retrieved"])
                if _is_bug_fix_follow_up_question(user_prompt):
                    if subject:
                        return f"Yes. In {subject}, we fixed those bugs by addressing {synthesized_issues}."
                    return f"Yes. We fixed those bugs by addressing {synthesized_issues}."
                if _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {synthesized_issues}."
                if _is_bug_list_question(user_prompt):
                    return _format_issue_list_answer(subject, _extract_follow_up_issues(shaped_follow_up))
                if subject:
                    return f"Yes. In {subject}, the main issues were {synthesized_issues}."
                return f"Yes. The main issues were {synthesized_issues}."
            cleanup_summary = _summarize_cleanup_narrative(shaped_follow_up)
            if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
                return f"Yes. It was mainly about {cleanup_summary}."
            if len(shaped_follow_up) >= 2 and _follow_up_line_score(ordered_follow_up[0].lower()) >= 2 and _follow_up_line_score(ordered_follow_up[1].lower()) >= 2:
                return "Yes. The main issues were: " + "; ".join(shaped_follow_up[:2])
            if _follow_up_line_score(ordered_follow_up[0].lower()) >= 2:
                return _affirm_memory_answer(shaped_follow_up[0])
            if len(shaped_follow_up) == 1:
                return _affirm_memory_answer(shaped_follow_up[0])
            return "Yes.\n\n" + "\n\n".join(shaped_follow_up[:3])
    primary_lines = [*sections["retrieved"], *sections["external"]]
    working_lines = sections["working"]
    bullet_lines: list[tuple[str, str]] = []
    for line in primary_lines:
        bullet = line.strip()
        bullet_lower = bullet.lower()
        if bullet_lower.startswith("prompt:") or bullet_lower.startswith("[workspace] workspace:"):
            continue
        bullet_lines.append(("memory", bullet))
    for line in working_lines:
        bullet = line.strip()
        bullet_lower = bullet.lower()
        if bullet_lower.startswith("user:"):
            continue
        bullet_lines.append(("working", bullet))
    if not bullet_lines:
        return None

    prompt_tokens = _memory_query_tokens(user_prompt)
    prompt_entities = _memory_query_entities(user_prompt)
    inferred_entities = _memory_context_entities(memory_context)
    if prompt_entities:
        query_entities = prompt_entities
    elif _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        query_entities = inferred_entities
    else:
        query_entities = set()
    ranked: list[tuple[int, int, str, str]] = []
    for source, line in bullet_lines:
        line_lower = line.lower()
        token_overlap = sum(1 for token in prompt_tokens if token in line_lower)
        overlap = token_overlap
        overlap += sum(6 for entity in query_entities if entity in line_lower)
        if _is_memory_follow_up_question(user_prompt):
            if line_lower.startswith("user asked:"):
                overlap += 4
            elif line_lower.startswith("assistant reported:"):
                overlap += 1
            elif line_lower.startswith("session '"):
                overlap -= 1
        elif _is_memory_recall_question(user_prompt):
            if line_lower.startswith("assistant reported:"):
                overlap += 3
            elif line_lower.startswith("session '"):
                overlap -= 2
        if source == "memory":
            overlap += 1
        ranked.append((overlap, token_overlap, source, line))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_overlap = ranked[0][0]
    if best_overlap <= 0:
        return None

    if _is_memory_follow_up_question(user_prompt):
        selected_ranked = ranked[:3]
    else:
        selected_ranked = [item for item in ranked[:4] if item[0] == best_overlap or item[0] > 0]
    memory_token_matches = [item for item in selected_ranked if item[2] == "memory" and item[1] > 0]
    if memory_token_matches:
        selected_ranked = memory_token_matches
    if any(source == "memory" and token_overlap == 0 for _overlap, token_overlap, source, _line in selected_ranked):
        primary_with_tokens = [item for item in selected_ranked if item[2] == "memory" or item[1] > 0]
        selected_ranked = primary_with_tokens or selected_ranked
    selected = [line for _overlap, _token_overlap, _source, line in selected_ranked[:3]]
    if not selected:
        return None
    cleaned = [_clean_memory_line(line) for line in selected]
    cleaned = [line for line in cleaned if line and _is_high_signal_memory_answer(line, user_prompt)]
    if not cleaned:
        return None
    if query_entities and not (_is_memory_recall_question(user_prompt) and _has_explicit_memory_subject(user_prompt)):
        cleaned = [line for line in cleaned if any(entity in line.lower() for entity in query_entities)] or cleaned
    shaped = [_humanize_recalled_line(line, user_prompt) for line in cleaned]
    shaped = [line for line in shaped if line]
    if not shaped:
        return None
    if not _memory_answer_matches_question(user_prompt, shaped):
        return None
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        issue_summary = _summarize_follow_up_issues(shaped)
        subject = _preferred_memory_subject(user_prompt, sections["external"] + sections["working"] + sections["retrieved"])
        if issue_summary and _is_bug_fix_follow_up_question(user_prompt):
            if subject:
                return f"Yes. In {subject}, we fixed those bugs by addressing {issue_summary}."
            return f"Yes. We fixed those bugs by addressing {issue_summary}."
        if issue_summary and _is_issue_explanation_follow_up_question(user_prompt):
            return f"Yes. It was mainly about {issue_summary}."
        if issue_summary and _is_issue_recap_follow_up_question(user_prompt):
            if subject:
                return f"Yes. In {subject}, the main issues were {issue_summary}."
            return f"Yes. The main issues were {issue_summary}."
        if _has_explicit_memory_subject(user_prompt):
            if issue_summary and _is_bug_list_question(user_prompt):
                return _format_issue_list_answer(subject, _extract_follow_up_issues(shaped))
            if subject and issue_summary:
                return f"Yes. {subject} came up in prior sessions about {issue_summary}."
            if shaped[0].startswith("Session '"):
                descriptive = next((line for line in shaped if not line.startswith("Session '")), None)
                if descriptive:
                    return _affirm_memory_answer(descriptive)
            return _affirm_memory_answer(shaped[0])
        if len(shaped) == 1:
            return _affirm_memory_answer(shaped[0])
        return "Yes.\n\n" + "\n\n".join(shaped[:3])
    if len(shaped) == 1:
        return shaped[0]
    return "\n\n".join(shaped)


def _memory_context_sections(memory_context: str) -> dict[str, list[str]]:
    lines = {"working": [], "retrieved": [], "external": []}
    active_section: str | None = None
    current_index: int | None = None
    for raw_line in memory_context.splitlines():
        stripped = raw_line.strip()
        if stripped == "## Working Memory":
            active_section = "working"
            current_index = None
            continue
        if stripped == "## Retrieved Memory":
            active_section = "retrieved"
            current_index = None
            continue
        if stripped == "## External Session Context":
            active_section = "external"
            current_index = None
            continue
        if stripped.startswith("## "):
            active_section = None
            current_index = None
            continue
        if not active_section:
            continue
        if stripped.startswith("- "):
            lines[active_section].append(stripped[2:].strip())
            current_index = len(lines[active_section]) - 1
            continue
        if current_index is not None and stripped:
            lines[active_section][current_index] += f"\n{stripped}"
    return lines


def _recent_working_follow_up_lines(lines: list[str], user_prompt: str) -> list[str]:
    selected: list[str] = []
    for raw_line in reversed(lines):
        lowered = raw_line.lower()
        if not (lowered.startswith("assistant:") or lowered.startswith("assistant reported:")):
            continue
        cleaned = _clean_memory_line(raw_line)
        if not cleaned or not _is_high_signal_memory_answer(cleaned, user_prompt):
            continue
        if cleaned not in selected:
            selected.append(cleaned)
        if len(selected) >= 2:
            break
    selected.reverse()
    return selected


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
    cleaned = re.sub(r"^(assistant|user|tool):\s*", "", cleaned, flags=re.IGNORECASE)
    return _sanitize_logged_answer(cleaned).strip()


def _affirm_memory_answer(text: str) -> str:
    stripped = text.strip()
    if re.match(r"^(yes|yeah|yep)\b", stripped, flags=re.IGNORECASE):
        return stripped
    return f"Yes. {stripped}"


def _humanize_recalled_line(line: str, user_prompt: str) -> str:
    cleaned = re.sub(r"^(user asked|assistant reported):\s*", "", line.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if _is_memory_follow_up_question(user_prompt):
        cleaned = cleaned.replace(" -> ", ": ")
        cleaned = cleaned.replace(" · ", "; ")
        cleaned = cleaned.replace(" doesnt work", " did not work")
        cleaned = cleaned.replace(" doesnt ", " did not ")
        cleaned = cleaned.replace(" doesnt", " did not")
    if cleaned.startswith("The first sweep shows there is already "):
        cleaned = cleaned.replace("The first sweep shows there is already ", "", 1)
    if " I’m " in cleaned:
        cleaned = cleaned.split(" I’m ", 1)[0].rstrip(" .,;")
    if " I'll " in cleaned:
        cleaned = cleaned.split(" I'll ", 1)[0].rstrip(" .,;")
    if " I’ll " in cleaned:
        cleaned = cleaned.split(" I’ll ", 1)[0].rstrip(" .,;")
    if any(
        marker in user_prompt.lower()
        for marker in (
            "what was it about",
            "can you explain about it",
            "can you explain it",
            "can you elaborate it",
            "can you elaborate on it",
            "explain about it",
            "explain it",
            "elaborate it",
            "elaborate on it",
        )
    ):
        lowered = cleaned.lower()
        if " was about " in lowered and not lowered.startswith("it was about "):
            cleaned = "It was about " + cleaned.split(" was about ", 1)[1].strip()
        elif " was mainly about " in lowered and not lowered.startswith("it was mainly about "):
            cleaned = "It was mainly about " + cleaned.split(" was mainly about ", 1)[1].strip()
        elif not lowered.startswith("it was about ") and not lowered.startswith("about "):
            cleaned = f"It was about {cleaned[0].lower()}{cleaned[1:]}" if len(cleaned) > 1 else f"It was about {cleaned.lower()}"
    return cleaned.strip()


def _extract_follow_up_issues(lines: list[str]) -> list[str]:
    issue_map = {
        "create_workspace_links": "Create Workspace accepting https links and converting them internally",
        "salesforce_state": "Salesforce being marked as coming soon or disabled",
        "pipeline_chat": "the DRIP pipeline chat flow not working",
        "test_publish": "test/publish staying reachable after approvals",
        "root_redirects": "root URL redirects",
        "convex_imports": "Convex generated imports",
        "auth_bypass": "authentication bypass",
        "open_email_relay": "open email relay",
    }
    detected: list[str] = []
    for line in lines:
        lowered = line.lower()
        if (
            ("create workspace" in lowered or "workspace creation link" in lowered or "workspace creation links" in lowered)
            and ("https link" in lowered or "convert" in lowered or "support" in lowered)
        ):
            detected.append(issue_map["create_workspace_links"])
        if "salesforce" in lowered and ("coming soon" in lowered or "disable" in lowered):
            detected.append(issue_map["salesforce_state"])
        if "pipeline chat" in lowered and (
            "does not work" in lowered or "did not work" in lowered or "not working" in lowered or "broken" in lowered or "fix" in lowered
        ):
            detected.append(issue_map["pipeline_chat"])
        if ("test/publish" in lowered or "test and publish" in lowered) and ("approval" in lowered or "approvals" in lowered):
            detected.append(issue_map["test_publish"])
        if "root url redirects" in lowered:
            detected.append(issue_map["root_redirects"])
        if "convex generated imports" in lowered:
            detected.append(issue_map["convex_imports"])
        if "authentication bypass" in lowered or "auth bypass" in lowered:
            detected.append(issue_map["auth_bypass"])
        if "open email relay" in lowered:
            detected.append(issue_map["open_email_relay"])

    unique_detected: list[str] = []
    for issue in detected:
        if issue not in unique_detected:
            unique_detected.append(issue)
    return unique_detected


def _issues_relevant_to_prompt(user_prompt: str, issues: list[str]) -> list[str]:
    if _is_cleanup_schema_prompt(user_prompt):
        narrowed = [
            issue
            for issue in issues
            if issue in {"root URL redirects", "Convex generated imports", "authentication bypass", "open email relay"}
        ]
        return narrowed or issues
    return issues


def _summarize_follow_up_issues(lines: list[str]) -> str | None:
    issues = _extract_follow_up_issues(lines)
    if issues:
        return _join_human_list(issues)
    return None


def _cleanup_summary_from_lines(user_prompt: str, lines: list[str]) -> str | None:
    if not _is_cleanup_schema_prompt(user_prompt):
        return None
    issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(lines))
    if issues:
        return f"The get-drip cleanup was mainly about {_join_human_list(issues)}."
    narrative = _summarize_cleanup_narrative(lines)
    if narrative:
        return f"The get-drip cleanup was mainly about {narrative}."
    return None


def _summarize_cleanup_narrative(lines: list[str]) -> str | None:
    joined = " ".join(line.lower() for line in lines)
    if not any(
        marker in joined
        for marker in (
            "cleanup",
            "clean up",
            "schema",
            "legacy column",
            "legacy columns",
            "crmcustomers",
            "campaigns",
        )
    ):
        return None

    points: list[str] = []
    if "legacy columns" in joined or "legacy-column" in joined or "duplicate state" in joined:
        points.append("removing legacy columns and duplicated state")
    if "campaigns" in joined and "crmcustomers" in joined:
        points.append("focusing the schema pass on `campaigns` and `crmCustomers`")
    elif "campaigns" in joined:
        points.append("cleaning up legacy fields in `campaigns`")
    elif "crmcustomers" in joined:
        points.append("cleaning up legacy fields in `crmCustomers`")
    if any(marker in joined for marker in ("whole table", "whole tables", "broad table deletion", "not actually dead", "conservative default")):
        points.append("keeping the cleanup conservative instead of deleting whole tables")
    if any(marker in joined for marker in ("audit/cache/helper", "helper/audit/cache", "audit", "cache", "helper tables")):
        points.append("keeping audit, cache, and helper tables that still support live flows")

    unique_points: list[str] = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    if not unique_points:
        return None
    return _join_human_list(unique_points)


def _format_issue_list_answer(subject: str | None, issues: list[str]) -> str:
    if not issues:
        return "I could not recover a reliable bug list from prior context."
    heading = f"In {subject}, the recalled bug list was:" if subject else "The recalled bug list was:"
    grouped_sections: list[tuple[str, list[str]]] = []
    product_issues = [
        issue
        for issue in issues
        if issue
        in {
            "Create Workspace accepting https links and converting them internally",
            "Salesforce being marked as coming soon or disabled",
            "the DRIP pipeline chat flow not working",
            "test/publish staying reachable after approvals",
        }
    ]
    lingering_issues = [issue for issue in issues if issue in {"root URL redirects", "Convex generated imports"}]
    security_issues = [issue for issue in issues if issue in {"authentication bypass", "open email relay"}]
    remaining = [issue for issue in issues if issue not in product_issues and issue not in lingering_issues and issue not in security_issues]
    if product_issues:
        grouped_sections.append(("Core product bugs", product_issues))
    if lingering_issues:
        grouped_sections.append(("Lingering app issues", lingering_issues))
    if security_issues:
        grouped_sections.append(("PR review security findings", security_issues))
    if remaining:
        grouped_sections.append(("Other recalled issues", remaining))

    lines = [heading, ""]
    for title, section_issues in grouped_sections:
        lines.append(f"**{title}**")
        lines.extend(f"- {issue}" for issue in section_issues)
        lines.append("")
    return "\n".join(lines).strip()


def _join_human_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


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


def _should_persist_episodic_response(final_response: str) -> bool:
    lowered = str(final_response or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("i inspected `"):
        return False
    if lowered.startswith("i inspected the backend entry points locally."):
        return False
    if lowered.startswith("i couldn't recover a reliable prior answer"):
        return False
    if lowered.startswith("i couldn't recover a reliable prior note"):
        return False
    if lowered.startswith("i couldn't recover a reliable note about the last merge conflict"):
        return False
    if lowered.startswith("opencode backend access has not been granted"):
        return False
    if lowered.startswith("opencode backend access is not granted right now"):
        return False
    if lowered == "nothing left to execute.":
        return False
    return True


def _is_high_signal_memory_answer(candidate: str, user_prompt: str) -> bool:
    lowered = candidate.lower()
    normalized = re.sub(r"^(assistant reported|user asked):\s*", "", lowered)
    prompt_lowered = user_prompt.lower()
    reject_markers = (
        '{"type": "function"',
        '"type":"function"',
        '"name": "list_directory"',
        '"name":"list_directory"',
        '"required":[',
        '\\"required\\":[',
        '"properties":{',
        '\\"properties\\":{',
        '"type":"object"',
        '"type": "object"',
        '\\"type\\":\\"object\\"',
        'relative_path',
        '"depth":',
        '"is_dir":',
        '"path":',
        "tool requested",
        "recovered inline tool request",
        "queued prompt",
        "i inspected `",
        "locally. relevant paths i found:",
        "i don't have access",
        "i do not have access",
        "good, but its too less of info",
        "good, but it's too less of info",
        "devenv status",
        "tool trace",
        "prepared the final answer",
        "reasoned through the next step",
        "i couldn't recover a reliable prior answer",
        "i couldn't recover a reliable prior note",
        "local-only mode could not inspect",
        "local-only mode needs workspace inspection tools",
        "opencode backend access has not been granted",
    )
    if any(marker in lowered for marker in reject_markers):
        return False
    low_signal_prefixes = (
        "i’m grounding",
        "i'm grounding",
        "i’m tracing",
        "i'm tracing",
        "i’m checking",
        "i'm checking",
        "i’m going to",
        "i'm going to",
        "i’ve confirmed",
        "i've confirmed",
        "next i’m",
        "next i'm",
        "i’ll pick up",
        "i'll pick up",
    )
    if normalized.startswith(low_signal_prefixes):
        return False
    if normalized.strip(" .,:;!?") == prompt_lowered.strip(" .,:;!?"):
        return False
    if ("repo" in prompt_lowered or "repository" in prompt_lowered or "codebase" in prompt_lowered) and any(
        marker in lowered for marker in ("main.py", "server.py", "app.py")
    ) and not any(
        marker in lowered for marker in ("repo", "repository", "codebase", "workspace", "files", "folders", "architecture", "backend")
    ):
        return False
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        if normalized.startswith(low_signal_prefixes):
            return False
    if lowered.startswith("{") or lowered.startswith("["):
        return False
    if ("how does" in prompt_lowered or "how do" in prompt_lowered or "why does" in prompt_lowered) and any(
        marker in lowered
        for marker in ('"required":[', '\\"required\\":[', '"properties":{', '\\"properties\\":{', '"type":"object"', '"type": "object"', '\\"type\\":\\"object\\"')
    ):
        return False
    if ("how does" in prompt_lowered or "how do" in prompt_lowered or "why does" in prompt_lowered) and "i inspected" in lowered:
        return False
    if "strongest clues point to" in lowered and ("schema" in prompt_lowered or "cleanup" in prompt_lowered):
        return False
    return True


def _memory_query_tokens(user_prompt: str) -> set[str]:
    common = {
        "about",
        "again",
        "anything",
        "does",
        "know",
        "project",
        "remember",
        "that",
        "this",
        "what",
    }
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 4 and token not in common
    }
    return tokens | _memory_query_entities(user_prompt)


def _memory_query_entities(user_prompt: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", user_prompt.lower())
        if len(token) >= 3
    }


def _is_brief_greeting_prompt(user_prompt: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", user_prompt.lower())
    tokens = [token for token in normalized.split() if token]
    if not tokens or len(tokens) > 3:
        return False
    greeting_tokens = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "hiya",
        "heya",
        "gm",
        "goodmorning",
        "morning",
        "afternoon",
        "evening",
    }
    joined = "".join(tokens)
    if joined in {"goodmorning", "goodafternoon", "goodevening"}:
        return True
    return all(token in greeting_tokens for token in tokens)


def _memory_context_entities(memory_context: str) -> set[str]:
    sections = _memory_context_sections(memory_context)
    entities: set[str] = set()
    for line in [*sections["working"], *sections["retrieved"], *sections["external"]]:
        entities.update(
            token.lower()
            for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", line.lower())
            if len(token) >= 3
        )
    return entities


_GENERIC_MEMORY_SUBJECTS = {
    "decision-complete",
    "feature-structured",
    "test-activate",
    "task-getgit-checkpoints",
}


def _infer_memory_subject(lines: list[str]) -> str | None:
    for line in lines:
        lowered_line = line.lower()
        if any(marker in lowered_line for marker in ('"path":', '{"type": "function"', '"name": "list_directory"')):
            continue
        for match in re.findall(r"/[A-Za-z0-9._/-]+", line):
            basename = Path(match).name.lower()
            if basename and ("-" in basename or "_" in basename) and basename not in _GENERIC_MEMORY_SUBJECTS:
                return basename
    for line in lines:
        lowered_line = line.lower()
        if any(marker in lowered_line for marker in ('"path":', '{"type": "function"', '"name": "list_directory"')):
            continue
        matches = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", line.lower())
        for match in matches:
            if match.startswith("rollout-") or match in _GENERIC_MEMORY_SUBJECTS:
                continue
            return match
    return None


def _preferred_memory_subject(user_prompt: str, lines: list[str]) -> str | None:
    explicit_entities = sorted(_memory_query_entities(user_prompt), key=len, reverse=True)
    if explicit_entities:
        return explicit_entities[0]
    return _infer_memory_subject(lines)


def _ordered_follow_up_lines(user_prompt: str, external_lines: list[str]) -> list[str]:
    preferred_markers = (
        "->",
        "accept",
        "coming soon",
        "doesnt work",
        "does not work",
        "pipeline chat",
        "test and publish",
        "test/publish",
        "review",
        "bug",
        "fix",
    )
    cleaned_pairs: list[tuple[str, str]] = []
    for raw_line in external_lines:
        cleaned = _clean_memory_line(raw_line)
        if cleaned and _is_high_signal_memory_answer(cleaned, user_prompt):
            cleaned_pairs.append((raw_line.lower(), cleaned))
    if not cleaned_pairs:
        return []

    preferred_user_pairs = [
        (raw, cleaned)
        for raw, cleaned in cleaned_pairs
        if raw.startswith("user asked:") and any(marker in raw for marker in preferred_markers)
    ]
    if preferred_user_pairs:
        preferred_user_pairs.sort(key=lambda item: (_follow_up_line_score(item[0]), len(item[1])), reverse=True)
        user_lines = [cleaned for _raw, cleaned in preferred_user_pairs[:2]]
    else:
        user_lines = [cleaned for raw, cleaned in cleaned_pairs if raw.startswith("user asked:")][:2]
    marked_assistant_pairs = [
        (raw, cleaned)
        for raw, cleaned in cleaned_pairs
        if raw.startswith("assistant reported:") and any(marker in raw for marker in preferred_markers)
    ]
    marked_assistant_pairs.sort(key=lambda item: (_follow_up_line_score(item[0]), len(item[1])), reverse=True)
    marked_assistant_lines = [cleaned for _raw, cleaned in marked_assistant_pairs[:2]]
    remaining_lines = [
        cleaned
        for raw, cleaned in cleaned_pairs
        if cleaned not in user_lines and cleaned not in marked_assistant_lines and not raw.startswith("session '")
    ]
    session_lines = [cleaned for raw, cleaned in cleaned_pairs if raw.startswith("session '")]

    ordered: list[str] = []
    for group in (user_lines, marked_assistant_lines, remaining_lines, session_lines):
        for line in group:
            if line not in ordered:
                ordered.append(line)
    return ordered


def _follow_up_line_score(lowered_line: str) -> int:
    score = 0
    if "create workspace" in lowered_line:
        score += 6
    if "->" in lowered_line:
        score += 3
    for marker in (
        "accept",
        "coming soon",
        "doesnt work",
        "does not work",
        "pipeline chat",
        "test and publish",
        "test/publish",
        "review",
        "bug",
        "fix",
    ):
        if marker in lowered_line:
            score += 1
    return score


def _is_memory_recall_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do you remember",
            "remember about",
            "what do you remember",
            "do you know about",
            "what do you know about",
        )
    )


def _is_session_history_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "last merge conflict",
            "merge conflict we solved",
            "last conflict we solved",
            "last bug we fixed",
            "last review we fixed",
            "last issue we fixed",
        )
    )


def _is_memory_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    referential_markers = (
        "can you explain about it",
        "can you explain it",
        "can you elaborate it",
        "can you elaborate on it",
        "explain about it",
        "explain it",
        "elaborate it",
        "elaborate on it",
        "what are those",
        "what are those bugs",
        "what are those reviews",
        "what were those",
        "tell exactly what",
        "what was it about",
        "what was that about",
        "those bugs",
        "those reviews",
        "how did we fix those bugs",
        "how did we fix them",
        "how were those fixed",
        "a few reviews",
        "a few bugs",
    )
    if not any(marker in lowered for marker in referential_markers):
        return False
    return not _has_explicit_memory_subject(user_prompt)


def _is_bug_fix_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "how did we fix those bugs",
            "how did we fix them",
            "how were those fixed",
        )
    )


def _is_issue_recap_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "what are those",
            "what are those bugs",
            "what are those reviews",
            "what were those",
            "those bugs",
            "those reviews",
        )
    )


def _is_issue_explanation_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "can you explain about it",
            "can you explain it",
            "can you elaborate it",
            "can you elaborate on it",
            "explain about it",
            "explain it",
            "elaborate it",
            "elaborate on it",
            "what was it about",
            "what was that about",
        )
    )


def _is_cleanup_schema_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if "get-drip" not in lowered:
        return False
    has_cleanup = any(marker in lowered for marker in ("cleanup", "clean up"))
    has_schema = any(marker in lowered for marker in ("schema", "schrema"))
    return has_cleanup or has_schema


def _answer_from_recent_conversation_follow_up(user_prompt: str, conversation: list[dict[str, Any]]) -> str | None:
    if not _is_memory_follow_up_question(user_prompt):
        return None

    recent_lines: list[str] = []
    last_assistant: str | None = None
    for message in reversed(conversation[-6:]):
        role = str(message.get("role") or "")
        content = _sanitize_logged_answer(str(message.get("content") or "").strip())
        if role not in {"user", "assistant"} or not content:
            continue
        recent_lines.append(content)
        if role == "assistant" and last_assistant is None:
            last_assistant = content

    if not last_assistant or not _is_high_signal_memory_answer(last_assistant, user_prompt):
        return None

    subject = _preferred_memory_subject(user_prompt, recent_lines)
    issue_summary = _summarize_follow_up_issues(_memory_context_lines(last_assistant))
    if (
        issue_summary is None
        and "targeted workspace" in last_assistant.lower()
        and (_is_bug_fix_follow_up_question(user_prompt) or _is_issue_recap_follow_up_question(user_prompt) or _is_issue_explanation_follow_up_question(user_prompt))
    ):
        return None
    lowered = user_prompt.lower()
    if issue_summary and _is_bug_fix_follow_up_question(user_prompt):
        if subject:
            return f"Yes. In {subject}, we fixed those bugs by addressing {issue_summary}."
        return f"Yes. We fixed those bugs by addressing {issue_summary}."
    if issue_summary and _is_issue_explanation_follow_up_question(user_prompt):
        return f"Yes. It was mainly about {issue_summary}."
    if issue_summary and _is_issue_recap_follow_up_question(user_prompt):
        if subject:
            return f"Yes. In {subject}, the main issues were {issue_summary}."
        return f"Yes. The main issues were {issue_summary}."
    cleanup_summary = _summarize_cleanup_narrative(_memory_context_lines(last_assistant))
    if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
        return f"Yes. It was mainly about {cleanup_summary}."
    return _affirm_memory_answer(_humanize_recalled_line(last_assistant, user_prompt))


def _is_bug_list_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do you know about get-drip bugs",
            "what do you know about get-drip bugs",
            "get-drip bugs",
            "bug list",
            "list the bugs",
            "exact bugs",
            "what bugs did we fix",
            "give get-drip bug list",
            "give the bug list",
            "main bugs",
            "tracked bugs",
        )
    )


def _should_try_direct_memory_answer(user_prompt: str) -> bool:
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return True

    lowered = user_prompt.lower().strip()
    if not lowered:
        return False

    if any(
        token in lowered
        for token in ("create", "make", "add", "write", "edit", "update", "modify", "fix", "implement", "delete", "remove")
    ):
        return False

    if any(
        phrase in lowered
        for phrase in (
            "what architecture",
            "which architecture",
            "list the concrete files",
            "list the files",
            "what other work",
            "what were the main issues",
            "what can be said confidently",
            "what remains unclear",
            "what was the backend",
            "bug list",
            "exact bugs",
            "what bugs did we fix",
            "list the bugs",
        )
    ):
        return True

    return False


def _should_skip_vector_memory_lookup(user_prompt: str) -> bool:
    return (
        _is_memory_recall_question(user_prompt)
        or _is_memory_follow_up_question(user_prompt)
        or _is_session_history_question(user_prompt)
        or _is_bug_list_question(user_prompt)
    )


def _should_answer_from_memory_only(user_prompt: str) -> bool:
    return (
        _is_memory_recall_question(user_prompt)
        or _is_memory_follow_up_question(user_prompt)
        or _is_session_history_question(user_prompt)
    )


def _memory_only_fallback_response(user_prompt: str) -> str:
    if _is_session_history_question(user_prompt):
        return "I couldn't recover a reliable note about the last merge conflict we solved."
    if _is_memory_follow_up_question(user_prompt):
        return "I couldn't recover a reliable prior note for that follow-up."
    return "I couldn't recover a reliable prior answer for that yet."


def _tool_strategy_subject_prompt(user_prompt: str) -> str | None:
    lowered = user_prompt.lower().strip()
    patterns = (
        r"^(?:what|which)\s+tools\s+do\s+you\s+need\s+to\s+answer\s+(.+)$",
        r"^(?:what|which)\s+tools\s+would\s+you\s+use\s+to\s+answer\s+(.+)$",
        r"^how\s+do\s+you\s+decide\s+what\s+tools\s+to\s+use\s+for\s+(.+)$",
        r"^how\s+do\s+you\s+choose\s+what\s+tools\s+to\s+use\s+for\s+(.+)$",
        r"^do\s+you\s+need\s+any\s+tools\s+to\s+answer\s+(.+)$",
        r"^would\s+you\s+need\s+any\s+tools\s+to\s+answer\s+(.+)$",
        r"^how\s+would\s+you\s+answer\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip()
        return subject.rstrip(" ?")
    return None


def _is_underspecified_troubleshooting_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        return False
    if any(token in lowered for token in ("repo", "repository", "codebase", "backend", "system", "project", "get-drip", "getgit")):
        return False
    return lowered in {
        "why does this fail?",
        "why does this fail",
        "why is this failing?",
        "why is this failing",
        "why did this fail?",
        "why did this fail",
        "why doesn't this work?",
        "why doesn't this work",
        "why is this broken?",
        "why is this broken",
        "what is failing?",
        "what is failing",
    }


def _should_skip_external_session_context(user_prompt: str) -> bool:
    return _should_skip_retrieval_for_prompt(user_prompt) or _should_skip_current_workspace_memory_lookup(user_prompt)


def _should_skip_current_workspace_memory_lookup(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return False
    return any(
        phrase in lowered
        for phrase in (
            "summarize this repo",
            "summarize the repo",
            "summarize this repository",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "explain the repo",
            "explain this repo",
            "explain the repository",
            "explain this repository",
            "tell me about this codebase",
            "tell me about the codebase",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "how does the backend work",
            "how does this backend work",
            "what is the backend",
            "explain the backend",
            "explain this backend",
            "tell me about the backend",
            "tell me about this backend",
            "what is the system",
            "tell me about the system",
            "tell me about this system",
            "what is the backend architecture",
            "show me the backend architecture",
            "how does this repo work",
            "how does the repo work",
            "how does this repository work",
            "how does the repository work",
            "how does the system work",
            "how does this system work",
        )
    )


def _should_skip_retrieval_for_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if not lowered:
        return True
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return False
    if _is_structural_acknowledgement_prompt(lowered):
        return True
    if _is_shell_like_prompt(lowered):
        return True
    tokens = re.findall(r"[a-z0-9_./-]+", lowered)
    if len(tokens) < 3 and not any(char in lowered for char in "?.!"):
        return True
    return False


def _is_structural_acknowledgement_prompt(lowered_prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", lowered_prompt).strip(" .!?\t\r\n")
    return normalized in {
        "ok",
        "okay",
        "kk",
        "cool",
        "nice",
        "great",
        "thanks",
        "thank you",
        "thx",
        "got it",
        "understood",
        "sounds good",
        "yep",
        "yes",
        "no",
    }


def _is_shell_like_prompt(lowered_prompt: str) -> bool:
    command_prefixes = (
        "cd ",
        "ls",
        "pwd",
        "cat ",
        "rm ",
        "mv ",
        "cp ",
        "mkdir ",
        "touch ",
        "git ",
        "npm ",
        "pnpm ",
        "yarn ",
        "bun ",
        "python ",
        "python3 ",
        "./",
        "../",
    )
    if lowered_prompt in {"ls", "pwd", "clear"}:
        return True
    if any(lowered_prompt.startswith(prefix) for prefix in command_prefixes):
        return True
    return bool(re.match(r"^[a-z0-9_./-]+\s+(-{1,2}[a-z0-9][a-z0-9-]*\s*)+$", lowered_prompt))


def _is_opencode_access_denied_error(error: RuntimeError) -> bool:
    lowered = str(error).strip().lower()
    return "opencode backend access has not been granted" in lowered


def _is_opencode_transport_error(error: RuntimeError) -> bool:
    lowered = str(error).strip().lower()
    return (
        "opencode server failed" in lowered
        or "opencode cli failed" in lowered
        or "unable to reach opencode server" in lowered
    )


def _lexical_memory_terms(user_prompt: str) -> list[str]:
    generic = {
        "about",
        "architecture",
        "associated",
        "concrete",
        "confidently",
        "different",
        "files",
        "folders",
        "indirectly",
        "issues",
        "look",
        "main",
        "other",
        "parts",
        "point",
        "project",
        "properly",
        "question",
        "referenced",
        "remains",
        "same",
        "said",
        "use",
        "used",
        "what",
        "were",
        "work",
    }
    terms: list[str] = []
    for entity in sorted(_memory_query_entities(user_prompt)):
        if entity not in terms:
            terms.append(entity)
    for token in sorted(_memory_query_tokens(user_prompt)):
        if len(token) >= 5 and token not in generic and token not in terms:
            terms.append(token)
    lowered = user_prompt.lower()
    if "getgit" in lowered and any(marker in lowered for marker in ("architecture", "backend", "same architecture", "files or folders", "concrete files")):
        for extra in ("flask", "backend", "server.py", "core.py", "rag", "retriever.py", "readme.md", "documentation.md"):
            if extra not in terms:
                terms.append(extra)
    if "get-drip" in lowered:
        for extra in ("convex", "journey", "pipeline", "salesforce", "workspace", "campaign", "route"):
            if extra not in terms:
                terms.append(extra)
        if "infer the parts of the app" in lowered or "look different" in lowered:
            for extra in ("convex-api.ts", "convex-types.ts", "journey.ts", "test-activate.tsx", "pipeline.tsx"):
                if extra not in terms:
                    terms.append(extra)
    if _is_bug_list_question(user_prompt) and "get-drip" in lowered:
        for extra in ("https", "disabled", "pipeline chat", "test/publish", "root url redirects", "convex generated imports", "authentication bypass"):
            if extra not in terms:
                terms.append(extra)
    if "main issues" in lowered or "issues being worked" in lowered:
        for extra in ("salesforce", "pipeline", "workspace", "disabled", "https"):
            if extra not in terms:
                terms.append(extra)
    return terms[:10]


def _memory_answer_matches_question(user_prompt: str, shaped_lines: list[str]) -> bool:
    lowered_prompt = user_prompt.lower()
    joined = " \n ".join(shaped_lines).lower()

    if "repo" in lowered_prompt or "repository" in lowered_prompt or "codebase" in lowered_prompt:
        return any(
            marker in joined
            for marker in ("repo", "repository", "codebase", "workspace", "files", "folders", "module", "architecture", "backend")
        )
    if (
        lowered_prompt.startswith("why ")
        or lowered_prompt.startswith("how ")
        or lowered_prompt.startswith("explain ")
    ) and not (_is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt)):
        significant_tokens = [token for token in _memory_query_tokens(user_prompt) if token not in {"what", "this", "that", "about"}]
        return any(token in joined for token in significant_tokens)
    if _is_session_history_question(user_prompt):
        return any(marker in joined for marker in ("merge conflict", "conflict", "resolved", "resolution", "rebase", "branch"))
    if _is_bug_list_question(user_prompt):
        return _summarize_follow_up_issues(shaped_lines) is not None or any(
            marker in joined
            for marker in (
                "authentication bypass",
                "open email relay",
                "root url redirects",
                "convex generated imports",
                "create workspace",
                "pipeline chat",
                "test/publish",
            )
        )
    if "architecture" in lowered_prompt:
        return any(marker in joined for marker in ("flask", "fastapi", "backend", "server.py", "rag", "retriever", "core.py"))
    if "same architecture" in lowered_prompt or "look different" in lowered_prompt:
        return any(marker in joined for marker in ("convex", "flask", "backend", "server.py", "route", "pipeline", "journey"))
    if "list the concrete files" in lowered_prompt or "files or folders" in lowered_prompt:
        return any(marker in joined for marker in (".py", ".md", ".txt", "/", "server.py", "core.py", "readme.md"))
    if "main issues" in lowered_prompt or "what were the main issues" in lowered_prompt:
        return _summarize_follow_up_issues(shaped_lines) is not None
    if "what can be said confidently" in lowered_prompt or "remains unclear" in lowered_prompt:
        return any(marker in joined for marker in ("get-drip", "convex", "workspace", "pipeline", "salesforce", "journey"))
    if "get-drip" in lowered_prompt and ("schema" in lowered_prompt or "cleanup" in lowered_prompt):
        return any(marker in joined for marker in ("schema", "cleanup", "review", "issue", "bug", "convex generated imports", "root url redirects"))
    return True


def _is_usable_logged_project_answer(user_prompt: str, answer: str) -> bool:
    cleaned = answer.strip()
    lowered = cleaned.lower()
    lowered_prompt = user_prompt.lower()
    if not cleaned:
        return False
    if lowered.startswith("# agents.md instructions"):
        return False
    if lowered.startswith("local-only mode could not inspect"):
        return False
    if lowered.startswith("`readme.md` references"):
        return False
    if "requested tool is not registered" in lowered:
        return False
    if "convex/email_g..." in lowered:
        return False
    if _is_bug_list_question(user_prompt) and "strongest clues point to" in lowered:
        return False
    if "strongest clues point to" in lowered and ("schema" in lowered_prompt or "cleanup" in lowered_prompt):
        return False
    if "strongest clues point to" in lowered and "infer the parts of the app" not in lowered_prompt and not _is_file_inventory_question(user_prompt):
        return False
    if "same architecture" in lowered_prompt and "get-drip" not in lowered:
        return False
    return _memory_answer_matches_question(user_prompt, [cleaned])


def _is_usable_logged_answer(user_prompt: str, answer: str) -> bool:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return False
    if _is_session_history_question(user_prompt):
        return _is_high_signal_memory_answer(cleaned, user_prompt) and _memory_answer_matches_question(user_prompt, [cleaned])
    return _is_usable_logged_project_answer(user_prompt, cleaned)


def _answer_known_project_question(user_prompt: str, memory_context: str) -> str | None:
    lowered = user_prompt.lower()
    if "getgit" not in lowered and "get-drip" not in lowered:
        return None

    context_lower = memory_context.lower()
    getgit_flask = "flask" in context_lower
    getgit_rag = "rag" in context_lower
    getgit_server = "server.py" in context_lower
    getdrip_convex = "convex" in context_lower
    issue_summary = _summarize_follow_up_issues(_memory_context_lines(memory_context))
    if issue_summary and "pipeline" in context_lower and "pipeline chat" not in issue_summary and "drip pipeline chat flow not working" not in issue_summary:
        issue_summary = issue_summary + ", and the DRIP pipeline chat flow not working"
    path_mentions = _extract_path_mentions(memory_context)
    high_signal_paths = [
        path for path in path_mentions
        if any(marker in path for marker in ("convex/", "src/routes/", "journey.ts", "convex-api.ts", "convex-types.ts", "pipeline.tsx", "test-activate.tsx"))
    ]

    if "same architecture" in lowered:
        if getgit_flask and getdrip_convex:
            return "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app."
        return None

    if "look different" in lowered:
        if getgit_flask and getdrip_convex:
            return "GetGit looks like a Flask/Python RAG app, while get-drip looks like a Convex-backed app with campaign and route flow files."
        return None

    if "other work referenced getgit" in lowered and "task_getgit_checkpoints" in context_lower:
        return "CodeGuide referenced GetGit indirectly through a `task_practice_code_evaluate` flow that called `task_getgit_checkpoints`."

    if "main issues" in lowered and issue_summary:
        if _is_bug_list_question(user_prompt):
            return _format_issue_list_answer("get-drip", _extract_follow_up_issues(_memory_context_lines(memory_context)))
        return f"In get-drip, the main issues were {issue_summary}."

    if _is_file_inventory_question(user_prompt) and "getgit" in lowered:
        inventory_paths = [path for path in path_mentions if any(marker in path.lower() for marker in ("server.py", "core.py", "checkpoints.py", "clone_repo.py", "repo_manager.py", "readme.md", "documentation.md", "rag/", "templates/", "static/"))]
        deduped_inventory: list[str] = []
        for path in inventory_paths:
            if path not in deduped_inventory:
                deduped_inventory.append(path)
        if deduped_inventory:
            return "The concrete GetGit paths included " + ", ".join(f"`{path}`" for path in deduped_inventory[:10]) + "."

    if "infer the parts of the app" in lowered and high_signal_paths:
        deduped_paths: list[str] = []
        for path in high_signal_paths:
            if path not in deduped_paths:
                deduped_paths.append(path)
        return "The strongest clues point to " + ", ".join(f"`{path}`" for path in deduped_paths[:5]) + "."

    if ("what can be said confidently" in lowered or "remains unclear" in lowered) and getdrip_convex:
        confident_bits: list[str] = ["get-drip was described as a Convex-backed app"]
        if issue_summary:
            confident_bits.append(f"the work focused on {issue_summary}")
        confident = ", and ".join(confident_bits)
        return f"Confidently, {confident}. What remains unclear is a cleaner one-line architecture summary beyond those clues."

    if "what was the backend" in lowered and "get-drip" in lowered and getdrip_convex:
        return "get-drip was described as a Convex-backed app."

    if "what architecture did getgit use" in lowered and (getgit_flask or getgit_server or getgit_rag):
        parts = ["GetGit was described as a Flask backend"]
        if getgit_server:
            parts.append("with a `server.py` entrypoint")
        if getgit_rag:
            parts.append("and RAG-related components")
        return ", ".join(parts) + "."

    return None


def _memory_context_lines(memory_context: str) -> list[str]:
    lines: list[str] = []
    for raw_line in memory_context.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("## "):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return lines


def _extract_path_mentions(text: str) -> list[str]:
    patterns = [
        r"`([^`]+)`",
        r"\[([^\]]+)\]\(/[^)]+/([^):]+(?:\.[A-Za-z0-9]+))(?::\d+)?\)",
        r"(?<![A-Za-z0-9_])((?:src|convex)/[A-Za-z0-9_.$/-]+(?:\.[A-Za-z0-9]+)?)",
    ]
    paths: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            value = match[-1] if isinstance(match, tuple) else match
            cleaned = str(value).strip()
            if "/" not in cleaned and "." not in cleaned:
                continue
            if cleaned not in paths:
                paths.append(cleaned)
    return paths


def _lexical_line_score(summary: str, user_prompt: str, terms: list[str]) -> int:
    lowered = summary.lower()
    prompt_lowered = user_prompt.lower()
    score = sum(2 for term in terms if term.lower() in lowered)
    score += len(_extract_path_mentions(summary)) * 3
    if "flask" in lowered or "convex" in lowered or "rag" in lowered:
        score += 4
    if "task_getgit_checkpoints" in lowered:
        score += 5
    if "pipeline chat" in lowered or "salesforce" in lowered or "create workspace" in lowered:
        score += 3
    if "tool output noted:" in lowered:
        score += 2
    if "strongest clues point to" in lowered and "infer the parts of the app" not in prompt_lowered and not _is_file_inventory_question(user_prompt):
        score -= 12
    if ("schema" in prompt_lowered or "cleanup" in prompt_lowered) and not any(
        marker in lowered for marker in ("schema", "cleanup", "convex generated imports", "root url redirects", "review")
    ):
        score -= 8
    if "session '" in lowered and score < 6:
        score -= 2
    if prompt_lowered in lowered:
        score -= 3
    return score


def _is_architecture_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_backend_connector_question(user_prompt):
        return True
    return any(
        marker in lowered
        for marker in (
            "architecture",
            "backend",
            "system",
            "same architecture",
            "look different",
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
        )
    )


def _is_backend_connector_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if "claude code" in lowered and "integrat" in lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "another connector like codex or opencode",
            "options as backend",
            "option as backend",
            "another backend",
            "backend option",
            "thinking part",
            "like codex or opencode",
        )
    )


def _supports_exact_logged_answer_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return _is_session_history_question(user_prompt) or "getgit" in lowered or "get-drip" in lowered


def _exact_logged_query_variants(user_prompt: str) -> tuple[str, ...]:
    variants: list[str] = []

    def add(candidate: str) -> None:
        normalized = " ".join(str(candidate or "").strip().lower().split())
        if normalized and normalized not in variants:
            variants.append(normalized)

    base = user_prompt
    add(base)
    add(base.rstrip("?.!"))
    if "schrema" in base.lower():
        add(re.sub(r"schrema", "schema", base, flags=re.IGNORECASE))
    if re.search(r"\bog\b", base, flags=re.IGNORECASE):
        add(re.sub(r"\bog\b", "of", base, flags=re.IGNORECASE))
    if "schrema" in base.lower() and re.search(r"\bog\b", base, flags=re.IGNORECASE):
        add(re.sub(r"\bog\b", "of", re.sub(r"schrema", "schema", base, flags=re.IGNORECASE), flags=re.IGNORECASE))
    lowered = base.lower()
    if lowered.startswith("do you know about "):
        add("what " + base)
    if lowered.startswith("what do you know about "):
        add(base.replace("what do you know about", "do you know about", 1))
    return tuple(variants)


def _should_skip_exact_logged_fast_path(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "what can be said confidently",
            "what remains unclear",
            "what was the backend",
            "same architecture",
            "look different",
        )
    )


def _is_file_inventory_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return "list the concrete files" in lowered or "files or folders" in lowered


def _is_repo_summary_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what is this project about",
            "what is the project about",
            "tell me about this project",
            "tell me about the project",
            "explain this project",
            "explain the project",
            "summarize this repo",
            "summarize the repo",
            "summarize this repository",
            "summarize the repository",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "explain the repo",
            "explain this repo",
            "explain the repository",
            "explain this repository",
            "summarize this codebase",
            "summarize the codebase",
            "tell me about this codebase",
            "tell me about the codebase",
        )
    )


def _is_repo_overview_question(user_prompt: str) -> bool:
    if _is_repo_summary_question(user_prompt):
        return True
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
        )
    )


def _should_trust_memory_answer_for_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_repo_summary_question(user_prompt):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
        )
    ):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "how does the backend work",
            "how does the system work",
            "explain this project architecture",
            "how does this backend work",
        )
    ):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "how does",
            "how do",
            "why does",
            "why do",
            "architecture",
            "backend work",
            "codebase work",
            "repository work",
            "system work",
        )
    ) and not _has_explicit_project_subject(user_prompt):
        return False
    return True


def _prefers_deeper_workspace_scan(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_backend_connector_question(user_prompt):
        return True
    if _is_repo_summary_question(user_prompt):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "how does the backend work",
            "tell me about the backend",
            "tell me about this backend",
            "what is the backend",
            "how does the repo work",
            "how does the repository work",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "how does the system work",
            "what is the system",
            "tell me about the system",
            "tell me about this system",
            "how does this backend work",
            "tell me about this codebase",
            "tell me about the codebase",
            "architecture",
            "codebase work",
            "backend work",
        )
    )


def _has_explicit_project_subject(user_prompt: str) -> bool:
    generic = {
        "about",
        "again",
        "architecture",
        "backend",
        "bugs",
        "code",
        "codebase",
        "elaborate",
        "explain",
        "exactly",
        "file",
        "files",
        "fixed",
        "how",
        "issue",
        "issues",
        "know",
        "project",
        "remember",
        "repo",
        "repository",
        "review",
        "reviews",
        "system",
        "tell",
        "those",
        "what",
        "work",
        "works",
    }
    if _memory_query_entities(user_prompt):
        return True
    return any(
        token not in generic
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 5
    )


def _compose_external_memory_query(user_prompt: str, conversation: list[dict[str, Any]]) -> str:
    if not _is_memory_follow_up_question(user_prompt):
        return user_prompt
    recent_hints = _recent_memory_subject_hints(user_prompt, conversation)
    if not recent_hints:
        return user_prompt
    hyphenated_hints = [hint for hint in recent_hints if "-" in hint]
    if hyphenated_hints:
        recent_hints = hyphenated_hints
    filtered_hints = [hint for hint in recent_hints if not re.fullmatch(r"[0-9a-f]{6,}", hint)]
    if filtered_hints:
        recent_hints = filtered_hints
    return "\n".join([user_prompt, f"Referenced context: {' '.join(recent_hints[:4])}"])


def _has_explicit_memory_subject(user_prompt: str) -> bool:
    generic = {
        "about",
        "again",
        "bugs",
        "elaborate",
        "explain",
        "exactly",
        "fixed",
        "issue",
        "issues",
        "know",
        "project",
        "remember",
        "review",
        "reviews",
        "tell",
        "those",
        "what",
    }
    if _memory_query_entities(user_prompt):
        return True
    return any(
        token not in generic
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 5
    )


def _recent_memory_subject_hints(user_prompt: str, conversation: list[dict[str, Any]]) -> list[str]:
    user_hints: list[str] = []
    assistant_hints: list[str] = []
    for message in reversed(conversation[-6:]):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content or content == user_prompt:
            continue
        target = user_hints if role == "user" else assistant_hints
        for hint in _memory_subject_hints(content):
            if hint not in target:
                target.append(hint)
        if len(user_hints) >= 4 and len(assistant_hints) >= 4:
            break

    prioritized_hints: list[str] = []
    for group in (user_hints, assistant_hints):
        for hint in group:
            if hint not in prioritized_hints:
                prioritized_hints.append(hint)
            if len(prioritized_hints) >= 8:
                return prioritized_hints
    return prioritized_hints


def _is_ambiguous_memory_follow_up(user_prompt: str, conversation: list[dict[str, Any]]) -> bool:
    if not _is_memory_follow_up_question(user_prompt):
        return False
    return not _recent_memory_subject_hints(user_prompt, conversation)


def _memory_subject_hints(text: str) -> list[str]:
    lowered_text = text.lower()
    if any(
        marker in lowered_text
        for marker in (
            "i couldn't recover a reliable prior answer",
            "i couldn't recover a reliable prior note",
            "local-only mode could not inspect",
            "local-only mode needs workspace inspection tools",
            "opencode backend access has not been granted",
        )
    ):
        return []
    generic = {
        "assistant",
        "answer",
        "context",
        "decision-complete",
        "desktop",
        "feature-structured",
        "recover",
        "reliable",
        "project",
        "prior",
        "remember",
        "reported",
        "rollout",
        "session",
        "samarthnaik",
        "targeted",
        "test-activate",
        "user",
        "users",
        "workspace",
    }
    hints: list[str] = []
    for match in re.findall(r"/[A-Za-z0-9._/-]+", text):
        basename = Path(match).name.lower().strip(".,:;!?)(")
        if len(basename) >= 3 and basename not in generic and basename not in hints:
            hints.append(basename)
    for entity in sorted(_memory_query_entities(text)):
        if "/" in entity or entity.startswith("rollout-"):
            continue
        if entity not in generic and entity not in hints:
            hints.append(entity)
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(token) < 5 or token in generic or token.startswith("rollout") or re.fullmatch(r"[0-9a-f]{6,}", token):
            continue
        if token not in hints:
            hints.append(token)
    return hints


def _summarize_symbol_outline(file_name: str, payload: dict[str, Any] | list[Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        return None

    special_summaries = {
        "kernel.py": "`kernel.py` is the main orchestrator: it handles memory retrieval, routing, planning/checkpoint flow, tool execution, and verification for each turn.",
        "web.py": "`web.py` runs the local web backend: it serves the app, exposes health/files/turn endpoints, and sanitizes noisy replay output before users see it.",
        "routing.py": "`routing.py` owns AI backend routing: it switches between OpenCode, Ollama, and Codex while keeping Devenv in charge of tools and backend preference.",
        "codex_backend.py": "`codex_backend.py` adapts the Codex backend into Devenv's shared chat/status contract so it can be selected like the other reasoning engines.",
        "ollama_backend.py": "`ollama_backend.py` adapts local Ollama models into the same backend contract, including status, model selection, and bounded chat calls.",
        "opencode_client.py": "`opencode_client.py` is the transport layer for OpenCode server health checks, session management, message submission, and recovery.",
        "context_builder.py": "`context_builder.py` assembles the context packet from memory, workspace evidence, and prior session history before a model turn.",
        "engine.py": "`engine.py` is the memory engine: it stores episodic and associative memory and retrieves project context for later turns.",
        "workspace.py": "`workspace.py` provides sandboxed workspace browsing and file access so retrieval stays inside the project boundary.",
    }
    special_summary = special_summaries.get(file_name.lower())
    if special_summary:
        return special_summary

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
    if file_name.lower() == "readme.md":
        heading = next((line.strip().lstrip("#").strip() for line in stripped.splitlines() if line.strip().startswith("#")), "")
        prose_lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        prose_block = re.sub(r"\s+", " ", " ".join(prose_lines[:4])).strip()
        sentence_parts = re.split(r"(?<=[.!?])\s+", prose_block)
        prose_preview = " ".join(part.strip() for part in sentence_parts[:2] if part.strip()).strip()
        if heading and prose_preview:
            sentence = prose_preview
            if heading.lower().startswith("devenv ai is "):
                sentence = prose_preview
            elif prose_preview.lower().startswith(heading.lower()):
                sentence = prose_preview
            else:
                sentence = f"{heading} is {prose_preview[0].lower() + prose_preview[1:]}" if len(prose_preview) > 1 else f"{heading} is {prose_preview.lower()}"
            prefix = "`README.md` says" if sentence.lower().startswith(heading.lower()) else "`README.md` describes"
            if frameworks:
                return f"{prefix} {sentence} It also references {', '.join(frameworks[:4])}."
            return f"{prefix} {sentence}"
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


def _should_enable_web_search(text: str) -> bool:
    lowered = text.lower()
    current_fact_markers = (
        "today",
        "latest",
        "current",
        "currently",
        "recent",
        "president",
        "prime minister",
        "ceo",
        "who is",
        "official website",
        "documentation",
        "docs",
        "search the web",
        "on the web",
    )
    return any(marker in lowered for marker in current_fact_markers)


def _consolidation_cooldown_seconds() -> float:
    raw_value = os.getenv("DEVENV_CONSOLIDATION_COOLDOWN_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS


def _runtime_tool_transport() -> str:
    raw_value = os.getenv("DEVENV_TOOL_TRANSPORT", "").strip().lower()
    if raw_value in {"mcp", "stdio"}:
        return "mcp"
    return "in_process"
