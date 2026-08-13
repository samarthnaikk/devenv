class KernelLifecycleMixin:
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
        self._runtime_state_path = str(Path(self.workspace_path) / ".devenv-runtime-state.json")
        self._load_runtime_state()

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
        self.reset_active_plan()
        self.state = AgentState.PLANNING
        self.session_usage_totals = {}
        self.session_id = str(uuid.uuid4())
        if hasattr(self.ai, "reset_session"):
            self.ai.reset_session()
        return self.session_id

    def reset_active_plan(self) -> bool:
        had_active_plan = self.active_blueprint is not None or bool(self.active_plan_prompt)
        self.active_blueprint = None
        self.active_plan_prompt = None
        self.state = AgentState.PLANNING
        self._save_runtime_state()
        return had_active_plan

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
        max_consecutive_tools: int = DEFAULT_MAX_CONSECUTIVE_TOOLS,
        planning_mode: PlanningMode = PlanningMode.AUTO,
        continue_plan: bool = False,
        local_only: bool = False,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
        backend_preference: str = "opencode",
        opencode_enabled: bool = False,
        ollama_enabled: bool = False,
        llama_cpp_enabled: bool = False,
        codex_enabled: bool = False,
        session_budget_tokens: int | None = None,
        no_memory: bool = False,
        incognito: bool = False,
    ) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        max_consecutive_tools = self._effective_max_consecutive_tools(
            requested_limit=max_consecutive_tools,
            local_only=local_only,
        )
        turn_started_at = time.perf_counter()
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        stage_traces: list[StageTrace] = []
        verification_results: list[VerificationResult] = []
        tool_policy_events = self._selected_tool_policy_events(selected_tools)
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
                llama_cpp_enabled=llama_cpp_enabled,
                codex_enabled=codex_enabled,
            )
        if session_budget_tokens is not None and self.session_usage_totals.get("total_tokens", 0) >= session_budget_tokens:
            turn_metadata["budget_state"] = {
                "blocked": True,
                "limit": session_budget_tokens,
                "used": self.session_usage_totals.get("total_tokens", 0),
                "remaining": 0,
            }
            return self._make_turn_result(
                final_response=None,
                steps=[],
                total_usage=dict(self.session_usage_totals),
                ai_logs=ai_logs,
                system_logs=system_logs + ["Session token budget reached before starting a new turn."],
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                error_message="Session token budget reached. Increase the budget to continue.",
                execution_mode=ExecutionMode.BLOCKED_FOR_CLARIFICATION.value,
                turn_outcome=TurnOutcome.BUDGET_STOP.value,
                memory_context="",
                started_at=turn_started_at,
                tool_policy_events=tool_policy_events,
            )
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        if _is_brief_greeting_prompt(user_prompt):
            fast_response = "Hi. What would you like me to recall or inspect?"
            self._mark_local_backend_response()
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
            return self._make_turn_result(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context="",
                started_at=turn_started_at,
                execution_mode=ExecutionMode.DIRECT_ANSWER.value,
            )

        conversation_follow_up = _answer_from_recent_conversation_follow_up(user_prompt, self.ephemeral_history)
        if conversation_follow_up is not None:
            self._mark_local_backend_response()
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
            return self._make_turn_result(
                final_response=conversation_follow_up,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context="",
                started_at=turn_started_at,
                execution_mode=ExecutionMode.DIRECT_ANSWER.value,
            )

        tool_strategy_response = self._answer_tool_strategy_question(user_prompt, selected_tools=turn_metadata["selected_tools"])
        if tool_strategy_response is not None:
            self._mark_local_backend_response()
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
            return self._make_turn_result(
                final_response=tool_strategy_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context="",
                started_at=turn_started_at,
                execution_mode=ExecutionMode.DIRECT_ANSWER.value,
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
            return self._make_turn_result(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context="",
                started_at=turn_started_at,
                execution_mode=ExecutionMode.BLOCKED_FOR_CLARIFICATION.value,
                turn_outcome=TurnOutcome.BLOCKED_BY_CLARIFICATION.value,
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
            return self._make_turn_result(
                final_response=fast_response,
                steps=[],
                total_usage={},
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context="",
                started_at=turn_started_at,
                execution_mode=ExecutionMode.BLOCKED_FOR_CLARIFICATION.value,
                turn_outcome=TurnOutcome.BLOCKED_BY_CLARIFICATION.value,
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
                return self._make_turn_result(
                    final_response=fast_response,
                    steps=[],
                    total_usage={},
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    memory_context="",
                    started_at=turn_started_at,
                    execution_mode=ExecutionMode.DIRECT_ANSWER.value,
                )

        if not incognito:
            self._record_working_memory(conversation)
        live_search_request = _is_explicit_live_search_prompt(user_prompt)
        if no_memory or incognito:
            memory_context, retrieval_metadata = "", dict(PRIVACY_DISABLED_METADATA)
        elif live_search_request:
            memory_context, retrieval_metadata = "", {
                "external_context_state": "live_search_only",
                "external_context_reason": "Live fact request routed directly to web search without prior project memory.",
                "external_context_session_count": 0,
                "external_context_session_ids": [],
            }
            system_logs.append("Skipped project memory for explicit live-search request")
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
        for event in tool_policy_events:
            if event.decision == "deny":
                system_logs.append(f"Selected tool denied: {event.tool_name} ({event.reason})")
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
            return self._make_turn_result(
                final_response=direct_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context=memory_context,
                started_at=turn_started_at,
                execution_mode=ExecutionMode.DIRECT_ANSWER.value,
                tool_policy_events=tool_policy_events,
            )
        if _should_try_direct_memory_answer(user_prompt):
            direct_memory_answer = self._answer_known_project_question_local(user_prompt, memory_context)
            if direct_memory_answer is None:
                direct_memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
            if direct_memory_answer is None and not _should_skip_exact_logged_fast_path(user_prompt):
                direct_memory_answer = self._lookup_exact_logged_answer(user_prompt)
            if direct_memory_answer is not None:
                self._mark_local_backend_response()
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
                return self._make_turn_result(
                    final_response=direct_memory_answer,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    memory_context=memory_context,
                    started_at=turn_started_at,
                    execution_mode=ExecutionMode.DIRECT_ANSWER.value,
                    tool_policy_events=tool_policy_events,
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
                return self._make_turn_result(
                    final_response=fallback_response,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    memory_context=memory_context,
                    started_at=turn_started_at,
                    execution_mode=ExecutionMode.DIRECT_ANSWER.value,
                    tool_policy_events=tool_policy_events,
                )
        blueprint, planning_conversation, creation_trace = self._checkpoint_creation_stage(
            user_prompt=user_prompt,
            memory_context=memory_context,
            continue_plan=continue_plan,
            local_only=local_only,
            planning_mode=planning_mode,
            selected_tools=turn_metadata["selected_tools"],
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            max_consecutive_tools=max_consecutive_tools,
            tool_policy_events=tool_policy_events,
        )
        stage_traces.append(creation_trace)
        self.active_blueprint = blueprint
        execution_objective = blueprint.original_objective or user_prompt
        self.active_plan_prompt = execution_objective
        turn_metadata["original_objective"] = execution_objective

        if planning_mode is PlanningMode.AUTO and self._is_explicit_plan_request(user_prompt):
            if _is_generic_explicit_plan_blueprint(blueprint):
                blueprint = self._parse_markdown_to_blueprint(
                    self._build_local_plan_markdown(user_prompt),
                    original_objective=user_prompt,
                )
                self.active_blueprint = blueprint
            final_plan_response = _blueprint_markdown_for_chat(blueprint)
            conversation.append({"role": "assistant", "content": final_plan_response})
            self._finalize_turn(
                user_prompt,
                final_plan_response,
                conversation,
                metadata=turn_metadata,
                persist_memory=(not incognito) and _should_persist_episodic_response(final_plan_response),
                persist_working_memory=not incognito,
            )
            system_logs.append("Explicit planning request returned blueprint without executing checkpoints")
            return self._make_turn_result(
                final_response=final_plan_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context=memory_context,
                started_at=turn_started_at,
                execution_mode=ExecutionMode.PLAN_ONLY.value,
                tool_policy_events=tool_policy_events,
            )

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
            return self._make_turn_result(
                final_response="Nothing left to execute.",
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                memory_context=memory_context,
                started_at=turn_started_at,
                execution_mode=ExecutionMode.PLAN_ONLY.value,
                tool_policy_events=tool_policy_events,
            )

        checkpoint = blueprint.tasks[active_index]
        self.active_blueprint = _set_active_task(blueprint, active_index)
        context_packet, context_trace = build_context_packet(
            checkpoint_id=checkpoint.task_id,
            checkpoint_objective=checkpoint.objective or checkpoint.description,
            memory_context=memory_context,
            local_model=self.local_small_model,
            tool_names=self._resolve_execution_tool_scope(
                execution_objective,
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
                user_prompt=execution_objective,
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
                memory_context=memory_context,
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
                return self._make_turn_result(
                    final_response=degraded_response,
                    steps=steps,
                    total_usage=total_usage,
                    ai_logs=ai_logs,
                    system_logs=system_logs,
                    stage_traces=stage_traces,
                    verification_results=verification_results,
                    metadata=turn_metadata,
                    error_message=str(exc),
                    execution_mode=self._execution_mode_value(),
                    turn_outcome=TurnOutcome.TOOL_FAILURE.value,
                    memory_context=memory_context,
                    started_at=turn_started_at,
                    tool_policy_events=tool_policy_events,
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
            return self._make_turn_result(
                final_response=None,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                error_message=str(exc),
                execution_mode=self._execution_mode_value(),
                turn_outcome=TurnOutcome.TOOL_FAILURE.value,
                memory_context=memory_context,
                started_at=turn_started_at,
                tool_policy_events=tool_policy_events,
            )

        self.active_blueprint = updated_blueprint
        final_response = _enforce_exact_output_contract(user_prompt, final_response)
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
            original_objective=execution_objective,
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
                repair_block_reason = self._describe_repair_chain_block(self.active_blueprint, checkpoint.task_id)
                system_logs.append(f"Verification failed; stopped automatic repair chaining ({repair_block_reason})")
            self._finalize_turn(
                user_prompt,
                final_response or "",
                conversation,
                persist_memory=(not incognito) and _should_persist_episodic_response(final_response or ""),
                persist_working_memory=not incognito,
                metadata=turn_metadata,
            )
            return self._make_turn_result(
                final_response=final_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                stage_traces=stage_traces,
                verification_results=verification_results,
                metadata=turn_metadata,
                execution_mode=ExecutionMode.REPAIR.value if appended_repair else ExecutionMode.VERIFICATION.value,
                turn_outcome=TurnOutcome.VERIFICATION_FAILURE.value,
                memory_context=memory_context,
                started_at=turn_started_at,
                tool_policy_events=tool_policy_events,
            )

        if _next_incomplete_task_index(self.active_blueprint) is None:
            self.state = AgentState.VERIFYING
            self.active_blueprint = _mark_blueprint_verified(self.active_blueprint, True)
            self.active_plan_prompt = None
        else:
            self.state = AgentState.EXECUTING

        final_response = _prefer_reference_results_over_empty_summary(final_response, steps, user_prompt)
        final_response = _enforce_exact_output_contract(user_prompt, final_response)

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
        return self._make_turn_result(
            final_response=final_response,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            stage_traces=stage_traces,
            verification_results=verification_results,
            metadata=turn_metadata,
            execution_mode=self._execution_mode_value(),
            turn_outcome=TurnOutcome.SUCCESS.value,
            memory_context=memory_context,
            started_at=turn_started_at,
            tool_policy_events=tool_policy_events,
        )

    def _execution_mode_value(self) -> str:
        if self.state is AgentState.VERIFYING:
            return ExecutionMode.VERIFICATION.value
        if self.state is AgentState.EXECUTING:
            return ExecutionMode.CHECKPOINT_EXECUTE.value
        if self.active_blueprint is not None and any(
            task.repair_origin_checkpoint_id is not None and not task.is_completed for task in self.active_blueprint.tasks
        ):
            return ExecutionMode.REPAIR.value
        return ExecutionMode.PLAN_ONLY.value

    def _effective_max_consecutive_tools(self, *, requested_limit: int, local_only: bool) -> int:
        preferred_backend = str(getattr(self.ai, "preferred_backend", "") or "").strip().lower()
        if local_only or preferred_backend in {"ollama", "llama_cpp"}:
            return max(requested_limit, 64)
        return requested_limit

    def _build_memory_summary(self, memory_context: str, metadata: dict[str, Any]) -> MemorySummary:
        privacy_mode = "incognito" if metadata.get("incognito") else "no_memory" if metadata.get("no_memory") else "default"
        return MemorySummary(
            used_working_memory=not metadata.get("incognito", False),
            used_associative_memory=bool(memory_context.strip()),
            used_external_context=metadata.get("external_context_session_count", 0) > 0,
            privacy_mode=privacy_mode,
            context_chars=len(memory_context),
        )

    def _build_repair_state(self, blueprint: ExecutionBlueprint | None) -> RepairState:
        if blueprint is None:
            return RepairState()
        repair_task = next(
            (
                task
                for task in blueprint.tasks
                if task.repair_origin_checkpoint_id is not None and not task.is_completed
            ),
            None,
        )
        if repair_task is None:
            return RepairState()
        return RepairState(
            active_checkpoint_id=repair_task.task_id,
            repair_attempt_count=repair_task.repair_attempt_count,
            max_repair_attempts=repair_task.max_repair_attempts,
            last_failure_reason=repair_task.status_reason or repair_task.description,
        )

    def _make_turn_result(
        self,
        *,
        final_response: str | None,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        stage_traces: list[StageTrace],
        verification_results: list[VerificationResult],
        metadata: dict[str, Any],
        memory_context: str,
        started_at: float,
        tool_policy_events: list | None = None,
        error_message: str | None = None,
        execution_mode: str | None = None,
        turn_outcome: str = TurnOutcome.SUCCESS.value,
    ) -> RuntimeTurnResult:
        self._save_runtime_state()
        return RuntimeTurnResult(
            final_response=final_response,
            steps=steps,
            total_usage=total_usage,
            ai_logs=ai_logs,
            system_logs=system_logs,
            stage_traces=stage_traces,
            verification_results=verification_results,
            metadata=metadata,
            state=self.state.name,
            blueprint=self.active_blueprint,
            error_message=error_message,
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
            execution_mode=execution_mode or self._execution_mode_value(),
            turn_outcome=turn_outcome,
            memory_summary=self._build_memory_summary(memory_context, metadata),
            tool_policy_events=list(tool_policy_events or []),
            repair_state=self._build_repair_state(self.active_blueprint),
        )

    def _load_runtime_state(self) -> None:
        state_path = Path(self._runtime_state_path)
        if not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load runtime state: path=%s", state_path, exc_info=True)
            return
        if not isinstance(payload, dict):
            return
        blueprint_payload = payload.get("active_blueprint")
        if isinstance(blueprint_payload, dict):
            self.active_blueprint = self._blueprint_from_dict(blueprint_payload)
        active_plan_prompt = payload.get("active_plan_prompt")
        if isinstance(active_plan_prompt, str) and active_plan_prompt.strip():
            self.active_plan_prompt = active_plan_prompt
        state_name = payload.get("state")
        if isinstance(state_name, str):
            try:
                self.state = AgentState[state_name]
            except KeyError:
                logger.debug("Ignoring unknown persisted agent state: %s", state_name)

    def _save_runtime_state(self) -> None:
        state_path = Path(self._runtime_state_path)
        payload = {
            "active_blueprint": self.active_blueprint.to_dict() if self.active_blueprint is not None else None,
            "active_plan_prompt": self.active_plan_prompt,
            "state": self.state.name,
        }
        try:
            state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            logger.warning("Failed to save runtime state: path=%s", state_path, exc_info=True)

    def _blueprint_from_dict(self, payload: dict[str, Any]) -> ExecutionBlueprint:
        tasks_payload = payload.get("tasks")
        tasks: list[CheckpointTask] = []
        if isinstance(tasks_payload, list):
            for item in tasks_payload:
                if not isinstance(item, dict):
                    continue
                task_id = item.get("task_id")
                description = item.get("description")
                if not isinstance(task_id, int) or not isinstance(description, str):
                    continue
                child_ids = item.get("child_checkpoint_ids")
                child_checkpoint_ids = tuple(
                    child_id for child_id in child_ids if isinstance(child_id, int)
                ) if isinstance(child_ids, list) else ()
                allowed_tool_names_raw = item.get("allowed_tool_names")
                allowed_tool_names = tuple(
                    tool_name for tool_name in allowed_tool_names_raw if isinstance(tool_name, str)
                ) if isinstance(allowed_tool_names_raw, list) else ()
                tasks.append(
                    CheckpointTask(
                        task_id=task_id,
                        description=description,
                        objective=item.get("objective") if isinstance(item.get("objective"), str) else None,
                        target_path_hint=item.get("target_path_hint") if isinstance(item.get("target_path_hint"), str) else None,
                        expected_artifact=item.get("expected_artifact") if isinstance(item.get("expected_artifact"), str) else "chat",
                        verification_mode=item.get("verification_mode") if isinstance(item.get("verification_mode"), str) else "chat",
                        repair_origin_checkpoint_id=item.get("repair_origin_checkpoint_id") if isinstance(item.get("repair_origin_checkpoint_id"), int) else None,
                        status_reason=item.get("status_reason") if isinstance(item.get("status_reason"), str) else None,
                        output_destination=item.get("output_destination") if isinstance(item.get("output_destination"), str) else None,
                        child_checkpoint_ids=child_checkpoint_ids,
                        is_completed=bool(item.get("is_completed")),
                        execution_trace_log=item.get("execution_trace_log") if isinstance(item.get("execution_trace_log"), str) else None,
                        allowed_tool_names=allowed_tool_names,
                        expects_mutation=bool(item.get("expects_mutation")),
                        requires_verification=bool(item.get("requires_verification")),
                        repair_attempt_count=item.get("repair_attempt_count") if isinstance(item.get("repair_attempt_count"), int) else 0,
                        max_repair_attempts=item.get("max_repair_attempts") if isinstance(item.get("max_repair_attempts"), int) else 1,
                    )
                )
        active_task_pointer = payload.get("active_task_pointer")
        return ExecutionBlueprint(
            raw_plan_markdown=payload.get("raw_plan_markdown") if isinstance(payload.get("raw_plan_markdown"), str) else "",
            original_objective=payload.get("original_objective") if isinstance(payload.get("original_objective"), str) else None,
            tasks=tasks,
            active_task_pointer=active_task_pointer if isinstance(active_task_pointer, int) else 0,
            verification_passed=bool(payload.get("verification_passed")),
        )

    def _selected_tool_policy_events(
        self,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None,
    ) -> list:
        requested = [
            tool_name.strip()
            for tool_name in (selected_tools or ())
            if isinstance(tool_name, str) and tool_name.strip()
        ]
        if not requested:
            return []
        resolved = self._resolve_selected_tools(selected_tools)
        events = []
        for tool_name in requested:
            if tool_name in resolved:
                events.append(
                    build_tool_policy_event(
                        tool_name,
                        ExecutionMode.CHECKPOINT_EXECUTE,
                        "allow",
                        "User explicitly selected this tool and it is available.",
                    )
                )
            else:
                events.append(
                    build_tool_policy_event(
                        tool_name,
                        ExecutionMode.CHECKPOINT_EXECUTE,
                        "deny",
                        "Requested tool is not registered in the current runtime.",
                    )
                )
        return events

    def _build_tool_client(self, *, db_path: str, vector_dir: str):
        transport = _runtime_tool_transport()
        if transport == "in_process":
            logger.info("Using in-process tool client for runtime execution")
            return _InProcessToolClient(self.tools)
        try:
            from ..mcp_client import MCPToolClient

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
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        max_consecutive_tools: int,
        tool_policy_events: list[ToolPolicyEvent],
    ) -> tuple[ExecutionBlueprint, list[dict[str, Any]], StageTrace]:
        should_resume_plan = planning_mode is not PlanningMode.FORCE_DIRECT and (continue_plan or self._is_plan_continue_request(user_prompt))
        if should_resume_plan and self.active_blueprint is not None and _next_incomplete_task_index(self.active_blueprint) is None:
            trace = StageTrace(
                stage=ProcessStage.CHECKPOINT_CREATION.value,
                success=True,
                summary="Resumed completed checkpoint plan",
                logs=[f"Checkpoint count: {len(self.active_blueprint.tasks)}"],
                payload={"continued": True, "completed": True},
            )
            return self.active_blueprint, [], trace
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
        if self._should_update_active_plan_from_follow_up(user_prompt, planning_mode):
            blueprint = self._build_direct_blueprint(user_prompt)
            trace = StageTrace(
                stage=ProcessStage.CHECKPOINT_CREATION.value,
                success=True,
                summary="Updated active checkpoint plan from follow-up instruction",
                logs=[f"Checkpoint count: {len(blueprint.tasks)}"],
                payload={"continued": True, "updated_from_follow_up": True},
            )
            return blueprint, [], trace

        planning_conversation: list[dict[str, Any]] = []
        should_plan = self._should_plan(user_prompt, planning_mode, selected_tools=selected_tools)
        if should_plan:
            prefer_local_planning = (
                local_only
                or (
                    self._is_backend_frontend_integration_request(user_prompt)
                    and (self._remote_backend_enabled() or self._local_integration_root_for_prompt(user_prompt))
                )
                or (
                getattr(self.ai, "preferred_backend", "") in {"ollama", "llama_cpp"}
                and self._text_requires_mutation_tools(user_prompt.lower())
                )
            )
            if prefer_local_planning:
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
                    tool_policy_events=tool_policy_events,
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
        if self._is_backend_frontend_integration_request(user_prompt):
            plan = self._build_local_plan_markdown(user_prompt)
            blueprint = self._parse_markdown_to_blueprint(plan, original_objective=user_prompt)
            return replace(blueprint, verification_passed=False)
        task = self._build_checkpoint_task(task_id=1, description=user_prompt, original_objective=user_prompt)
        seeded_tasks = self._seed_direct_checkpoint_tasks(user_prompt, task)
        return ExecutionBlueprint(
            raw_plan_markdown="\n".join(f"- [ ] {seed_task.description}" for seed_task in seeded_tasks),
            original_objective=user_prompt,
            tasks=seeded_tasks,
            active_task_pointer=0,
            verification_passed=False,
        )

    def _seed_direct_checkpoint_tasks(self, user_prompt: str, task: CheckpointTask) -> list[CheckpointTask]:
        if task.expected_artifact == "chat":
            return [task]
        if not self._should_pre_split_direct_checkpoint(user_prompt, task):
            return [task]

        descriptions = self._decompose_checkpoint(task)
        tasks: list[CheckpointTask] = []
        for index, description in enumerate(descriptions, start=1):
            child = self._build_checkpoint_task(
                task_id=index,
                description=description,
                original_objective=user_prompt,
                repair_origin_checkpoint_id=task.repair_origin_checkpoint_id,
            )
            if description.lower().startswith("inspect "):
                child = replace(
                    child,
                    expects_mutation=False,
                    requires_verification=False,
                    verification_mode="chat",
                )
            tasks.append(child)
        return tasks or [task]

    def _build_checkpoint_task(self, *, task_id: int, description: str, original_objective: str, repair_origin_checkpoint_id: int | None = None) -> CheckpointTask:
        target_path_hint = self._derive_scaffold_target_path(original_objective, description)
        expected_artifact = self._infer_expected_artifact(original_objective, description, target_path_hint)
        verification_mode = self._infer_verification_mode(expected_artifact, original_objective, description)
        output_destination = self._infer_output_destination(expected_artifact)
        expects_mutation = expected_artifact in {"frontend", "code"}
        requires_verification = verification_mode != "chat"
        allowed_tool_names = tuple(
            self._resolve_execution_tool_scope(
                original_objective,
                description,
                selected_tools=None,
            )
        )
        if not allowed_tool_names:
            allowed_tool_names = tuple(
                allowed_tool_names_for_mode(
                    ExecutionMode.CHECKPOINT_EXECUTE,
                    set(TOOL_POLICY_REGISTRY),
                )
            )
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
            allowed_tool_names=allowed_tool_names,
            expects_mutation=expects_mutation,
            requires_verification=requires_verification,
            repair_attempt_count=1 if repair_origin_checkpoint_id is not None else 0,
            max_repair_attempts=2 if expects_mutation else 1,
        )

    def _infer_expected_artifact(self, user_prompt: str, task_description: str, target_path_hint: str | None) -> str:
        text = f"{user_prompt} {task_description}".lower()
        backend_markers = (
            "backend",
            "server",
            "api",
            "route",
            "routes",
            "endpoint",
            "service",
            "controller",
            "model",
            "database",
            "schema",
            "auth",
            "integration",
        )
        frontend_markers = ("html", "css", "javascript", "frontend", "ui")
        document_markers = ("pdf", "document", "report", "brief", "handout", "invoice")
        if any(token in text for token in backend_markers) and self._text_requires_mutation_tools(text):
            return "code"
        if any(token in text for token in ("html", "css", "javascript", "frontend")) and (
            self._text_requires_mutation_tools(text) or self._is_scaffold_request(text)
        ):
            return "frontend"
        if "pdf" in text or (
            any(token in text for token in document_markers)
            and any(token in text for token in ("create", "generate", "build", "make", "export"))
        ):
            return "document"
        if any(token in text for token in ("create", "write", "edit", "modify", "update", "fix", "implement")):
            return "code"
        return "chat"

    def _infer_verification_mode(self, expected_artifact: str, user_prompt: str, task_description: str) -> str:
        if expected_artifact == "frontend":
            return "frontend"
        if expected_artifact == "code":
            return "code"
        if expected_artifact == "document":
            return "file"
        text = f"{user_prompt} {task_description}".lower()
        if any(token in text for token in ("file", "folder", "remove", "delete")):
            return "file"
        return "chat"

    def _infer_output_destination(self, expected_artifact: str) -> str:
        if expected_artifact == "frontend":
            return "file_write"
        if expected_artifact == "code":
            return "file_edit"
        if expected_artifact == "document":
            return "artifact_write"
        return "chat"

    def _context_char_limit_for_checkpoint(self, checkpoint: CheckpointTask) -> int:
        if checkpoint.expected_artifact == "frontend":
            return SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT
        if checkpoint.expected_artifact == "chat":
            return DIRECT_MEMORY_CHAR_LIMIT
        return EXECUTION_MEMORY_CHAR_LIMIT
