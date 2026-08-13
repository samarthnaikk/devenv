class KernelLocalRuntimeMixin:
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
                    shaped_response = (
                        best_response
                        if best_response.lower().startswith("based on the codebase, here's")
                        else _shape_logged_answer_for_prompt(user_prompt, best_response)
                    )
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
        scaffold_kind = _local_scaffold_kind(user_prompt)
        checkpoint_limit = 1 if scaffold_kind == "calendar" else len(checkpoint_indexes)
        final_response: str | None = None

        for index in checkpoint_indexes[:checkpoint_limit]:
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
            checkpoint_requires_mutation = task.expects_mutation
            if local_tool_call is not None:
                ai_logs.append(f"Local-only tool requested: {local_tool_call.tool_name}")
                step = self._execute_tool_call(local_tool_call)
                steps.append(step)
                system_logs.append(f"Tool step {len(steps)}: {local_tool_call.tool_name} success={step.success}")
                if not step.success:
                    raise RuntimeError(step.output)
                context_only_completion = self._complete_context_only_checkpoint_from_steps(
                    user_prompt=user_prompt,
                    task_description=task.description,
                    candidate_steps=[step],
                )
                if context_only_completion is not None:
                    final_response = context_only_completion
                    trace_log = _summarize_execution_note(final_response)
                    working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                    self.active_blueprint = working_blueprint
                    ai_logs.append(f"Checkpoint completed locally after successful inspection: {task.description}")
                    system_logs.append(f"Checkpoint {index + 1} completed after successful inspection")
                    continue
            elif checkpoint_requires_mutation:
                raise RuntimeError(f"Local-only execution cannot perform mutation checkpoint: {task.description}")

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
        tool_policy_events: list[ToolPolicyEvent],
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
                    planning_allowed_tools = self._planning_allowed_tool_names()
                    if tool_call.tool_name not in planning_allowed_tools:
                        logger.warning("Blocked non-planning tool during planning: tool=%s", tool_call.tool_name)
                        system_logs.append(f"Blocked planning tool call: {tool_call.tool_name}")
                        denied_event = build_tool_policy_event(
                            tool_call.tool_name,
                            ExecutionMode.PLAN_ONLY,
                            "deny",
                            "Tool is not allowed in planning mode.",
                        )
                        tool_policy_events.append(denied_event)
                        ai_logs.append(f"Planning tool denied by policy: {denied_event.to_dict()}")
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

    def _should_use_deterministic_execution_tool(
        self,
        *,
        user_prompt: str,
        task: CheckpointTask,
    ) -> bool:
        if self._is_scaffold_request(f"{user_prompt} {task.description}".lower()):
            return True
        integration_root = self._local_integration_root_for_prompt(user_prompt)
        if not integration_root:
            return False
        mentioned_path = _first_backticked_path(task.description)
        if mentioned_path and (
            mentioned_path.startswith(f"{integration_root}/")
            or mentioned_path in {
                "core/runtime/web.py",
                "core/ai/routing.py",
                "interface/website/src/api.js",
                "interface/website/src/App.js",
                "interface/website/src/components/Composer.js",
            }
        ):
            return True
        lowered = task.description.lower()
        return lowered.startswith("inspect ") and integration_root in lowered

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
        # Execute every remaining checkpoint in one bounded turn, then verify the completed plan.
        checkpoint_indexes = self._execution_checkpoint_indexes(working_blueprint)

        for index in checkpoint_indexes:
            task = working_blueprint.tasks[index]
            working_blueprint = _set_active_task(working_blueprint, index)
            self.active_blueprint = working_blueprint
            checkpoint_step_start = len(steps)
            system_logs.append(f"Current checkpoint {index + 1}/{len(working_blueprint.tasks)}: {task.description}")
            scoped_tool_names = self._resolve_execution_tool_scope(
                user_prompt,
                task.description,
                checkpoint=task,
                selected_tools=selected_tools,
            )
            execution_memory = self._resolve_execution_memory(
                user_prompt=user_prompt,
                task_description=task.description,
                memory_context=memory_context,
            )
            system_logs.append(f"Execution memory chars sent: {len(execution_memory)}")
            system_logs.append(f"Execution tool scope size: {len(scoped_tool_names)}")
            deterministic_tool_call = None
            if self._should_use_deterministic_execution_tool(
                user_prompt=user_prompt,
                task=task,
            ):
                deterministic_tool_call = self._build_local_execution_tool_call(
                    user_prompt,
                    task.description,
                )
                if deterministic_tool_call is not None:
                    ai_logs.append(
                        f"Deterministic execution tool prepared for checkpoint {index + 1}: {deterministic_tool_call.tool_name}"
                    )
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
            workspace_hints = self._workspace_file_hints_from_steps(steps, user_prompt=user_prompt, task_description=task.description)
            if workspace_hints:
                step_conversation.append(
                    {
                        "role": "system",
                        "content": (
                            "Known workspace files from recent inspection: "
                            f"{', '.join(workspace_hints)}. Prefer these real paths over placeholder paths."
                        ),
                    }
                )
            tool_iterations = 0
            checkpoint_requires_mutation = self._checkpoint_requires_mutation(user_prompt, task.description) and any(
                tool_name in scoped_tool_names for tool_name in (*WRITE_EXECUTION_TOOLS, *DELETE_EXECUTION_TOOLS)
            )

            while True:
                checkpoint_has_successful_mutation = any(
                    step.success and step.tool_name in (*WRITE_EXECUTION_TOOLS, *DELETE_EXECUTION_TOOLS)
                    for step in steps[checkpoint_step_start:]
                )
                checkpoint_has_failed_step = any(not step.success for step in steps[checkpoint_step_start:])
                if deterministic_tool_call is not None:
                    tool_iterations += 1
                    if tool_iterations > max_consecutive_tools:
                        raise RuntimeError("Execution tool limit reached before the checkpoint completed.")
                    self._mark_local_backend_response()
                    ai_logs.append(
                        f"Deterministic tool requested: checkpoint={index + 1} tool={deterministic_tool_call.tool_name}"
                    )
                    step = self._execute_tool_call(deterministic_tool_call)
                    steps.append(step)
                    system_logs.append(
                        f"Tool step {len(steps)}: {deterministic_tool_call.tool_name} success={step.success} deterministic=true"
                    )
                    if not step.success:
                        raise RuntimeError(step.output)
                    context_only_completion = self._complete_context_only_checkpoint_from_steps(
                        user_prompt=user_prompt,
                        task_description=task.description,
                        candidate_steps=[step],
                    )
                    if context_only_completion is not None:
                        final_response = context_only_completion
                    else:
                        final_response = self._build_local_checkpoint_response(
                            user_prompt,
                            task.description,
                            execution_memory,
                        )
                    trace_log = _summarize_execution_note(final_response)
                    working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                    self.active_blueprint = working_blueprint
                    ai_logs.append(f"Checkpoint completed deterministically: {task.description}")
                    system_logs.append(
                        f"Checkpoint {index + 1} completed with deterministic workspace execution"
                    )
                    break
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
                    context_only_completion = self._complete_context_only_checkpoint_from_steps(
                        user_prompt=user_prompt,
                        task_description=task.description,
                        candidate_steps=converted_steps,
                    )
                    if context_only_completion is not None:
                        final_response = context_only_completion
                        trace_log = _summarize_execution_note(final_response)
                        working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                        self.active_blueprint = working_blueprint
                        ai_logs.append(f"Checkpoint completed after successful inspection: {task.description}")
                        system_logs.append(f"Checkpoint {index + 1} completed after successful inspection")
                        break
                    if checkpoint_requires_mutation and not any(
                        step.success and step.tool_name in (*WRITE_EXECUTION_TOOLS, *DELETE_EXECUTION_TOOLS)
                        for step in converted_steps
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
                    normalized_arguments = self.sandbox.normalize_arguments(self._repair_tool_arguments(tool_call))
                    if (
                        self._checkpoint_is_context_only(user_prompt, task.description)
                        and steps
                        and steps[-1].success
                        and steps[-1].tool_name == tool_call.tool_name == "list_directory"
                        and steps[-1].arguments == normalized_arguments
                    ):
                        final_response = _summarize_directory_listing(
                            str(normalized_arguments.get("path") or self.workspace_path),
                            steps[-1].output,
                        )
                        trace_log = _summarize_execution_note(final_response)
                        working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                        self.active_blueprint = working_blueprint
                        ai_logs.append(f"Checkpoint completed after redundant inspection was detected: {task.description}")
                        system_logs.append(f"Checkpoint {index + 1} completed after duplicate list_directory call")
                        break
                    if (
                        checkpoint_requires_mutation
                        and steps
                        and steps[-1].success
                        and steps[-1].tool_name == tool_call.tool_name == "write_file"
                    ):
                        previous_path = str(steps[-1].arguments.get("path") or "")
                        next_path = str(normalized_arguments.get("path") or "")
                        if previous_path and previous_path == next_path:
                            final_response = f"Updated `{next_path}` in the workspace."
                            trace_log = _summarize_execution_note(final_response)
                            working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                            self.active_blueprint = working_blueprint
                            ai_logs.append(f"Checkpoint completed after redundant write was detected: {task.description}")
                            system_logs.append(f"Checkpoint {index + 1} completed after duplicate write_file call")
                            break
                    reusable_step = _find_reusable_tool_step(steps, tool_call.tool_name, normalized_arguments)
                    if reusable_step is not None:
                        ai_logs.append(f"Reused prior tool result: {tool_call.tool_name}")
                        step_conversation.append(_assistant_tool_call_message(ai_response, [tool_call]))
                        steps.append(
                            ToolExecutionStep(
                                step_id=tool_call.call_id,
                                tool_name=tool_call.tool_name,
                                arguments=normalized_arguments,
                                output=reusable_step.output,
                                success=reusable_step.success,
                                is_sandboxed_violation=False,
                                data=dict(reusable_step.data or {}),
                            )
                        )
                        system_logs.append(f"Tool step {len(steps)}: {tool_call.tool_name} success={reusable_step.success} reused=true")
                        step_conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, reusable_step.output))
                        continue
                    ai_logs.append(f"Tool requested: {tool_call.tool_name}")
                    step_conversation.append(_assistant_tool_call_message(ai_response, [tool_call]))
                    step = self._execute_tool_call(tool_call)
                    steps.append(step)
                    system_logs.append(f"Tool step {len(steps)}: {tool_call.tool_name} success={step.success}")
                    context_only_completion = self._complete_context_only_checkpoint_from_steps(
                        user_prompt=user_prompt,
                        task_description=task.description,
                        candidate_steps=[step],
                    )
                    if context_only_completion is not None:
                        final_response = context_only_completion
                        trace_log = _summarize_execution_note(final_response)
                        working_blueprint = _mark_checkpoint_completed(working_blueprint, index, trace_log)
                        self.active_blueprint = working_blueprint
                        ai_logs.append(f"Checkpoint completed after successful inspection: {task.description}")
                        system_logs.append(f"Checkpoint {index + 1} completed after successful inspection")
                        break
                    step_conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                    continue

                if checkpoint_requires_mutation and not checkpoint_has_successful_mutation and (
                    tool_iterations == 0 or checkpoint_has_failed_step
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
                    if tool_iterations == 0:
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

    def _complete_context_only_checkpoint_from_steps(
        self,
        *,
        user_prompt: str,
        task_description: str,
        candidate_steps: list[ToolExecutionStep],
    ) -> str | None:
        if not self._checkpoint_is_context_only(user_prompt, task_description):
            return None
        successful_steps = [step for step in candidate_steps if step.success]
        if not successful_steps:
            return None
        last_step = successful_steps[-1]
        if last_step.tool_name == "list_directory":
            target_path = str(last_step.arguments.get("path") or self.workspace_path)
            return _summarize_directory_listing(target_path, last_step.output)
        if last_step.tool_name == "read_file":
            path_value = str(last_step.arguments.get("path") or "")
            payload = dict(last_step.data or {})
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                focused_answer = _extract_context_only_file_answer(
                    user_prompt=user_prompt,
                    task_description=task_description,
                    file_name=Path(path_value).name,
                    content=content,
                )
                if focused_answer:
                    return focused_answer
                summary = _summarize_local_text_file(Path(path_value).name, content)
                if summary:
                    return summary
                return f"Inspected `{Path(path_value).name}` successfully."
        return None

    def _workspace_file_hints_from_steps(
        self,
        steps: list[ToolExecutionStep],
        *,
        user_prompt: str,
        task_description: str,
    ) -> list[str]:
        for step in reversed(steps):
            if not step.success or step.tool_name != "list_directory":
                continue
            payload = _extract_tool_payload_json(step.output)
            entries = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                continue
            paths: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("is_dir"):
                    continue
                relative_path = entry.get("relative_path")
                if isinstance(relative_path, str) and relative_path.strip():
                    paths.append(relative_path.strip())
            if not paths:
                continue
            scored: list[tuple[int, str]] = []
            lowered = f"{user_prompt} {task_description}".lower()
            for path in paths:
                score = 0
                lowered_path = path.lower()
                if "theme" in lowered or "ui" in lowered:
                    if lowered_path.endswith("styles.css"):
                        score += 10
                    if lowered_path.endswith("index.html"):
                        score += 8
                    if lowered_path.endswith("script.js"):
                        score += 6
                if lowered_path.endswith(("styles.css", "index.html", "script.js", "main.py")):
                    score += 4
                scored.append((score, path))
            scored.sort(key=lambda item: (-item[0], item[1]))
            ordered = [path for _score, path in scored]
            return list(dict.fromkeys(ordered[:6]))
        return []

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
                    allowed_tool_names=task.allowed_tool_names,
                    expects_mutation=task.expects_mutation,
                    requires_verification=task.requires_verification,
                    repair_attempt_count=task.repair_attempt_count,
                    max_repair_attempts=task.max_repair_attempts,
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
        checkpoint: CheckpointTask | None = None,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[str]:
        combined_prompt = f"{user_prompt}\n{task_description}"
        scoped = self._tool_scope_for_prompt(
            combined_prompt,
            selected_tools=selected_tools,
            execution_phase=True,
        )
        if checkpoint is None and self._is_scaffold_request(combined_prompt.lower()):
            scaffold_only = [tool_name for tool_name in scoped if tool_name in SCAFFOLD_EXECUTION_TOOLS]
            if scaffold_only:
                return sorted(scaffold_only)
        if checkpoint is None or not checkpoint.allowed_tool_names:
            return scoped
        checkpoint_scope = {tool_name for tool_name in checkpoint.allowed_tool_names if tool_name in self.tools}
        if not checkpoint_scope:
            return scoped
        narrowed = [tool_name for tool_name in scoped if tool_name in checkpoint_scope]
        resolved = narrowed or sorted(checkpoint_scope)
        explicit_path = _first_backticked_path(task_description)
        if explicit_path and Path(explicit_path).suffix and self._checkpoint_is_context_only(user_prompt, task_description):
            resolved = [tool_name for tool_name in resolved if tool_name != "list_directory"]
            if "read_file" in self.tools and "read_file" not in resolved:
                resolved.append("read_file")
            if explicit_path.endswith(".py") and "inspect_symbols" in self.tools and "inspect_symbols" not in resolved:
                resolved.append("inspect_symbols")
        return resolved

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
        active_task = blueprint.tasks[checkpoint_index - 1] if 0 <= checkpoint_index - 1 < len(blueprint.tasks) else None
        target_path_hint = (active_task.target_path_hint if active_task else None) or self._derive_scaffold_target_path(
            user_prompt, task_description
        )
        plan_context = self._build_checkpoint_context(blueprint, checkpoint_index - 1)
        expects_mutation = active_task.expects_mutation if active_task is not None else self._checkpoint_requires_mutation(
            user_prompt, task_description
        )
        if self._is_scaffold_request(f"{user_prompt} {task_description}".lower()) or (active_task and active_task.expected_artifact == "frontend"):
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
        lines = [
            f"Original request:\n{user_prompt}\n\n"
            f"Current checkpoint ({checkpoint_index}/{total_checkpoints}):\n- [ ] {task_description}\n\n"
            f"Workspace root: {self.workspace_path}\n"
            "Use real workspace paths discovered from tools. Do not invent /workspace or external paths."
        ]
        if target_path_hint:
            lines.append(f"All new files for this request must stay under: {target_path_hint}")
        if active_task and active_task.allowed_tool_names:
            lines.append(
                "Allowed tools for this checkpoint: "
                + ", ".join(f"`{tool_name}`" for tool_name in active_task.allowed_tool_names)
            )
        mentioned_path = _first_backticked_path(task_description)
        if mentioned_path and Path(mentioned_path).suffix:
            lines.append(f"This checkpoint names a file directly: inspect `{mentioned_path}` with `read_file` or `inspect_symbols`, not `list_directory`.")
        if expects_mutation:
            lines.append("This checkpoint expects a real workspace mutation before it can be considered complete.")
        if active_task and active_task.requires_verification:
            lines.append(f"Verification will run after completion using mode: {active_task.verification_mode}.")
        if plan_context:
            lines.append(plan_context)
        lines.append("Complete only this checkpoint, then stop.")
        return "\n".join(lines)

    def _is_scaffold_request(self, text: str) -> bool:
        creation_markers = ("create", "make", "add", "build", "generate")
        frontend_markers = ("html", "css", "js", "javascript", "frontend", "ui")
        non_backend_markers = ("dont connect with backend", "don't connect with backend", "no need to connect to backend")
        backend_markers = (
            "backend",
            "api",
            "server",
            "route",
            "routes",
            "endpoint",
            "controller",
            "service",
            "database",
            "schema",
            "model",
            "auth",
            "integrate with frontend",
            "frontend integration",
        )
        file_markers = ("folder", "file")
        app_markers = ("app", "website", "page", "dashboard")
        if any(marker in text for marker in backend_markers) and not any(marker in text for marker in non_backend_markers):
            return False
        scaffold_match = (
            any(marker in text for marker in creation_markers)
            and any(marker in text for marker in frontend_markers)
            and (
                any(marker in text for marker in file_markers)
                or (
                    any(marker in text for marker in app_markers)
                    and _local_scaffold_kind(text) in {"notes", "todo", "weather", "date", "kanban"}
                )
            )
        ) or any(marker in text for marker in non_backend_markers)
        if scaffold_match:
            return True
        edit_markers = ("update", "modify", "change", "theme", "restyle", "light mode", "light theme", "dark mode", "dark theme")
        if any(marker in text for marker in edit_markers):
            return False
        return False

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
            "apply",
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

    def _should_plan(
        self,
        user_prompt: str,
        planning_mode: PlanningMode,
        *,
        selected_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> bool:
        if planning_mode is PlanningMode.FORCE_PLAN:
            return True
        if planning_mode is PlanningMode.FORCE_DIRECT:
            return False
        selected_scope = self._resolve_selected_tools(selected_tools)
        if not selected_scope:
            selected_scope = {
                tool_name.strip()
                for tool_name in (selected_tools or ())
                if isinstance(tool_name, str) and tool_name.strip()
            }
        lowered = user_prompt.lower()
        if selected_scope == {"knowledge_search"} and any(
            marker in lowered for marker in ("reference", "references", "repo", "repos", "example", "examples", "video", "videos", "tutorial")
        ):
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

    def _is_explicit_plan_request(self, user_prompt: str) -> bool:
        text = user_prompt.lower().strip()
        plan_markers = (
            "plan ",
            "plan:",
            "make a plan",
            "give me a plan",
            "show me a plan",
            "create a plan",
            "outline ",
            "roadmap ",
            "execution plan",
        )
        return any(text.startswith(marker) or marker in text for marker in plan_markers)

    def _execution_checkpoint_indexes(self, blueprint: ExecutionBlueprint) -> list[int]:
        indexes: list[int] = []
        for index, task in enumerate(blueprint.tasks):
            if task.is_completed:
                continue
            if task.description.strip().lower().startswith("verify "):
                continue
            indexes.append(index)
        return indexes

    def _should_update_active_plan_from_follow_up(self, user_prompt: str, planning_mode: PlanningMode) -> bool:
        if planning_mode is PlanningMode.FORCE_DIRECT:
            return False
        if self.active_blueprint is None or _next_incomplete_task_index(self.active_blueprint) is None:
            return False
        if self._is_plan_continue_request(user_prompt) or self._is_plan_exit_request(user_prompt):
            return False
        if not self.active_plan_prompt:
            return False
        lowered = user_prompt.lower()
        if not self._text_requires_mutation_tools(lowered):
            return False
        active_tokens = set(_prompt_keywords(self.active_plan_prompt))
        current_tokens = set(_prompt_keywords(user_prompt))
        if not active_tokens or not current_tokens:
            return False
        overlap = len(active_tokens & current_tokens) / max(min(len(active_tokens), len(current_tokens)), 1)
        follow_up_markers = ("ok", "okay", "now", "next", "also", "instead", "integrate", "add", "update", "implement", "wire")
        return overlap >= 0.3 or any(marker in lowered for marker in follow_up_markers)

    def _checkpoint_is_context_only(self, user_prompt: str, task_description: str) -> bool:
        lowered = task_description.lower().lstrip("- *")
        return lowered.startswith(("inspect ", "gather ", "identify ", "review ", "read ", "list ", "analyze ", "explore "))

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
            "apply",
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
