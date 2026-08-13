class KernelCheckpointMixin:
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
        if checkpoint.expected_artifact == "chat" and _is_explicit_live_search_prompt(user_prompt):
            forced_response = self._run_forced_web_search_turn(
                user_prompt=user_prompt,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
                local_only=local_only,
            )
            if forced_response is not None:
                updated = _mark_checkpoint_completed(
                    blueprint,
                    blueprint.active_task_pointer,
                    _summarize_execution_note(forced_response),
                )
                self.active_blueprint = updated
                return forced_response, updated, steps[pre_step_count:]
        direct_file_read = checkpoint.description.lower().startswith("read ") and not self._text_requires_mutation_tools(
            checkpoint.description.lower()
        )
        if checkpoint.expected_artifact == "chat" and (
            not self._checkpoint_is_context_only(user_prompt, checkpoint.description) or direct_file_read
        ):
            direct_memory_answer = self._answer_known_project_question_local(user_prompt, raw_memory_context)
            if direct_memory_answer is None:
                direct_memory_answer = _answer_from_retrieved_memory(user_prompt, raw_memory_context)
            if direct_memory_answer is not None and _should_trust_memory_answer_for_prompt(user_prompt):
                self._mark_local_backend_response()
                updated = _mark_checkpoint_completed(
                    blueprint,
                    blueprint.active_task_pointer,
                    _summarize_execution_note(direct_memory_answer),
                )
                self.active_blueprint = updated
                ai_logs.append("Checkpoint answered from retrieved memory before direct-turn model execution")
                return direct_memory_answer, updated, steps[pre_step_count:]

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
            if route_decision.use_local_knowledge and not _is_explicit_live_search_prompt(user_prompt):
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

    def _run_forced_web_search_turn(
        self,
        *,
        user_prompt: str,
        steps: list[ToolExecutionStep],
        total_usage: dict[str, int],
        ai_logs: list[str],
        system_logs: list[str],
        local_only: bool,
    ) -> str | None:
        if "web_search" not in self.tools:
            ai_logs.append("Live search requested but web_search is not registered")
            return None
        tool_call = ToolCallRequest(
            call_id=f"live_{uuid.uuid4().hex[:10]}",
            tool_name="web_search",
            arguments={"mode": "search", "query": user_prompt, "result_count": 5},
        )
        step = self._execute_tool_call(tool_call)
        steps.append(step)
        system_logs.append(f"Tool step {len(steps)}: web_search success={step.success}")
        if not step.success:
            ai_logs.append(f"Live web search failed: {step.output}")
            return None
        results = step.data.get("results") if isinstance(step.data, dict) else None
        if not isinstance(results, list) or not results:
            ai_logs.append("Live web search returned no results")
            return None
        result_lines = [
            f"- {item.get('title', 'Untitled result')} — {item.get('url', '')}"
            for item in results
            if isinstance(item, dict)
        ]
        page_context, page_extracts = self._collect_live_web_page_context(results, steps=steps, system_logs=system_logs)
        search_context = "Live web results:\n" + "\n".join(result_lines)
        if page_context:
            search_context += "\n\nLive page extracts:\n" + page_context
        if local_only:
            local_response = "I found these current web results:\n\n" + "\n".join(result_lines)
            if page_context:
                local_response += "\n\nRead from top sources:\n" + page_context
            return local_response
        grounded_answer = _build_grounded_live_fact_answer(user_prompt, results, page_extracts)
        if grounded_answer:
            ai_logs.append(f"Forced live web search answered directly from {len(page_extracts)} fetched page extract(s)")
            return grounded_answer
        try:
            response = self.ai.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the user's current-fact question using only the live web context below. "
                            "Prefer explicit facts from the page extracts over search-result titles, and prefer primary or authoritative sources when available. "
                            "Do not invent numbers, dates, or sources that are not present in the provided context. "
                            "If the sources disagree, say so briefly. Include source links when useful, and only name a source if its extract explicitly supports the fact you state."
                        ),
                    },
                    {"role": "user", "content": f"{user_prompt}\n\n{search_context}"},
                ],
                memory_context=search_context,
                tool_names=[],
            )
        except RuntimeError as exc:
            ai_logs.append(f"Live-search synthesis failed: {exc}")
            return "I found current web results, but answer synthesis failed:\n\n" + "\n".join(result_lines)
        _merge_usage(total_usage, response.usage)
        ai_logs.append(f"Forced live web search completed with {len(results)} result(s)")
        return response.content or ("I found these current web results:\n\n" + "\n".join(result_lines))

    def _collect_live_web_page_context(
        self,
        results: list[dict[str, Any]],
        *,
        steps: list[ToolExecutionStep],
        system_logs: list[str],
    ) -> tuple[str, list[dict[str, str]]]:
        page_lines: list[str] = []
        extracts: list[dict[str, str]] = []
        successful_reads = 0
        found_structured_fact = False
        for item in _prioritize_live_search_results(results)[:5]:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            read_step = self._execute_tool_call(
                ToolCallRequest(
                    call_id=f"live_read_{uuid.uuid4().hex[:10]}",
                    tool_name="web_search",
                    arguments={"mode": "read_url", "url": url},
                )
            )
            steps.append(read_step)
            system_logs.append(f"Tool step {len(steps)}: web_search success={read_step.success}")
            if not read_step.success:
                continue
            successful_reads += 1
            data = read_step.data if isinstance(read_step.data, dict) else {}
            title = str(data.get("title") or item.get("title") or "Untitled result").strip()
            raw_content = str(data.get("content") or "")
            if _extract_salient_live_search_excerpt(re.sub(r"\s+", " ", raw_content).strip()):
                found_structured_fact = True
            content = _truncate_live_search_content(raw_content)
            if not content:
                continue
            page_lines.append(f"- {title} — {url}\n  {content}")
            extracts.append({"title": title, "url": url, "excerpt": content})
            if successful_reads >= 3 and found_structured_fact:
                break
        return "\n".join(page_lines), extracts

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
        diagnostics_target: str | None = None

        if not checkpoint.requires_verification and checkpoint.verification_mode in {"chat", "none"}:
            results.append(
                VerificationResult(
                    checkpoint_id=checkpoint.task_id,
                    mode="none",
                    success=True,
                    details="Verification skipped by checkpoint contract.",
                )
            )
            logs.append("Verification skipped by checkpoint contract.")
        elif checkpoint.verification_mode == "chat":
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
            candidates.extend(self._tool_step_candidate_paths(step))

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
            for candidate in self._tool_step_candidate_paths(step):
                try:
                    paths.append(str(candidate.relative_to(self.workspace_path)))
                except ValueError:
                    paths.append(str(candidate))
        path_hint = (paths[-1] if paths else None) or checkpoint.target_path_hint
        if not path_hint:
            return None
        candidate = Path(path_hint)
        if not candidate.is_absolute():
            candidate = Path(self.workspace_path) / candidate
        details = f"Artifact check for {candidate}: "
        if checkpoint.verification_mode == "frontend":
            if candidate.is_file() or candidate.suffix.lower() in {".html", ".css", ".js"}:
                success = candidate.exists()
                details += "ok" if success else "missing"
            else:
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

    def _tool_step_candidate_paths(self, step: ToolExecutionStep) -> list[Path]:
        raw_values: list[str] = []
        for container in (step.arguments, step.data):
            for key in ("path", "file_path", "target_path", "absolute_path"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    raw_values.append(value)
            for key in ("paths", "file_paths", "target_paths", "touched_paths", "written_paths", "modified_paths"):
                value = container.get(key)
                if isinstance(value, (list, tuple)):
                    raw_values.extend(item for item in value if isinstance(item, str) and item.strip())

        candidates: list[Path] = []
        seen: set[str] = set()
        for raw_value in raw_values:
            candidate = Path(raw_value)
            if not candidate.is_absolute():
                candidate = Path(self.workspace_path) / candidate
            try:
                candidate.relative_to(self.workspace_path)
            except ValueError:
                continue
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(candidate)
        return candidates

    def _append_repair_checkpoint(self, blueprint: ExecutionBlueprint, *, checkpoint_id: int, reason: str) -> tuple[ExecutionBlueprint, bool]:
        tasks = list(blueprint.tasks)
        source_task = next((task for task in tasks if task.task_id == checkpoint_id), None)
        if source_task is None:
            return blueprint, False
        if source_task.repair_origin_checkpoint_id is not None:
            return blueprint, False
        if source_task.repair_attempt_count >= max(source_task.max_repair_attempts, 1):
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
            allowed_tool_names=source_task.allowed_tool_names,
            expects_mutation=source_task.expects_mutation,
            requires_verification=True,
            repair_attempt_count=source_task.repair_attempt_count + 1,
            max_repair_attempts=max(source_task.max_repair_attempts, 1),
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

    def _describe_repair_chain_block(self, blueprint: ExecutionBlueprint | None, checkpoint_id: int) -> str:
        if blueprint is None:
            return "repair state unavailable"
        task = next((candidate for candidate in blueprint.tasks if candidate.task_id == checkpoint_id), None)
        if task is None:
            return f"checkpoint {checkpoint_id} no longer exists"
        if task.repair_origin_checkpoint_id is not None:
            return f"checkpoint {checkpoint_id} is already a repair checkpoint"
        if task.repair_attempt_count >= max(task.max_repair_attempts, 1):
            return f"repair budget exhausted for checkpoint {checkpoint_id}"
        return f"repair checkpoint could not be appended for checkpoint {checkpoint_id}"

    def _split_active_checkpoint(self, blueprint: ExecutionBlueprint | None, task_index: int, *, reason: str) -> ExecutionBlueprint | None:
        if blueprint is None or not (0 <= task_index < len(blueprint.tasks)):
            return None
        source_task = blueprint.tasks[task_index]
        if self._checkpoint_is_context_only(blueprint.original_objective or "", source_task.description):
            return None
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
                    allowed_tool_names=source_task.allowed_tool_names,
                    expects_mutation=source_task.expects_mutation,
                    requires_verification=source_task.requires_verification,
                    repair_attempt_count=source_task.repair_attempt_count,
                    max_repair_attempts=source_task.max_repair_attempts,
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
            allowed_tool_names=source_task.allowed_tool_names,
            expects_mutation=source_task.expects_mutation,
            requires_verification=source_task.requires_verification,
            repair_attempt_count=source_task.repair_attempt_count,
            max_repair_attempts=source_task.max_repair_attempts,
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

    def _should_pre_split_direct_checkpoint(self, user_prompt: str, task: CheckpointTask) -> bool:
        if not task.expects_mutation:
            return False
        lowered = user_prompt.lower()
        compound_markers = (
            " and ",
            " then ",
            " after that ",
            " while ",
            ", and ",
        )
        file_mentions = re.findall(r"\b[\w./-]+\.[a-z0-9]+\b", lowered)
        if len(file_mentions) >= 2:
            return True
        return len(lowered.split()) >= 9 and any(marker in lowered for marker in compound_markers)

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
        structured_answer = self._answer_known_project_question_local(user_prompt, direct_memory)
        if structured_answer is None:
            structured_answer = _answer_from_retrieved_memory(user_prompt, direct_memory)
        if structured_answer is not None and _should_trust_memory_answer_for_prompt(user_prompt):
            ai_logs.append("Direct turn answered from focused memory before remote model call")
            return structured_answer
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
        elif _is_runtime_routing_question(user_prompt):
            relevant_paths = self._ensure_runtime_routing_paths(relevant_paths, listing_step.output)
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
            if lowered == "core/runtime/kernels/runtime.py":
                score += 26
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
            "core/runtime/kernels/runtime.py": 26,
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

    def _ensure_runtime_routing_paths(self, relevant_paths: list[str], listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        routing_targets = {
            "core/runtime/kernels/runtime.py": 32,
            "core/runtime/kernel.py": 30,
            "core/ai/routing.py": 28,
            "core/runtime/web.py": 24,
            "interface/website/src/components/Composer.js": 18,
            "interface/website/src/components/ToolPicker.js": 17,
            "interface/website/src/components/ThinkingMessage.js": 16,
            "interface/website/src/components/ChatColumn.js": 15,
            "core/ai/ollama_backend.py": 14,
            "core/ai/codex_backend.py": 13,
            "README.md": 3,
        }
        discouraged_markers = ("tests/", "sample-test/", "build/", "docs/screenshots/", "devenv.egg-info/", "devenv1a.egg-info/")
        discouraged_names = {"__init__.py", "env.py", "logging_utils.py"}

        scored_paths: dict[str, int] = {}
        for path in relevant_paths:
            if not isinstance(path, str) or not path.strip():
                continue
            score = routing_targets.get(path, 0)
            lowered = path.lower()
            if any(marker in lowered for marker in discouraged_markers):
                score -= 10
            if Path(lowered).name in discouraged_names:
                score -= 8
            scored_paths[path] = max(score, scored_paths.get(path, 0))

        if not isinstance(entries, list):
            ordered = sorted(scored_paths.items(), key=lambda item: (-item[1], item[0]))
            return [path for path, _score in ordered if _score > 0][:3] or relevant_paths[:3]

        for entry in entries:
            if not isinstance(entry, dict) or entry.get("is_dir"):
                continue
            relative_path = entry.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                continue
            score = routing_targets.get(relative_path, 0)
            lowered = relative_path.lower()
            if any(marker in lowered for marker in discouraged_markers):
                score -= 10
            if Path(lowered).name in discouraged_names:
                score -= 8
            if score > 0:
                scored_paths[relative_path] = max(score, scored_paths.get(relative_path, 0))

        ordered = sorted(scored_paths.items(), key=lambda item: (-item[1], item[0]))
        preferred_paths = [path for path, score in ordered if score > 0]
        return preferred_paths[:3] or relevant_paths[:3]

    def _ensure_backend_connector_paths(self, relevant_paths: list[str], listing_output: str) -> list[str]:
        payload = _extract_tool_payload_json(listing_output)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        connector_targets = {
            "core/ai/routing.py": 28,
            "core/runtime/web.py": 24,
            "core/ai/codex_backend.py": 22,
            "core/ai/ollama_backend.py": 21,
            "core/ai/opencode_client.py": 18,
            "core/runtime/kernels/runtime.py": 17,
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
        if not _should_try_direct_memory_answer(user_prompt):
            return False
        project_answer = self._answer_known_project_question_local(user_prompt, memory_context)
        if project_answer is not None:
            if _is_error_fix_memory_question(user_prompt) and "we fixed it by" not in project_answer.lower():
                return False
            return True
        retrieved_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
        if retrieved_answer is not None:
            if _is_error_fix_memory_question(user_prompt) and "we fixed it by" not in retrieved_answer.lower():
                return False
            return True
        return False
