class KernelPlanningMixin:
    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolExecutionStep:
        logger.info("Intercepted tool call: tool=%s arguments=%s", tool_call.tool_name, tool_call.arguments)
        normalized_arguments = self.sandbox.normalize_arguments(self._repair_tool_arguments(tool_call))
        explicit_path = self._active_checkpoint_explicit_path()
        active_task = None
        if self.active_blueprint and 0 <= self.active_blueprint.active_task_pointer < len(self.active_blueprint.tasks):
            active_task = self.active_blueprint.tasks[self.active_blueprint.active_task_pointer]
        if (
            tool_call.tool_name == "list_directory"
            and explicit_path
            and active_task is not None
            and self._checkpoint_is_context_only(self.active_plan_prompt or "", active_task.description)
            and Path(explicit_path).suffix
            and "read_file" in self.tools
        ):
            resolved_explicit = (Path(self.workspace_path) / explicit_path).resolve()
            if resolved_explicit.exists():
                logger.info(
                    "Forced explicit file checkpoint inspection to read_file: requested=%s explicit=%s",
                    normalized_arguments.get("path"),
                    resolved_explicit,
                )
                tool_call = ToolCallRequest(
                    call_id=tool_call.call_id,
                    tool_name="read_file",
                    arguments={"path": str(resolved_explicit), "features": "content"},
                )
                normalized_arguments = self.sandbox.normalize_arguments(tool_call.arguments)
        if tool_call.tool_name == "list_directory":
            path_value = normalized_arguments.get("path")
            if isinstance(path_value, str):
                resolved_path = Path(path_value).expanduser().resolve()
                if resolved_path.is_file() and "read_file" in self.tools:
                    logger.info(
                        "Repaired list_directory file inspection into read_file: requested=%s repaired=%s",
                        path_value,
                        resolved_path,
                    )
                    tool_call = ToolCallRequest(
                        call_id=tool_call.call_id,
                        tool_name="read_file",
                        arguments={"path": str(resolved_path), "features": "content"},
                    )
                    normalized_arguments = self.sandbox.normalize_arguments(tool_call.arguments)
        unsafe_argument = self.sandbox.find_unsafe_argument(normalized_arguments)
        if unsafe_argument is not None:
            _key, value = unsafe_argument
            logger.warning("Sandbox violation detected: tool=%s path=%s", tool_call.tool_name, value)
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=normalized_arguments,
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
        arguments = {
            (key[:-1] if isinstance(key, str) and key.endswith(":") else key): value
            for key, value in dict(tool_call.arguments).items()
        }
        if "max_depth" in arguments and isinstance(arguments["max_depth"], str) and arguments["max_depth"].isdigit():
            arguments["max_depth"] = int(arguments["max_depth"])

        if tool_call.tool_name == "list_directory":
            path_value = arguments.get("path")
            if "mode" not in arguments:
                arguments["mode"] = "recursive"
            if isinstance(path_value, str):
                repaired_path = self._repair_directory_path(path_value)
                if repaired_path is not None:
                    arguments["path"] = repaired_path
        if tool_call.tool_name == "run_diagnostics" and not arguments.get("target_path"):
            arguments["target_path"] = self.workspace_path
        if tool_call.tool_name in {"read_file", "edit_file", "write_file", "remove_file"}:
            path_value = arguments.get("path")
            if isinstance(path_value, str):
                path_value = _sanitize_model_generated_path(path_value)
                arguments["path"] = path_value
                repaired_path = self._repair_workspace_file_path(path_value)
                if repaired_path is not None:
                    arguments["path"] = repaired_path
        if tool_call.tool_name == "generate_pdf":
            arguments["workspace_root"] = self.workspace_path
            output_path = arguments.get("output_path")
            if isinstance(output_path, str):
                output_path = _sanitize_model_generated_path(output_path)
                repaired_pdf_path = self._repair_workspace_pdf_output_path(output_path)
                arguments["output_path"] = repaired_pdf_path if repaired_pdf_path is not None else output_path
        if tool_call.tool_name == "write_file" and "mode" not in arguments:
            path_value = arguments.get("path")
            if isinstance(path_value, str):
                target = Path(path_value)
                if not target.is_absolute():
                    target = (Path(self.workspace_path) / target).resolve()
                arguments["mode"] = "overwrite" if target.exists() else "fresh"
        if tool_call.tool_name in WRITE_EXECUTION_TOOLS | DELETE_EXECUTION_TOOLS | frozenset({"edit_file"}):
            path_value = arguments.get("path")
            if isinstance(path_value, str):
                target_path_hint = self._active_scaffold_target_path()
                if target_path_hint:
                    repaired_path = self._repair_scaffold_path(path_value, target_path_hint)
                    if repaired_path is not None:
                        arguments["path"] = repaired_path

        return arguments

    def _repair_workspace_pdf_output_path(self, path_value: str) -> str | None:
        cleaned = str(path_value or "").strip()
        if not cleaned:
            return None
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            if candidate.suffix.lower() == ".pdf" and ".." not in candidate.parts:
                return cleaned
            return None
        try:
            relative = candidate.resolve().relative_to(Path(self.workspace_path).resolve())
        except ValueError:
            return None
        if relative.suffix.lower() != ".pdf" or ".." in relative.parts:
            return None
        return relative.as_posix()

    def _active_scaffold_target_path(self) -> str | None:
        prompt = self.active_plan_prompt or ""
        task_description = ""
        if self.active_blueprint and 0 <= self.active_blueprint.active_task_pointer < len(self.active_blueprint.tasks):
            task_description = self.active_blueprint.tasks[self.active_blueprint.active_task_pointer].description
        return self._derive_scaffold_target_path(prompt, task_description)

    def _derive_scaffold_target_path(self, user_prompt: str, task_description: str = "") -> str | None:
        prompt_text = user_prompt.strip().lower()
        task_text = task_description.strip().lower()
        combined = f"{prompt_text} {task_text}".strip()
        if "frontend" in prompt_text and "calendar" in prompt_text:
            return "calendar/frontend"
        if not self._is_scaffold_request(combined):
            return None
        direct_match = re.search(r"\b([a-z0-9_.-]+/[a-z0-9_./-]+)\b", prompt_text)
        if direct_match:
            candidate = direct_match.group(1).strip("/")
            suffix = Path(candidate).suffix.lower()
            if suffix in {".html", ".css", ".js", ".py"}:
                return str(Path(candidate).parent).replace("\\", "/")
            return candidate
        nested_match = re.search(r"\b([a-z0-9_-]+)\s+folder\s+(?:inside|in|under)\s+([a-z0-9_/-]+)\b", prompt_text)
        if nested_match:
            child, parent = nested_match.groups()
            return f"{parent.strip('/')}/{child.strip('/')}"
        explicit_folder_match = re.search(r"\bfolder\s+([a-z0-9_/-]+)\b", prompt_text)
        if explicit_folder_match:
            candidate = explicit_folder_match.group(1).strip("/")
            if candidate and candidate not in {"in", "inside", "under"}:
                return candidate
        scaffold_kind = _local_scaffold_kind(user_prompt, task_description)
        if scaffold_kind == "notes":
            return "notesapp"
        if scaffold_kind == "kanban":
            return "kanbanapp"
        if scaffold_kind == "weather":
            return "weatherapp"
        if scaffold_kind == "date":
            return "dateapp"
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
        if tool_name not in WRITE_EXECUTION_TOOLS:
            return None

        path_value = arguments.get("path")
        content_value = arguments.get("content")
        if not isinstance(path_value, str):
            return None
        target_path_hint = self._active_scaffold_target_path()
        target_path: Path | None = None
        requested_path = Path(path_value).expanduser()
        if target_path_hint is not None:
            target_path = (Path(self.workspace_path) / target_path_hint).resolve()
            if not requested_path.is_absolute():
                requested_path = (Path(self.workspace_path) / requested_path).resolve()
            if requested_path == target_path and isinstance(content_value, str) and not content_value.strip():
                return (
                    f"Use write_file on a file inside {target_path_hint} such as "
                    f"{target_path_hint}/index.html, not on the folder itself."
                )
        if isinstance(content_value, str) and not content_value.strip():
            active_task = None
            if self.active_blueprint and 0 <= self.active_blueprint.active_task_pointer < len(self.active_blueprint.tasks):
                active_task = self.active_blueprint.tasks[self.active_blueprint.active_task_pointer]
            if active_task is not None:
                lowered = active_task.description.lower()
                if any(marker in lowered for marker in ("create ", "add ", "wire ", "connect ", "update ", "implement ")):
                    return f"write_file for `{path_value}` requires non-empty content for this checkpoint."
        if target_path_hint is None:
            return None
        return None

    def _repair_directory_path(self, requested_path: str) -> str | None:
        candidate = Path(requested_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.workspace_path) / candidate

        if candidate.exists():
            return str(candidate.resolve())

        requested_text = str(requested_path).strip().lower()
        if candidate.is_absolute() and requested_text.startswith("/workspace/"):
            logger.info("Repaired placeholder workspace directory path: requested=%s repaired=%s", requested_path, self.workspace_path)
            return str(Path(self.workspace_path).resolve())

        requested_name = candidate.name.lower()
        requested_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", requested_name)
            if token and token not in {"app", "workspace", "project", "src"}
        }
        try:
            directories: list[Path] = []
            ignored_dirs = set(NOISE_DIRECTORIES) | {"site-packages", "codereferences"}
            for entry in Path(self.workspace_path).rglob("*"):
                if any(part in ignored_dirs for part in entry.parts):
                    continue
                if entry.is_dir():
                    directories.append(entry)
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

        token_matches = [
            entry
            for entry in directories
            if requested_tokens
            and requested_tokens.intersection(
                {
                    token
                    for token in re.split(r"[^a-z0-9]+", entry.name.lower())
                    if token and token not in {"app", "workspace", "project", "src"}
                }
            )
        ]
        if len(token_matches) == 1:
            logger.info("Repaired missing directory path by token overlap: requested=%s repaired=%s", requested_path, token_matches[0])
            return str(token_matches[0].resolve())

        return None

    def _repair_workspace_file_path(self, requested_path: str) -> str | None:
        active_explicit_path = self._active_checkpoint_explicit_path()
        sanitized_requested_path = _sanitize_model_generated_path(requested_path)
        candidate = Path(sanitized_requested_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.workspace_path) / candidate

        if candidate.exists():
            return str(candidate.resolve())

        if active_explicit_path:
            explicit_candidate = Path(active_explicit_path)
            if _normalized_path_identity(explicit_candidate.name) == _normalized_path_identity(candidate.name):
                return str((Path(self.workspace_path) / explicit_candidate).resolve())

        requested_name = candidate.name.lower()
        requested_suffix = candidate.suffix.lower()
        if not requested_name and not requested_suffix:
            return None

        matches_by_name: list[Path] = []
        matches_by_suffix: list[Path] = []
        ignored_dirs = set(NOISE_DIRECTORIES) | {"site-packages", "codereferences"}
        try:
            for entry in Path(self.workspace_path).rglob("*"):
                if any(part in ignored_dirs for part in entry.parts):
                    continue
                if not entry.is_file():
                    continue
                lowered_name = entry.name.lower()
                if lowered_name == requested_name:
                    matches_by_name.append(entry)
                if requested_suffix and entry.suffix.lower() == requested_suffix:
                    matches_by_suffix.append(entry)
        except OSError:
            return None

        if len(matches_by_name) == 1:
            logger.info("Repaired missing file path by basename: requested=%s repaired=%s", requested_path, matches_by_name[0])
            return str(matches_by_name[0].resolve())
        if requested_suffix and len(matches_by_suffix) == 1:
            logger.info("Repaired missing file path by unique suffix: requested=%s repaired=%s", requested_path, matches_by_suffix[0])
            return str(matches_by_suffix[0].resolve())

        return None

    def _active_checkpoint_explicit_path(self) -> str | None:
        if self.active_blueprint and 0 <= self.active_blueprint.active_task_pointer < len(self.active_blueprint.tasks):
            description = self.active_blueprint.tasks[self.active_blueprint.active_task_pointer].description
            mentioned = _first_backticked_path(description)
            if mentioned:
                return mentioned
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

    def _retrieve_card_memory_context(self, user_prompt: str) -> tuple[str, list[Any]]:
        memory = getattr(self, "memory", None)
        if memory is None or not hasattr(memory, "retrieve_cards"):
            return "", []
        if _should_skip_retrieval_for_prompt(user_prompt) or _should_skip_current_workspace_memory_lookup(user_prompt):
            return "", []
        try:
            from core.memory.card_retrieval import DEFAULT_MIN_SCORE
            from core.runtime.context_builder import _build_query_variants
        except Exception:
            return "", []
        lanes = list(_build_query_variants(user_prompt))
        try:
            matches = memory.retrieve_cards(user_prompt, lanes=lanes)
        except Exception as exc:
            logger.warning("Card retrieval failed; continuing without card context: error=%s", exc)
            return "", []
        if not matches:
            return "", []
        if max(match.score for match in matches) < DEFAULT_MIN_SCORE:
            return "", []
        lines = ["## Interaction Memory"]
        for match in matches:
            card = match.card
            intent = " ".join(card.intent_text.split())[:200]
            answer = " ".join(card.answer_text.split())[:300]
            lines.append(
                f"- [{card.project or 'unknown'}] {intent} — {answer} (source: {card.provider}:{card.session_id[:8]})"
            )
        return "\n".join(lines), matches

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
        if _should_skip_trace_memory_lookup(user_prompt):
            metadata["external_context_reason"] = "Skipped memory retrieval for a trace-inspection prompt."
            return "", metadata
        if self._should_skip_current_workspace_memory_lookup(user_prompt):
            metadata["external_context_reason"] = "Skipped memory retrieval for a current-workspace inspection prompt."
            return "", metadata
        lexical_query = _compose_external_memory_query(user_prompt, self.ephemeral_history)
        lexical_context = self._retrieve_lexical_memory_context(user_prompt, search_query=lexical_query)
        if lexical_context:
            memory_context = lexical_context
            self._persist_last_retrieval_trace(RetrievalTrace(markdown_context=memory_context))
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
        card_context, card_matches = self._retrieve_card_memory_context(user_prompt)
        if card_context:
            combined = card_context if not memory_context.strip() else f"{memory_context.rstrip()}\n\n{card_context}"
            metadata.update(
                {
                    "card_context_state": "reused_prior_cards",
                    "card_context_count": len(card_matches),
                    "card_context_sources": [match.card.session_id for match in card_matches],
                }
            )
            self._persist_last_retrieval_trace(RetrievalTrace(markdown_context=combined))
            return combined, metadata
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
        fallback_context = "## Retrieved Memory\n" + "\n".join(unique_lines[:6])
        if (
            _is_memory_recall_question(user_prompt)
            or _is_memory_follow_up_question(user_prompt)
            or _is_session_history_question(user_prompt)
        ):
            known_answer = _answer_known_project_question(user_prompt, fallback_context)
            if known_answer is not None and _memory_answer_matches_question(user_prompt, [known_answer]):
                return fallback_context
            retrieved_answer = _answer_from_retrieved_memory(user_prompt, fallback_context)
            if retrieved_answer is not None and _memory_answer_matches_question(user_prompt, [retrieved_answer]):
                return fallback_context
            return ""
        return fallback_context

    def _compact_lexical_memory_context(self, user_prompt: str, lines: list[str]) -> str:
        if not lines or not _should_try_direct_memory_answer(user_prompt):
            return ""

        selected: list[str] = []
        for line in lines[:6]:
            selected.append(line)
            candidate_context = "## Retrieved Memory\n" + "\n".join(selected)
            known_answer = _answer_known_project_question(user_prompt, candidate_context)
            if known_answer is not None and _memory_answer_matches_question(user_prompt, [known_answer]):
                return candidate_context
            retrieved_answer = _answer_from_retrieved_memory(user_prompt, candidate_context)
            if retrieved_answer is not None and _memory_answer_matches_question(user_prompt, [retrieved_answer]):
                return candidate_context
        return ""
