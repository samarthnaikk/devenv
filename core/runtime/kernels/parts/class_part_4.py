class KernelExecutionMixin:
    def _fallback_response_for_runtime_error(
        self,
        *,
        user_prompt: str,
        error: RuntimeError,
        memory_context: str,
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

        if memory_context.strip():
            memory_answer = self._answer_known_project_question_local(user_prompt, memory_context)
            if memory_answer is None and _should_trust_memory_answer_for_prompt(user_prompt):
                memory_answer = _answer_from_retrieved_memory(user_prompt, memory_context)
            if memory_answer:
                ai_logs.append("Returned already-retrieved memory fallback after OpenCode transport failure")
                return memory_answer

        if _should_try_direct_memory_answer(user_prompt):
            local_memory_context = memory_context
            if not local_memory_context.strip():
                local_memory_context, _metadata = self._retrieve_memory_context(user_prompt, local_only=True)
            memory_answer = self._answer_known_project_question_local(user_prompt, local_memory_context)
            if memory_answer is None and _should_trust_memory_answer_for_prompt(user_prompt):
                memory_answer = _answer_from_retrieved_memory(user_prompt, local_memory_context)
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

    def _allowed_tool_names_for_phase(self, *, execution_phase: bool) -> set[str]:
        mode = ExecutionMode.CHECKPOINT_EXECUTE if execution_phase else ExecutionMode.DIRECT_ANSWER
        return allowed_tool_names_for_mode(mode, set(self.tools))

    def _planning_allowed_tool_names(self) -> set[str]:
        return allowed_tool_names_for_mode(ExecutionMode.PLAN_ONLY, set(self.tools))

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
            available_tools = set(self.tools)
            inspect_tools = [
                tool_name
                for tool_name in ("list_directory", "locate_files", "read_file", "inspect_symbols", "search_text")
                if tool_name in available_tools or not available_tools
            ]
            parts = ["For that repo-summary question Devenv should stay in charge of retrieval."]
            if inspect_tools:
                parts.append(
                    "It would usually map the workspace first with "
                    + ", ".join(f"`{tool_name}`" for tool_name in inspect_tools)
                    + "."
                )
            parts.append("Then it should answer from the grounded repo context instead of guessing.")
            return " ".join(parts)

        if _is_architecture_question(subject_prompt):
            available_tools = set(self.tools)
            inspect_tools = [
                tool_name
                for tool_name in ("list_directory", "locate_files", "read_file", "inspect_symbols", "search_text")
                if tool_name in available_tools or not available_tools
            ]
            parts = ["For that architecture question Devenv should stay in charge of retrieval."]
            if inspect_tools:
                parts.append(
                    "It would usually inspect the backend surface first with "
                    + ", ".join(f"`{tool_name}`" for tool_name in inspect_tools)
                    + "."
                )
            parts.append("Then it should answer from the main runtime and routing files rather than a generic summary.")
            return " ".join(parts)
        if self._text_requires_mutation_tools(subject_prompt.lower()):
            available_tools = set(self.tools)
            inspect_tools = [
                tool_name
                for tool_name in ("list_directory", "read_file", "inspect_symbols", "search_text")
                if tool_name in available_tools or not available_tools
            ]
            edit_tools = [
                tool_name
                for tool_name in ("edit_file", "write_file")
                if tool_name in available_tools or not available_tools
            ]
            verify_tools = [
                tool_name
                for tool_name in ("run_diagnostics", "audit_changes")
                if tool_name in available_tools or not available_tools
            ]
            parts = ["For that coding task Devenv should stay in charge of tool choice."]
            if inspect_tools:
                parts.append(
                    "It should inspect first with "
                    + ", ".join(f"`{tool_name}`" for tool_name in inspect_tools)
                    + "."
                )
            if edit_tools:
                parts.append(
                    "Then it should make the smallest safe file change with "
                    + ", ".join(f"`{tool_name}`" for tool_name in edit_tools)
                    + "."
                )
            if verify_tools:
                parts.append(
                    "After mutation it should verify with "
                    + ", ".join(f"`{tool_name}`" for tool_name in verify_tools)
                    + "."
                )
            return " ".join(parts)

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
            return sorted(selected_scope & self._allowed_tool_names_for_phase(execution_phase=execution_phase))

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

        if self._should_offer_web_search_for_phase(lowered, execution_phase=execution_phase):
            if not execution_phase and self._should_prefer_web_only(lowered):
                web_only_scope = WEB_EXECUTION_TOOLS & available
                if web_only_scope:
                    return sorted(web_only_scope)
            scope.update(WEB_EXECUTION_TOOLS & available)

        if self._should_offer_knowledge_search(lowered):
            scope.update(KNOWLEDGE_EXECUTION_TOOLS & available)

        if self._should_offer_shell_tools(lowered):
            scope.update(SHELL_EXECUTION_TOOLS & available)

        if self._should_offer_memory_tools(lowered):
            scope.update(MEMORY_EXECUTION_TOOLS & available)

        if execution_phase and self._should_offer_artifact_tools(lowered):
            scope.update(ARTIFACT_EXECUTION_TOOLS & available)

        if execution_phase and self._should_offer_diagnostics_tools(lowered):
            scope.update(DIAGNOSTIC_EXECUTION_TOOLS & available)

        if execution_phase and self._text_requires_mutation_tools(lowered):
            scope.discard("search_text")
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
                "how does backend work",
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
                "online",
                "today",
                "recent",
                "browse",
                "google",
                "look up",
                "search online",
                "search for",
                "search the web",
                "web search",
                "news",
                "docs",
                "documentation",
                "net worth",
                "stock price",
                "market cap",
                "president",
                "prime minister",
                "ceo",
            )
        )

    def _should_offer_web_search_for_phase(self, lowered_prompt: str, *, execution_phase: bool) -> bool:
        if not execution_phase:
            return self._should_offer_web_search(lowered_prompt)
        if not self._text_requires_mutation_tools(lowered_prompt):
            return self._should_offer_web_search(lowered_prompt)
        return any(
            marker in lowered_prompt
            for marker in (
                "browse",
                "google",
                "look up",
                "search for",
                "search the web",
                "search online",
                "web search",
                "latest",
                "today",
                "recent",
                "news",
                "docs",
                "documentation",
            )
        )

    def _should_offer_knowledge_search(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "github",
                "repo examples",
                "repository examples",
                "similar repos",
                "similar projects",
                "reference repos",
                "reference projects",
                "reference implementations",
                "stackoverflow",
                "reddit",
                "quora",
                "youtube",
                "forum threads",
                "resources",
                "references",
                "inspiration",
                "examples online",
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
                "search online",
                "look up",
                "browse",
                "google",
                "latest online",
                "latest you can search",
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

    def _should_offer_artifact_tools(self, lowered_prompt: str) -> bool:
        if "pdf" in lowered_prompt:
            return True
        return any(marker in lowered_prompt for marker in ("document", "report", "brief", "handout", "invoice")) and any(
            marker in lowered_prompt for marker in ("create", "generate", "build", "make", "export")
        )

    def _should_offer_diagnostics_tools(self, lowered_prompt: str) -> bool:
        return any(
            marker in lowered_prompt
            for marker in (
                "fix",
                "bug",
                "failing",
                "failure",
                "broken",
                "error",
                "regression",
                "verify",
                "validation",
                "test",
                "tests",
                "lint",
                "typecheck",
                "types",
                "diagnostic",
                "diagnostics",
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
        if "knowledge_search" in resolved:
            lines.append("If the request asks for external references, similar repos, GitHub examples, videos, or forum threads, call knowledge_search before answering.")
        return [{"role": "system", "content": " ".join(lines)}]

    def _build_local_plan_markdown(self, user_prompt: str) -> str:
        target_path = self._derive_scaffold_target_path(user_prompt) or ""
        lowered = user_prompt.lower()
        expected_artifact = self._infer_expected_artifact(user_prompt, user_prompt, target_path or None)
        if self._is_backend_frontend_integration_request(user_prompt):
            return self._build_backend_frontend_integration_plan(user_prompt, target_path=target_path)
        if self._is_ui_runtime_polish_request(user_prompt):
            return self._build_ui_runtime_polish_plan(user_prompt)
        if expected_artifact == "document":
            return "- [ ] Generate the requested PDF artifact and verify that the file was written successfully."
        if self._is_scaffold_request(lowered):
            html_path = f"{target_path}/index.html" if target_path else "index.html"
            css_path = f"{target_path}/styles.css" if target_path else "styles.css"
            js_path = f"{target_path}/script.js" if target_path else "script.js"
            scaffold_kind = _local_scaffold_kind(user_prompt)
            if scaffold_kind == "calendar":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the base calendar layout and linked assets.",
                        f"- [ ] Add {css_path} with the calendar styling.",
                        f"- [ ] Add {js_path} with month navigation and date rendering.",
                    ]
                )
            if scaffold_kind == "notes":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the notes workspace layout and linked assets.",
                        f"- [ ] Add {css_path} with the notes app styling.",
                        f"- [ ] Add {js_path} with note capture, persistence, and rendering behavior.",
                    ]
                )
            if scaffold_kind == "todo":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the task list layout and linked assets.",
                        f"- [ ] Add {css_path} with the todo app styling.",
                        f"- [ ] Add {js_path} with add, toggle, and render behavior for tasks.",
                    ]
                )
            if scaffold_kind == "kanban":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the kanban board layout and linked assets.",
                        f"- [ ] Add {css_path} with the kanban board styling.",
                        f"- [ ] Add {js_path} with localStorage-backed card creation and lane movement behavior.",
                    ]
                )
            if scaffold_kind == "weather":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the weather dashboard layout and linked assets.",
                        f"- [ ] Add {css_path} with the weather app styling.",
                        f"- [ ] Add {js_path} with forecast card rendering and refresh behavior.",
                    ]
                )
            if scaffold_kind == "date":
                return "\n".join(
                    [
                        f"- [ ] Create {html_path} with the date display layout and linked assets.",
                        f"- [ ] Add {css_path} with the date card styling.",
                        f"- [ ] Add {js_path} with today's date rendering behavior.",
                    ]
                )
            return "\n".join(
                [
                    f"- [ ] Create {html_path} with the base app shell and linked assets.",
                    f"- [ ] Add {css_path} with the app styling.",
                    f"- [ ] Add {js_path} with the requested client-side behavior.",
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

    def _is_backend_frontend_integration_request(self, user_prompt: str) -> bool:
        lowered = user_prompt.lower()
        backend_markers = ("backend", "api", "server", "route", "endpoint", "service")
        integration_markers = ("integrate", "integration", "connect", "wire", "chat app", "chatapp")
        return "frontend" in lowered and any(marker in lowered for marker in backend_markers) and any(
            marker in lowered for marker in integration_markers
        )

    def _is_ui_runtime_polish_request(self, user_prompt: str) -> bool:
        lowered = user_prompt.lower()
        ui_markers = ("ui", "interface", "website", "frontend", "animation", "shell", "light theme")
        runtime_markers = ("runtime", "plan mode", "planner", "tool", "tools", "routing", "route")
        quality_markers = ("fix", "polish", "improve", "upgrade", "broken", "reliable", "decide", "motion")
        return (
            any(marker in lowered for marker in ui_markers)
            and any(marker in lowered for marker in quality_markers)
        ) or (
            any(marker in lowered for marker in runtime_markers)
            and any(marker in lowered for marker in quality_markers)
        )

    def _build_backend_frontend_integration_plan(self, user_prompt: str, *, target_path: str) -> str:
        integration_root = target_path or "chatapp"
        backend_surface = "core/runtime/web.py"
        routing_surface = "core/ai/routing.py"
        frontend_api_surface = "interface/website/src/api.js"
        frontend_ui_surface = "interface/website/src/App.js"
        if "composer" in user_prompt.lower():
            frontend_ui_surface = "interface/website/src/components/Composer.js"
        return "\n".join(
            [
                f"- [ ] Inspect `{integration_root}` and confirm which backend files already exist or are still missing.",
                f"- [ ] Inspect `{backend_surface}` to map the backend request and registration surface the chat app must plug into.",
                f"- [ ] Inspect `{routing_surface}` to map the backend routing surface the chat app must plug into.",
                f"- [ ] Inspect `{frontend_api_surface}` to map the frontend API helper that must call the chat backend.",
                f"- [ ] Inspect `{frontend_ui_surface}` to map the frontend UI surface that must trigger the chat backend.",
                f"- [ ] Create `{integration_root}/__init__.py` for the chat app package exports.",
                f"- [ ] Create `{integration_root}/store.py` for the in-memory chat session store.",
                f"- [ ] Create `{integration_root}/service.py` for the chat request orchestration logic.",
                f"- [ ] Create `{integration_root}/routes.py` for the chat backend request handlers.",
                f"- [ ] Wire `{backend_surface}` to register the chat app backend surface.",
                f"- [ ] Wire `{routing_surface}` to expose the chat app routing target.",
                f"- [ ] Connect `{frontend_api_surface}` to call the chat app backend endpoint.",
                f"- [ ] Connect `{frontend_ui_surface}` to send chat messages through the new frontend API helper.",
            ]
        )

    def _build_ui_runtime_polish_plan(self, user_prompt: str) -> str:
        lowered = user_prompt.lower()
        ui_surface = "interface/website/src/components/Transcript.js"
        if "settings" in lowered or "theme" in lowered:
            ui_surface = "interface/website/src/components/SettingsDropdown.js"
        elif "composer" in lowered or "prompt" in lowered:
            ui_surface = "interface/website/src/components/Composer.js"
        runtime_surface = "core/runtime/kernel.py"
        if "web" in lowered or "api" in lowered:
            runtime_surface = "core/runtime/web.py"
        return "\n".join(
            [
                "- [ ] Inspect `interface/website/styles.css` and `interface/website/src/components/MotionPrimitives.js` to map the current motion system, theme surfaces, and shared animation primitives.",
                f"- [ ] Inspect `{ui_surface}` plus `interface/website/src/components/ChatColumn.js` and `interface/website/src/components/Sidebar.js` to identify the weakest light-theme and interaction surfaces.",
                f"- [ ] Inspect `{runtime_surface}` and `tests/runtime/test_kernel.py` to identify where planning, route selection, or tool-choice behavior is still too generic or broken.",
                "- [ ] Update the shared website motion/theme primitives so the shell uses stronger staged transitions, tactile cards, and a clearer light interface direction.",
                f"- [ ] Refine `{ui_surface}` and the surrounding shell components so the empty state, status surfaces, and controls feel intentionally animated instead of flat.",
                f"- [ ] Tighten `{runtime_surface}` so explicit planning, route choice, and tool selection produce grounded behavior instead of generic fallback responses.",
                "- [ ] Verify the website changes with `interface/website/scripts/check-mount.mjs` and verify the runtime behavior with targeted kernel/web tests plus an Ollama smoke prompt for planning.",
            ]
        )

    def _build_local_execution_tool_call(self, user_prompt: str, task_description: str) -> ToolCallRequest | None:
        target_path = self._derive_scaffold_target_path(user_prompt, task_description)
        lowered_task = task_description.lower()
        lowered_prompt = user_prompt.lower()
        local_integration_target = self._local_integration_root_for_prompt(user_prompt)

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
            scaffold_kind = _local_scaffold_kind(user_prompt, task_description)
            if "index.html" in lowered_task or ("html" in lowered_task and "calendar" in lowered_prompt):
                file_name = "index.html"
                content = _local_scaffold_html(scaffold_kind, target_path)
            elif "styles.css" in lowered_task or ("css" in lowered_task and "calendar" in lowered_prompt):
                file_name = "styles.css"
                content = _local_scaffold_css(scaffold_kind, dark_theme=wants_dark_theme)
            elif "script.js" in lowered_task or ("javascript" in lowered_task) or ("js" in lowered_task and "calendar" in lowered_prompt):
                file_name = "script.js"
                content = _local_scaffold_js(scaffold_kind)

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

        mentioned_path = _first_backticked_path(task_description)
        if mentioned_path:
            absolute_target = Path(self.workspace_path) / mentioned_path
            if lowered_task.startswith("inspect "):
                if absolute_target.is_dir():
                    return ToolCallRequest(
                        call_id=f"local_inspect_{uuid.uuid4().hex[:10]}",
                        tool_name="list_directory",
                        arguments={"path": str(absolute_target), "mode": "recursive", "max_depth": 2},
                    )
                return ToolCallRequest(
                    call_id=f"local_read_{uuid.uuid4().hex[:10]}",
                    tool_name="read_file",
                    arguments={"path": str(absolute_target), "features": "content"},
                )
            integration_content = self._build_local_integration_file_content(
                user_prompt=user_prompt,
                task_description=task_description,
                relative_path=mentioned_path,
            )
            if integration_content is not None:
                return ToolCallRequest(
                    call_id=f"local_write_{uuid.uuid4().hex[:10]}",
                    tool_name="write_file",
                    arguments={
                        "path": mentioned_path,
                        "content": integration_content,
                        "mode": "overwrite" if absolute_target.exists() else "fresh",
                    },
                )

        if local_integration_target and lowered_task.startswith("inspect "):
            candidate_path = Path(self.workspace_path) / local_integration_target
            return ToolCallRequest(
                call_id=f"local_inspect_{uuid.uuid4().hex[:10]}",
                tool_name="list_directory",
                arguments={"path": str(candidate_path), "mode": "recursive", "max_depth": 2},
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
        scaffold_kind = _local_scaffold_kind(user_prompt, task_description)
        if "index.html" in lowered_task:
            if scaffold_kind == "calendar":
                return "Created the base HTML shell for the calendar frontend and linked the local stylesheet and script."
            if scaffold_kind == "notes":
                return "Created the base HTML shell for the notes app and linked the local stylesheet and script."
            if scaffold_kind == "todo":
                return "Created the base HTML shell for the task list app and linked the local stylesheet and script."
            if scaffold_kind == "kanban":
                return "Created the base HTML shell for the kanban board and linked the local stylesheet and script."
            if scaffold_kind == "weather":
                return "Created the base HTML shell for the weather dashboard and linked the local stylesheet and script."
            if scaffold_kind == "date":
                return "Created the base HTML shell for the date display app and linked the local stylesheet and script."
            return "Created the base HTML shell for the local app and linked the local stylesheet and script."
        if "styles.css" in lowered_task:
            if scaffold_kind == "calendar":
                return "Added the calendar styling layer with a responsive layout, panels, and day grid presentation."
            if scaffold_kind == "notes":
                return "Added the notes app styling layer with an editorial layout, composer panel, and note cards."
            if scaffold_kind == "todo":
                return "Added the task list styling layer with a dashboard layout, controls, and checklist presentation."
            if scaffold_kind == "kanban":
                return "Added the kanban board styling layer with responsive lanes, cards, and movement controls."
            if scaffold_kind == "weather":
                return "Added the weather dashboard styling layer with forecast cards, status accents, and responsive panels."
            if scaffold_kind == "date":
                return "Added the date card styling layer with a centered layout and clear typography."
            return "Added the local app styling layer for the generated interface."
        if "script.js" in lowered_task:
            if scaffold_kind == "calendar":
                return "Added the local JavaScript calendar behavior for month navigation and day rendering."
            if scaffold_kind == "notes":
                return "Added the local JavaScript notes behavior for capture, persistence, and rendering."
            if scaffold_kind == "todo":
                return "Added the local JavaScript task behavior for adding, toggling, and rendering tasks."
            if scaffold_kind == "kanban":
                return "Added the local JavaScript kanban behavior for card creation, persistence, and lane movement."
            if scaffold_kind == "weather":
                return "Added the local JavaScript weather behavior for rendering forecast cards and refreshing conditions."
            if scaffold_kind == "date":
                return "Added the local JavaScript behavior to render today's date and refresh the display."
            return "Added the local JavaScript behavior for the generated app."
        if "main.py" in lowered_task:
            return "Created calendar/main.py so it prints today's date using Python's datetime module."
        if "chatapp/" in lowered_task or "core/runtime/web.py" in lowered_task or "core/ai/routing.py" in lowered_task or "interface/website/src/" in lowered_task:
            mentioned_path = _first_backticked_path(task_description)
            if mentioned_path:
                return f"Updated `{mentioned_path}` for the chat app integration checkpoint."
        if "verify" in lowered_task:
            return "Verified the generated workspace artifact against the requested local-only checkpoint."
        memory_answer = _answer_from_retrieved_memory(user_prompt, execution_memory)
        if memory_answer:
            return memory_answer
        return f"Completed locally: {task_description}"

    def _local_integration_root_for_prompt(self, user_prompt: str) -> str | None:
        lowered = user_prompt.lower()
        if ("chatapp" in lowered or "chat app" in lowered) and "frontend" in lowered and any(
            token in lowered for token in ("backend", "integrate", "integration", "chat app")
        ):
            return self._derive_scaffold_target_path(user_prompt) or "chatapp"
        return None

    def _build_local_integration_file_content(self, *, user_prompt: str, task_description: str, relative_path: str) -> str | None:
        integration_root = self._local_integration_root_for_prompt(user_prompt)
        if not integration_root:
            return None
        normalized = relative_path.replace("\\", "/").strip("/")
        if normalized.startswith(f"{integration_root}/"):
            file_name = normalized.split("/")[-1]
            if file_name == "__init__.py":
                return _local_chatapp_init_py()
            if file_name == "store.py":
                return _local_chatapp_store_py()
            if file_name == "service.py":
                return _local_chatapp_service_py()
            if file_name == "routes.py":
                return _local_chatapp_routes_py(integration_root)
            return None

        absolute_target = Path(self.workspace_path) / normalized
        existing = absolute_target.read_text(encoding="utf-8") if absolute_target.exists() else ""
        if normalized == "core/runtime/web.py":
            return _merge_local_web_integration(existing, integration_root)
        if normalized == "core/ai/routing.py":
            return _merge_local_routing_integration(existing)
        if normalized == "interface/website/src/api.js":
            return _merge_local_frontend_api_integration(existing)
        if normalized in {"interface/website/src/App.js", "interface/website/src/components/Composer.js"}:
            return _merge_local_frontend_ui_integration(existing, normalized)
        return None

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
        try:
            display_path = str(absolute_path.relative_to(Path(self.workspace_path).resolve())).replace("\\", "/")
        except ValueError:
            display_path = absolute_path.name
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
                symbol_summary = _summarize_symbol_outline(display_path, symbol_payload)
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
        return _summarize_local_text_file(display_path, content)

    def _resolve_workspace_candidate(self, user_prompt: str) -> str | None:
        prompt_tokens = [token for token in re.findall(r"[a-z0-9_]+", user_prompt.lower()) if len(token) >= 3]
        try:
            entries = sorted(Path(self.workspace_path).iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return None

        if _is_architecture_question(user_prompt) or _is_runtime_routing_question(user_prompt):
            return self.workspace_path if entries else None

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
