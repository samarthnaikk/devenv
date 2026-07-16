from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from unittest import mock

from core.ai.models import AIExecutedToolStep, AIResponse, ToolCallRequest
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.models import EpisodicLog
from core.memory.storage import SQLiteMemoryStore
from core.memory.vector_index import InMemoryVectorIndex
from core.runtime import DevenvKernel
from core.runtime.context_builder import ContextBuilderService
from core.runtime.kernel import (
    _answer_from_retrieved_memory,
    _compose_external_memory_query,
    _find_reusable_tool_step,
    _memory_context_sections,
    _prefer_reference_results_over_empty_summary,
    _sanitize_logged_answer,
    _should_try_direct_memory_answer,
    _summarize_local_text_file,
    _summarize_directory_listing,
)
from core.runtime.local_model import FallbackLocalModel, SentenceTransformerLocalModel, load_local_small_model
from core.runtime.local_router import LocalRouteDecision
from core.runtime.models import ExternalSessionProviderConfig, PlanningMode, RuntimeTurnResult
from core.tools.edit_file import EditFileTool
from core.tools.inspect_symbols import InspectSymbolsTool
from core.tools.list_directory import ListDirectoryTool
from core.tools.locate_files import LocateFilesTool
from core.tools.manage_memory import ManageMemoryTool
from core.tools.peek_lines import PeekLinesTool
from core.tools.read_file import ReadFileTool
from core.tools.run_shell import RunShellTool
from core.tools.search_text import SearchTextTool
from core.tools.track_symbol import TrackSymbolTool
from core.tools.web_search import WebSearchTool
from core.tools.write_file import WriteFileTool
from core.runtime.models import ToolExecutionStep


@dataclass(frozen=True)
class FakeRetrievalResult:
    markdown_context: str


class FakeMemory:
    def __init__(self) -> None:
        self.working_memory_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.logs: list[tuple[str, str, dict[str, Any] | None]] = []
        self.consolidation_runs = 0

    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        self.working_memory_calls.append((messages, active_state))

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context=f"## Retrieved Memory\n- Prompt: {current_prompt}")

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.logs.append((user_prompt, agent_response, metadata))
        return "log-1"

    def run_consolidation(self):
        self.consolidation_runs += 1
        return type("Result", (), {"processed_logs": 1, "created_nodes": (), "updated_nodes": ()})()


class FailingMemory(FakeMemory):
    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        raise RuntimeError("memory offline")


class EmptyMemory(FakeMemory):
    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context="")


class ProjectMemory(FakeMemory):
    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(
            markdown_context="\n".join(
                [
                    "## Retrieved Memory",
                    "- get-drip was described as a Convex-backed app.",
                    "- The main issues were root URL redirects, Convex generated imports, and authentication bypass.",
                ]
            )
        )


class ProjectMemoryWithBugLog(ProjectMemory):
    def __init__(self) -> None:
        super().__init__()

        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="bug-log-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about get-drip bugs",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. Create Workspace accepting https links and converting them internally\n2. Salesforce being marked as coming soon or disabled\n3. The DRIP pipeline chat flow not working\n4. test/publish staying reachable after approvals\n5. root URL redirects\n6. Convex generated imports\n7. authentication bypass",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        self.store = FakeStore()


class FakeAI:
    def __init__(self, responses: list[AIResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.registered_tools: list[str] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.last_request_payload: dict[str, Any] | None = None

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AIResponse:
        self.chat_calls.append(
            {
                "messages": messages,
                "memory_context": memory_context,
                "temperature": temperature,
                "tool_names": sorted(tool_names) if tool_names is not None else None,
            }
        )
        return self.responses.pop(0)


class RateLimitedAI(FakeAI):
    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AIResponse:
        self.last_request_payload = {
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": memory_context or ""},
                *messages,
            ],
            "temperature": temperature,
            "tools": [],
            "tool_choice": "auto",
        }
        self.chat_calls.append(
            {
                "messages": messages,
                "memory_context": memory_context,
                "temperature": temperature,
                "tool_names": sorted(tool_names) if tool_names is not None else None,
            }
        )
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class ExplodingAI(FakeAI):
    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AIResponse:
        raise AssertionError("Local-only mode should not call the remote AI client")


class AccessDeniedAI(FakeAI):
    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AIResponse:
        raise RuntimeError("OpenCode backend access has not been granted.")


class TransportErrorAI(FakeAI):
    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> AIResponse:
        raise RuntimeError("OpenCode server failed: OpenCode server request failed with status 400.")


class DevenvKernelTest(unittest.TestCase):
    def test_find_reusable_tool_step_matches_successful_read_only_call(self) -> None:
        previous = ToolExecutionStep(
            step_id="step-1",
            tool_name="knowledge_search",
            arguments={"query": "chat app", "sources": ["github"]},
            output="knowledge_search gathered 2 resource(s)",
            success=True,
            is_sandboxed_violation=False,
            data={"query": "chat app", "resources": []},
        )

        reused = _find_reusable_tool_step([previous], "knowledge_search", {"query": "chat app", "sources": ["github"]})

        self.assertIs(reused, previous)

    def test_reference_results_replace_empty_knowledge_summary(self) -> None:
        step = ToolExecutionStep(
            step_id="step-1",
            tool_name="knowledge_search",
            arguments={"query": "chat app"},
            output="knowledge_search gathered 2 resource(s)",
            success=True,
            is_sandboxed_violation=False,
            data={
                "resources": [
                    {
                        "source": "github",
                        "results": [{"title": "chat-app/example", "url": "https://github.com/chat-app/example"}],
                    }
                ]
            },
        )

        response = _prefer_reference_results_over_empty_summary(
            "The search did not yield any relevant results.",
            [step],
            "I want to add a chat app to this codebase",
        )

        self.assertIn("Here are outside references", response)
        self.assertIn("https://github.com/chat-app/example", response)

    def test_direct_memory_answer_skips_repo_explanation_questions(self) -> None:
        self.assertFalse(_should_try_direct_memory_answer("how does retrieval work?"))
        self.assertFalse(_should_try_direct_memory_answer("can you explain how the retrieval works?"))
        self.assertFalse(_should_try_direct_memory_answer("how does this repo work?"))

    def test_sentence_transformer_local_model_falls_back_when_embedding_model_is_unavailable(self) -> None:
        model = SentenceTransformerLocalModel()
        with mock.patch("core.runtime.local_model._embedding_model", side_effect=OSError("offline")):
            selection = model.distill(
                "summarize get-drip bugs",
                [
                    "Create Workspace accepting https links and converting them internally",
                    "The DRIP pipeline chat flow not working",
                    "root URL redirects",
                ],
                max_lines=2,
            )

        self.assertTrue(selection.used_fallback)
        self.assertEqual(selection.model_name, "deterministic-fallback")
        self.assertTrue(selection.selected_lines)

    def test_load_local_small_model_defaults_to_deterministic_fallback(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVENV_USE_SENTENCE_TRANSFORMER_LOCAL_MODEL", None)
            model = load_local_small_model()

        self.assertIsInstance(model, FallbackLocalModel)

    def test_register_tool_syncs_runtime_and_ai(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())

        self.assertIn("read_file", kernel.tools)
        self.assertIn("list_directory", kernel.tools)
        self.assertEqual(ai.registered_tools, ["read_file", "list_directory"])

    def test_ollama_preference_disables_default_automatic_tool_scope(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())

            scope = kernel._resolve_direct_tool_scope("Explain the repo")

        self.assertEqual(scope, [])

    def test_ollama_preference_still_allows_execution_phase_tool_scope(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(EditFileTool())
            kernel.register_tool(WriteFileTool())

            scope = kernel._resolve_execution_tool_scope(
                "Add a light theme to this calendar app",
                "Update the styles and markup for a lighter visual design.",
            )

        self.assertIn("read_file", scope)
        self.assertIn("list_directory", scope)
        self.assertIn("edit_file", scope)

    def test_repair_tool_arguments_maps_missing_file_to_unique_workspace_suffix_match(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            frontend_dir = Path(tempdir) / "frontend"
            frontend_dir.mkdir(parents=True)
            (frontend_dir / "script.js").write_text("console.log('ok');\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

            repaired = kernel._repair_tool_arguments(
                ToolCallRequest(
                    call_id="call_1",
                    tool_name="read_file",
                    arguments={"path": "./src/main.js"},
                )
            )

        self.assertTrue(repaired["path"].endswith("frontend/script.js"))

    def test_repair_tool_arguments_normalizes_trailing_colon_keys_and_sets_write_mode(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "styles.css"
            target.write_text("body {}\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

            repaired = kernel._repair_tool_arguments(
                ToolCallRequest(
                    call_id="call_1",
                    tool_name="write_file",
                    arguments={
                        "path": str(target),
                        "content": "body { color: black; }\n",
                        "replace_block:": "noop",
                    },
                )
            )

        self.assertEqual(repaired["mode"], "overwrite")
        self.assertIn("replace_block", repaired)
        self.assertNotIn("replace_block:", repaired)

    def test_execution_scope_for_local_edit_prompt_does_not_add_web_search_from_current_wording(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(EditFileTool())
            kernel.register_tool(WriteFileTool())
            kernel.register_tool(WebSearchTool())

            scope = kernel._resolve_execution_tool_scope(
                "Add a light theme to this calendar app and update the UI accordingly",
                "Identify the current framework or technology used in the calendar app (e.g., React, Vue, Angular).",
            )

        self.assertIn("read_file", scope)
        self.assertIn("edit_file", scope)
        self.assertNotIn("web_search", scope)

    def test_execution_scope_for_mutation_checkpoint_does_not_add_web_search_without_explicit_browse_intent(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(EditFileTool())
            kernel.register_tool(WriteFileTool())
            kernel.register_tool(WebSearchTool())

            scope = kernel._resolve_execution_tool_scope(
                "Add a light theme to this calendar app and update the UI accordingly",
                "Identify resources or examples of light themes that can be applied to a calendar app UI.",
            )

        self.assertIn("read_file", scope)
        self.assertIn("edit_file", scope)
        self.assertNotIn("web_search", scope)

    def test_execution_scope_does_not_treat_research_word_as_web_search_intent(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(EditFileTool())
            kernel.register_tool(WriteFileTool())
            kernel.register_tool(WebSearchTool())

            scope = kernel._resolve_execution_tool_scope(
                "Add a light theme to this calendar app and update the UI accordingly",
                "Research how to implement a light theme in a calendar application.",
            )

        self.assertIn("read_file", scope)
        self.assertIn("edit_file", scope)
        self.assertNotIn("web_search", scope)

    def test_execution_scope_for_mutation_prompt_drops_search_text_to_avoid_looping(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(SearchTextTool())
            kernel.register_tool(EditFileTool())
            kernel.register_tool(WriteFileTool())

            scope = kernel._resolve_execution_tool_scope(
                "Add a light theme to this calendar app and update the UI accordingly",
                "Research how to implement a light theme in a calendar application.",
            )

        self.assertIn("read_file", scope)
        self.assertIn("edit_file", scope)
        self.assertNotIn("search_text", scope)

    def test_ollama_mutation_prompt_uses_local_planning_blueprint_shape(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        ai.preferred_backend = "ollama"

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            blueprint, _conversation, trace = kernel._checkpoint_creation_stage(
                user_prompt="Add a light theme to this calendar app and update the UI accordingly",
                memory_context="",
                continue_plan=False,
                local_only=False,
                planning_mode=PlanningMode.AUTO,
                steps=[],
                total_usage={},
                ai_logs=[],
                system_logs=[],
                max_consecutive_tools=5,
            )

        self.assertEqual(trace.summary, "Created ordered checkpoint blueprint")
        self.assertTrue(blueprint.tasks)
        self.assertIn("Inspect the relevant workspace files", blueprint.tasks[0].description)
        self.assertEqual(ai.chat_calls, [])

    def test_edit_prompt_is_not_treated_as_scaffold_request(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

            result = kernel._is_scaffold_request("add a light theme to this calendar app and update the ui accordingly")

        self.assertFalse(result)

    def test_themed_creation_prompt_is_still_treated_as_scaffold_request(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

            result = kernel._is_scaffold_request("create a dark theme frontend folder in calendar with html css and js")

        self.assertTrue(result)

    def test_repair_directory_path_recovers_nested_workspace_project(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            nested = Path(tempdir) / "sample-test" / "calendar"
            nested.mkdir(parents=True)
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

            repaired = kernel._repair_directory_path("/tmp/calendar_app")

        self.assertEqual(repaired, str(nested.resolve()))

    def test_execute_tool_call_repairs_directory_path_before_sandbox_check(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            nested = Path(tempdir) / "sample-test" / "calendar"
            nested.mkdir(parents=True)
            kernel = DevenvKernel(str(nested), memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())

            step = kernel._execute_tool_call(
                ToolCallRequest(
                    call_id="call_1",
                    tool_name="list_directory",
                    arguments={"path": "/workspace/MyProject", "mode": "recursive"},
                )
            )

        self.assertTrue(step.success)
        self.assertFalse(step.is_sandboxed_violation)
        self.assertEqual(step.arguments["path"], str(nested.resolve()))

    def test_execution_checkpoint_indexes_auto_advance_from_inspect_to_apply(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            blueprint = kernel._parse_markdown_to_blueprint(
                "\n".join(
                    [
                        "- [ ] Inspect the relevant workspace files for the requested change.",
                        "- [ ] Apply the requested update inside the matching file or folder.",
                        "- [ ] Verify the result in the workspace.",
                    ]
                ),
                original_objective="add light theme and update the UI",
            )

            indexes = kernel._execution_checkpoint_indexes(blueprint)

        self.assertEqual(indexes, [0, 1])

    def test_default_memory_paths_are_scoped_under_project_root(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

        workspace_root = Path(tempdir).resolve()
        self.assertEqual(kernel.db_path, str((workspace_root / "memory.db").resolve()))
        self.assertEqual(kernel.vector_dir, str((workspace_root / "vectors").resolve()))

    def test_execute_turn_returns_direct_ai_response(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Final answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            result = kernel.execute_turn("Explain the repo")

        self.assertEqual(result.final_response, "Final answer")
        self.assertEqual(result.total_usage["prompt_tokens"], 10)
        self.assertEqual(result.total_usage["completion_tokens"], 4)
        self.assertEqual(ai.chat_calls[0]["memory_context"], "## Context Packet\nPrompt: Explain the repo")
        self.assertEqual(ai.chat_calls[0]["tool_names"], [])
        self.assertIsNotNone(result.blueprint)
        self.assertTrue(result.blueprint.verification_passed)
        self.assertEqual(memory.logs[0][0], "Explain the repo")
        self.assertEqual(memory.logs[0][1], "Final answer")
        self.assertEqual(memory.logs[0][2]["workspace_path"], str(Path(tempdir).resolve()))
        self.assertEqual(memory.logs[0][2]["session_id"], kernel.session_id)
        self.assertEqual(memory.working_memory_calls[0][1]["session_id"], kernel.session_id)
        self.assertEqual(memory.consolidation_runs, 1)
        self.assertTrue(result.ai_logs)
        self.assertTrue(result.system_logs)

    def test_execute_turn_accepts_backend_executed_mcp_steps_for_codex(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Codex final answer",
                    finish_reason="stop",
                    usage={"prompt_tokens": 8, "completion_tokens": 3},
                    backend="codex",
                    executed_steps=(
                        AIExecutedToolStep(
                            step_id="step_1",
                            tool_name="read_file",
                            arguments={"path": "README.md"},
                            output='{"success": true, "output": "hello", "data": {}}',
                            success=True,
                        ),
                    ),
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            result = kernel.execute_turn("Explain the repo", backend_preference="codex", codex_enabled=True)

        self.assertEqual(result.final_response, "Codex final answer")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "read_file")
        self.assertEqual(result.steps[0].arguments["path"], "README.md")
        self.assertTrue(result.steps[0].success)

    def test_execute_turn_limits_tool_scope_to_selected_tools(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Searched answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 4, "completion_tokens": 2},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(WriteFileTool())
            result = kernel.execute_turn("Search the web for the docs", selected_tools=["read_file"])

        self.assertEqual(result.final_response, "Searched answer")
        self.assertEqual(ai.chat_calls[0]["tool_names"], ["read_file"])
        self.assertIn("User selected tools: read_file", result.system_logs)

    def test_execute_turn_answers_tool_strategy_question_locally_for_memory_recall(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("what tools do you need to answer what do you know about get-drip bugs")

        self.assertEqual(
            result.final_response,
            "For that question I would not need workspace tools first. Devenv should answer from memory/retrieval, and only fall back if prior context is not reliable enough.",
        )

    def test_execute_turn_answers_tool_strategy_question_locally_for_web_lookup(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(WebSearchTool())
            result = kernel.execute_turn("what tools do you need to answer search the latest docs for opencode")

        self.assertEqual(
            result.final_response,
            "For that question I would use `web_search` first, then answer from the retrieved results.",
        )

    def test_execute_turn_answers_bug_fix_follow_up_from_recent_conversation(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {
                    "role": "assistant",
                    "content": "In get-drip, the recalled bug list was:\n\n**Core product bugs**\n- Create Workspace accepting https links and converting them internally\n- Salesforce being marked as coming soon or disabled\n- the DRIP pipeline chat flow not working",
                }
            ]
            result = kernel.execute_turn("how did we fix those bugs")

        self.assertEqual(
            result.final_response,
            "Yes. In get-drip, we fixed those bugs by addressing Create Workspace accepting https links and converting them internally, Salesforce being marked as coming soon or disabled, and the DRIP pipeline chat flow not working.",
        )

    def test_execute_turn_answers_explain_it_follow_up_from_recent_conversation(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {
                    "role": "assistant",
                    "content": "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
                }
            ]
            result = kernel.execute_turn("can you explain about it")

        self.assertIn("It was mainly about root URL redirects", result.final_response or "")
        self.assertNotIn("main.py", result.final_response or "")

    def test_execute_turn_answers_what_are_those_follow_up_from_recent_conversation(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {
                    "role": "assistant",
                    "content": "In get-drip, the recalled bug list was:\n\n**Core product bugs**\n- Create Workspace accepting https links and converting them internally\n- Salesforce being marked as coming soon or disabled\n- the DRIP pipeline chat flow not working",
                }
            ]
            result = kernel.execute_turn("what are those")

        self.assertEqual(
            result.final_response,
            "Yes. In get-drip, the main issues were Create Workspace accepting https links and converting them internally, Salesforce being marked as coming soon or disabled, and the DRIP pipeline chat flow not working.",
        )

    def test_execute_turn_does_not_explain_memory_fallback_text(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {
                    "role": "assistant",
                    "content": "I couldn't recover a reliable prior answer for that yet.",
                }
            ]
            result = kernel.execute_turn("can you explain about it")

        self.assertEqual(result.final_response, "What should I explain? I don't have a clear prior subject in this thread yet.")

    def test_local_only_memory_recall_without_tools_uses_memory_fallback(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("what do you know about clean up schrema og get-drip", local_only=True)

        self.assertEqual(result.final_response, "I couldn't recover a reliable prior answer for that yet.")
        self.assertEqual(result.steps, [])

    def test_execute_turn_skips_external_context_fetch_when_local_memory_answer_is_ready(self) -> None:
        memory = FakeMemory()
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="cleanup-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schema of get-drip",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. root URL redirects\n2. Convex generated imports\n3. authentication bypass (critical)",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        memory.store = FakeStore()
        ai = ExplodingAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            result = kernel.execute_turn("what do you know about clean up schrema og get-drip")

        self.assertEqual(builder.calls, 0)
        self.assertEqual(
            result.final_response,
            "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_execute_turn_skips_external_context_for_current_workspace_repo_summary_prompt(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(markdown_context="")
        ai = FakeAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("summarize this repo")

        self.assertEqual(builder.calls, 0)
        self.assertEqual(
            result.metadata["external_context_reason"],
            "Skipped memory retrieval for a current-workspace inspection prompt.",
        )
        self.assertIn("README.md", result.final_response or "")

    def test_execute_turn_skips_all_memory_lookup_for_current_workspace_repo_prompt_when_tools_are_available(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))
        ai = FakeAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("Explain the repo")

        self.assertEqual(builder.calls, 0)
        self.assertEqual(
            result.metadata["external_context_reason"],
            "Skipped memory retrieval for a current-workspace inspection prompt.",
        )
        self.assertIn("README.md", result.final_response or "")

    def test_retrieve_memory_context_skips_low_context_acknowledgement_prompt(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))
        ai = FakeAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            memory_context, metadata = kernel._retrieve_memory_context("thanks")

        self.assertEqual(memory_context, "")
        self.assertEqual(builder.calls, 0)
        self.assertEqual(metadata["external_context_reason"], "Skipped memory retrieval for a low-context prompt.")

    def test_retrieve_memory_context_skips_shell_like_prompt(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))
        ai = FakeAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            memory_context, metadata = kernel._retrieve_memory_context("npm run dev")

        self.assertEqual(memory_context, "")
        self.assertEqual(builder.calls, 0)
        self.assertEqual(metadata["external_context_reason"], "Skipped memory retrieval for a low-context prompt.")

    def test_local_only_cleanup_schema_prompt_prefers_cleanup_session_over_validator_session(self) -> None:
        memory = EmptyMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "05"
            sessions_dir.mkdir(parents=True)
            cleanup_id = "session-cleanup"
            validators_id = "session-validators"
            (codex_root / "session_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": cleanup_id, "thread_name": "Clean up schema", "updated_at": "2026-07-05T12:00:00Z"}),
                        json.dumps({"id": validators_id, "thread_name": "Explain validators", "updated_at": "2026-07-05T12:05:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-05T12-00-00-{cleanup_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-05T12:00:00Z", "type": "session_meta", "payload": {"id": cleanup_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-05T12:00:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "The schema cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-05T12-05-00-{validators_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-05T12:05:00Z", "type": "session_meta", "payload": {"id": validators_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-05T12:05:01Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "They are Convex argument validators defined in shared.ts for mock CRM sources."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                tempdir,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = kernel.execute_turn("what do you know about clean up schrema og get-drip", local_only=True)

        self.assertIn("root URL redirects", result.final_response or "")
        self.assertNotIn("validators", (result.final_response or "").lower())

    def test_build_tool_client_defaults_to_in_process_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ExplodingAI([]))
            kernel.register_tool(ListDirectoryTool())
            with mock.patch.dict(os.environ, {}, clear=False):
                client = kernel._build_tool_client(db_path=kernel.db_path, vector_dir=kernel.vector_dir)

        self.assertEqual(client.__class__.__name__, "_InProcessToolClient")

    def test_build_tool_client_uses_mcp_transport_when_requested(self) -> None:
        fake_client = object()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ExplodingAI([]))
            with mock.patch.dict(os.environ, {"DEVENV_TOOL_TRANSPORT": "mcp"}, clear=False):
                with mock.patch("core.runtime.mcp_client.MCPToolClient", return_value=fake_client) as client_mock:
                    client = kernel._build_tool_client(db_path=kernel.db_path, vector_dir=kernel.vector_dir)

        self.assertIs(client, fake_client)
        self.assertEqual(client_mock.call_count, 1)

    def test_execute_turn_appends_external_session_context_to_memory(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="I found the prior project session.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 8},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "20"
            sessions_dir.mkdir(parents=True)
            session_id = "session-project-1"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Project Atlas review", "updated_at": "2026-06-20T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                json.dumps({"session_id": session_id, "ts": 1, "text": "We worked on Project Atlas ingestion and review fixes."}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-20T09-59-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-20T09:59:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-06-20T09:59:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "Do you know Project Atlas?"}}),
                        json.dumps({"timestamp": "2026-06-20T09:59:02Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Project Atlas had an ingestion path and review fixes."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                str(workspace),
                memory=memory,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            kernel.local_router = _disabled_router()
            kernel.execute_turn("Do you know about Project Atlas?")

        memory_context = ai.chat_calls[0]["memory_context"] or ""
        self.assertIn("## Context Packet", memory_context)
        self.assertIn("Project Atlas", memory_context)

    def test_execute_turn_skips_repeat_consolidation_within_cooldown(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(content="First answer", tool_calls=(), finish_reason="stop", usage={"prompt_tokens": 4}),
                AIResponse(content="Second answer", tool_calls=(), finish_reason="stop", usage={"prompt_tokens": 4}),
            ]
        )
        previous = os.environ.get("DEVENV_CONSOLIDATION_COOLDOWN_SECONDS")
        os.environ["DEVENV_CONSOLIDATION_COOLDOWN_SECONDS"] = "900"
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
                kernel.local_router = _disabled_router()
                kernel.execute_turn("Explain the repo")
                kernel.execute_turn("Explain the repo again")
        finally:
            if previous is None:
                os.environ.pop("DEVENV_CONSOLIDATION_COOLDOWN_SECONDS", None)
            else:
                os.environ["DEVENV_CONSOLIDATION_COOLDOWN_SECONDS"] = previous

        self.assertEqual(memory.consolidation_runs, 1)

    def test_local_only_mode_can_answer_from_external_session_context(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "20"
            sessions_dir.mkdir(parents=True)
            session_id = "session-project-2"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Project Atlas review", "updated_at": "2026-06-20T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                json.dumps({"session_id": session_id, "ts": 1, "text": "Project Atlas had an ingestion path and review fixes."}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-20T09-59-00-{session_id}.jsonl").write_text(
                json.dumps({"timestamp": "2026-06-20T09:59:02Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "Project Atlas had an ingestion path and review fixes."}})
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                str(workspace),
                memory=memory,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = kernel.execute_turn("Do you know about Project Atlas?", local_only=True)

        self.assertIn("Project Atlas", result.final_response or "")

    def test_direct_tool_scope_starts_with_no_tools_for_memory_question(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(SearchTextTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(WriteFileTool())
            kernel.register_tool(RunShellTool())
            kernel.register_tool(WebSearchTool())
            kernel.register_tool(ManageMemoryTool(FakeMemory()))

            scope = kernel._resolve_direct_tool_scope("what do you know about get-drip?")

        self.assertEqual(scope, [])

    def test_direct_tool_scope_offers_web_search_for_current_docs_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(WebSearchTool())

            scope = kernel._resolve_direct_tool_scope("search the latest docs for opencode")

        self.assertEqual(scope, ["web_search"])

    def test_direct_tool_scope_keeps_web_search_available_for_ollama_latest_web_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI([])
            ai.preferred_backend = "ollama"
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=ai)
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(WebSearchTool())

            scope = kernel._resolve_direct_tool_scope("what is elon musk's net worth, latest you can search online")

        self.assertEqual(scope, ["web_search"])

    def test_direct_tool_scope_offers_knowledge_search_for_reference_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(WebSearchTool())
            from core.tools.knowledge_search import KnowledgeSearchTool

            kernel.register_tool(KnowledgeSearchTool())

            scope = kernel._resolve_direct_tool_scope("find similar github repos and stackoverflow references for a calendar feature")

        self.assertEqual(scope, ["knowledge_search"])

    def test_direct_tool_scope_uses_compact_code_inspection_set_for_backend_question(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(LocateFilesTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            kernel.register_tool(SearchTextTool())
            kernel.register_tool(TrackSymbolTool())
            kernel.register_tool(WebSearchTool())
            kernel.register_tool(RunShellTool())

            scope = kernel._resolve_direct_tool_scope("how does the backend work?")

        self.assertEqual(scope, ["inspect_symbols", "list_directory", "peek_lines", "read_file"])

    def test_direct_tool_scope_uses_compact_repo_summary_set(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(LocateFilesTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            kernel.register_tool(SearchTextTool())
            kernel.register_tool(TrackSymbolTool())

            scope = kernel._resolve_direct_tool_scope("summarize this repo")

        self.assertEqual(scope, ["inspect_symbols", "list_directory", "peek_lines", "read_file"])

    def test_tool_strategy_question_uses_actual_repo_summary_retrieval_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())

            answer = kernel._answer_tool_strategy_question("what tools do you need to answer summarize this repo")

        self.assertIn("list_directory", answer or "")
        self.assertIn("read_file", answer or "")
        self.assertIn("inspect_symbols", answer or "")
        self.assertNotIn("peek_lines", answer or "")

    def test_tool_strategy_question_uses_actual_backend_retrieval_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())

            answer = kernel._answer_tool_strategy_question("what tools do you need to answer how does the backend work")

        self.assertIn("list_directory", answer or "")
        self.assertIn("inspect_symbols", answer or "")
        self.assertNotIn("peek_lines", answer or "")

    def test_direct_turn_sends_compact_code_inspection_scope_for_backend_question(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="The backend uses routes and services.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 5, "completion_tokens": 2},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(LocateFilesTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            kernel.register_tool(SearchTextTool())
            kernel.register_tool(TrackSymbolTool())
            kernel.register_tool(WebSearchTool())
            kernel.register_tool(RunShellTool())
            result = kernel.execute_turn("how does the backend work?")

        self.assertEqual(result.final_response, "The backend uses routes and services.")
        self.assertEqual(ai.chat_calls[0]["tool_names"], ["inspect_symbols", "list_directory", "peek_lines", "read_file"])

    def test_direct_turn_sends_no_tools_for_project_summary_prompt(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Get-drip was a Convex-backed app with campaign-related fixes.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 4, "completion_tokens": 2},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(SearchTextTool())
            result = kernel.execute_turn("what can be said confidently about get-drip?")

        self.assertEqual(result.final_response, "Get-drip was a Convex-backed app with campaign-related fixes.")
        self.assertEqual(ai.chat_calls[0]["tool_names"], [])

    def test_local_only_confident_getdrip_prompt_uses_memory_synthesis_not_exact_log_replay(self) -> None:
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=ProjectMemory(), ai=ai)
            result = kernel.execute_turn("what can be said confidently about get-drip?", local_only=True)

        self.assertEqual(
            result.final_response,
            "Confidently, get-drip was described as a Convex-backed app, and the work focused on root URL redirects, Convex generated imports, and authentication bypass. What remains unclear is a cleaner one-line architecture summary beyond those clues.",
        )
        self.assertEqual(result.steps, [])

    def test_local_only_getdrip_backend_prompt_uses_memory_synthesis(self) -> None:
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=ProjectMemory(), ai=ai)
            result = kernel.execute_turn("what was the backend of get-drip?", local_only=True)

        self.assertEqual(result.final_response, "get-drip was described as a Convex-backed app.")
        self.assertEqual(result.steps, [])

    def test_local_only_synthesis_prompt_ignores_exact_bug_log_in_favor_of_memory_facts(self) -> None:
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=ProjectMemoryWithBugLog(), ai=ai)
            result = kernel.execute_turn("what can be said confidently about get-drip?", local_only=True)

        self.assertIn("Convex-backed app", result.final_response or "")
        self.assertNotIn("Core product bugs", result.final_response or "")
        self.assertEqual(result.steps, [])

    def test_local_only_generic_backend_question_prefers_workspace_over_unrelated_memory(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- RVIDA backend is not a widely known or standard term."
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# docs\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does the backend work?", local_only=True)

        self.assertIn("main orchestrator", result.final_response or "")
        self.assertNotIn("RVIDA backend is not a widely known", result.final_response or "")
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_local_only_retrieval_question_prefers_workspace_over_recalled_chat_text(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context=(
                "## Retrieved Memory\n"
                "- The retrieval-memory aspect of this repository involves tracing through code paths related to memory retrieval and documenting each file's role.\n"
            )
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text(
                "def execute_turn(prompt):\n"
                "    return prompt\n\n"
                "def _retrieve_memory_context(prompt):\n"
                "    return prompt\n",
                encoding="utf-8",
            )
            (runtime_dir / "context_builder.py").write_text(
                "class ContextBuilderService:\n"
                "    def build_runtime_memory_context(self, prompt):\n"
                "        return prompt\n",
                encoding="utf-8",
            )
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does retrieval of memory work?", local_only=True)

        self.assertIn("I inspected the backend entry points locally.", result.final_response or "")
        self.assertNotIn("documenting each file's role", result.final_response or "")
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_local_only_repo_summary_prefers_workspace_over_recalled_summary_text(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- Now I have a comprehensive picture of the repo. Let me summarize it."
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("summarize this repo", local_only=True)

        self.assertIn("README.md", result.final_response or "")
        self.assertNotIn("Now I have a comprehensive picture", result.final_response or "")
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_local_knowledge_route_does_not_use_unrelated_memory_for_generic_backend_question(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- RVIDA backend is not a widely known or standard term."
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does the backend work?")

        self.assertIn("main orchestrator", result.final_response or "")
        self.assertIn("AI backend routing", result.final_response or "")
        self.assertNotIn("RVIDA backend is not a widely known", result.final_response or "")
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_resolve_workspace_candidate_avoids_fuzzy_docs_match_for_generic_backend_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            (Path(tempdir) / "docs").mkdir()
            (Path(tempdir) / "core").mkdir()
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

            candidate = kernel._resolve_workspace_candidate("how does the backend work?")

        self.assertEqual(candidate, str(Path(tempdir).resolve()))

    def test_execution_tool_scope_adds_mutation_and_shell_tools_only_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(WriteFileTool())
            kernel.register_tool(RunShellTool())

            scope = kernel._resolve_execution_tool_scope(
                "fix the backend and run tests",
                "Edit the relevant file and run diagnostics",
            )

        self.assertIn("write_file", scope)
        self.assertIn("run_shell", scope)
        self.assertIn("read_file", scope)

    def test_local_only_mode_can_answer_from_external_tool_output_context(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "20"
            sessions_dir.mkdir(parents=True)
            session_id = "session-review-1"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Review notes", "updated_at": "2026-06-20T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-06-20T09-59-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-06-20T09:59:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/tmp/other"}}),
                        json.dumps({"timestamp": "2026-06-20T09:59:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "What did Sharmil say?"}}),
                        json.dumps({"timestamp": "2026-06-20T09:59:02Z", "type": "response_item", "payload": {"type": "function_call_output", "output": "Sharmil001 | Comment: use a convex action instead of the frontend route."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                str(workspace),
                memory=memory,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            result = kernel.execute_turn("What were the issues Sharmil was talking about?", local_only=True)

        self.assertIn("convex action", (result.final_response or "").lower())

    def test_local_only_mode_recalls_multiple_named_projects_from_old_sessions(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "30"
            sessions_dir.mkdir(parents=True)
            fixtures = [
                (
                    "session-opencode",
                    "Integrate Codex and OpenCode",
                    "Do you know about OpenCode?",
                    "OpenCode sessions should be read and chunked for context building.",
                ),
                (
                    "session-supabase",
                    "Add Supabase auth sign in/up",
                    "Do you know about Supabase?",
                    "Supabase email sign in and sign up were added with migrations in mind.",
                ),
                (
                    "session-verticalaxis",
                    "Update VerticalAxis header colors",
                    "Do you know about VerticalAxis?",
                    "VerticalAxis needed header color updates in the LaTeX file.",
                ),
            ]

            index_lines: list[str] = []
            for session_id, title, query, answer in fixtures:
                index_lines.append(json.dumps({"id": session_id, "thread_name": title, "updated_at": "2026-06-30T10:00:00Z"}))
                (sessions_dir / f"rollout-2026-06-30T10-00-00-{session_id}.jsonl").write_text(
                    "\n".join(
                        [
                            json.dumps({"timestamp": "2026-06-30T10:00:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                            json.dumps({"timestamp": "2026-06-30T10:00:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": query}}),
                            json.dumps({"timestamp": "2026-06-30T10:00:02Z", "type": "event_msg", "payload": {"type": "agent_message", "message": answer}}),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (codex_root / "session_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                str(workspace),
                memory=memory,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            opencode = kernel.execute_turn("Do you know about OpenCode?", local_only=True)
            supabase = kernel.execute_turn("Do you know about Supabase?", local_only=True)
            vertical_axis = kernel.execute_turn("Do you know about VerticalAxis?", local_only=True)
            no_match = kernel.execute_turn("Do you know about Project Atlas?", local_only=True)

        self.assertIn("opencode", (opencode.final_response or "").lower())
        self.assertIn("supabase", (supabase.final_response or "").lower())
        self.assertIn("verticalaxis", (vertical_axis.final_response or "").lower())
        self.assertNotIn("opencode", (no_match.final_response or "").lower())
        self.assertNotIn("supabase", (no_match.final_response or "").lower())

    def test_execute_turn_uses_planning_for_change_requests(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="- [ ] Create README.md",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 5},
                ),
                AIResponse(
                    content="Done",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"completion_tokens": 2},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Create a README.md")

        self.assertEqual(ai.chat_calls[0]["tool_names"], [])
        self.assertIsNotNone(result.blueprint)
        self.assertEqual(result.blueprint.tasks[0].description, "Create README.md")

    def test_direct_turn_recovers_inline_tool_json_and_continues(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=(
                        "To inspect the backend, I need to list the directory.\n\n"
                        '[{"name":"list_directory","parameters":{"path":"backend","mode":"recursive","max_depth":2}}]'
                    ),
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 4},
                ),
                AIResponse(
                    content="The backend has routes and services.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"completion_tokens": 3},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            backend_path = Path(tempdir) / "backend"
            backend_path.mkdir()
            (backend_path / "routes.py").write_text("print('ok')", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("how does the backend work?")

        self.assertEqual(result.final_response, "The backend has routes and services.")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "list_directory")
        self.assertIn("Recovered inline tool request: list_directory", result.ai_logs)

    def test_local_knowledge_route_answers_from_memory_without_ai(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- [episode] This repo uses a FastAPI backend and a job-processing service."
        )
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            result = kernel.execute_turn("how does the repo work?")

        self.assertIn("FastAPI backend", result.final_response)

    def test_repo_summary_prompt_routes_to_local_knowledge(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("summarize this repo")

        self.assertEqual(len(ai.chat_calls), 0)
        self.assertEqual(result.steps[0].tool_name, "list_directory")
        self.assertIn("README.md", result.final_response or "")
        self.assertNotIn("env.py", result.final_response or "")
        self.assertEqual(ai.chat_calls, [])
        self.assertIn("## Project Overview", result.final_response or "")

    def test_repo_overview_prompt_prefers_repo_summary_shape_over_backend_only_shape(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does the repo work?")

        self.assertIn("README.md", result.final_response or "")
        self.assertIn("kernel.py", result.final_response or "")
        self.assertIn("## Project Overview", result.final_response or "")
        self.assertNotIn("I inspected the backend entry points locally.", result.final_response or "")

    def test_project_about_prompt_routes_to_local_repo_overview(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo project\nThis project helps inspect local code.\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("what is this project about")

        self.assertEqual(ai.chat_calls, [])
        self.assertIn("## Project Overview", result.final_response or "")
        self.assertIn("README.md", result.final_response or "")

    def test_backend_connector_prompt_routes_to_local_architecture_answer(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            ai_dir = Path(tempdir) / "core" / "ai"
            runtime_dir = Path(tempdir) / "core" / "runtime"
            ai_dir.mkdir(parents=True)
            runtime_dir.mkdir(parents=True)
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (ai_dir / "codex_backend.py").write_text("class CodexAICore:\n    pass\n", encoding="utf-8")
            (ai_dir / "ollama_backend.py").write_text("class OllamaAICore:\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("can we integrate claude code in this")

        self.assertEqual(ai.chat_calls, [])
        self.assertIn("## Backend Connector Fit", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertIn("claude_backend.py", result.final_response or "")

    def test_backend_connector_follow_up_prompt_stays_local_and_actionable(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            ai_dir = Path(tempdir) / "core" / "ai"
            runtime_dir = Path(tempdir) / "core" / "runtime"
            ai_dir.mkdir(parents=True)
            runtime_dir.mkdir(parents=True)
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (ai_dir / "codex_backend.py").write_text("class CodexAICore:\n    pass\n", encoding="utf-8")
            (ai_dir / "ollama_backend.py").write_text("class OllamaAICore:\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn(
                "no like how codex and opencode are options as backend (like the thinking part) not their UI or website, like that I want for claude also"
            )

        self.assertEqual(ai.chat_calls, [])
        self.assertIn("## Backend Connector Fit", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertIn("Codex", result.final_response or "")
        self.assertIn("Ollama", result.final_response or "")
        self.assertTrue(any(path.endswith("core/runtime/web.py") for path in result.metadata.get("files_touched", [])))

    def test_tell_me_about_repo_prompt_uses_current_workspace_summary_not_prior_memory(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- The commit is in\n- requirements.txt is the main entrypoint."
        )
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("tell me about this repo")

        self.assertIn("README.md", result.final_response or "")
        self.assertIn("kernel.py", result.final_response or "")
        self.assertNotIn("The commit is in", result.final_response or "")

    def test_tell_me_about_backend_prompt_prefers_backend_entry_points(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            (Path(tempdir) / "requirements.txt").write_text("lancedb>=0.16.0\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("tell me about the backend")

        self.assertIn("kernel.py", result.final_response or "")
        self.assertIn("web.py", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertNotIn("requirements.txt", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")

    def test_tool_strategy_parser_handles_how_do_you_decide_tool_prompt(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how do you decide what tools to use for tell me about the backend?")

        self.assertIn("Devenv should stay in charge of retrieval", result.final_response or "")
        self.assertIn("inspect_symbols", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")

    def test_system_prompt_prefers_current_workspace_architecture_summary(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- An unrelated old session said the system used persona scoring."
        )
        ai = TransportErrorAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does the system work?")

        self.assertIn("kernel.py", result.final_response or "")
        self.assertIn("web.py", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertNotIn("persona scoring", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")

    def test_tell_me_about_system_prompt_prefers_current_workspace_architecture_summary(self) -> None:
        memory = FakeMemory()
        ai = TransportErrorAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("tell me about the system")

        self.assertIn("kernel.py", result.final_response or "")
        self.assertIn("web.py", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")

    def test_what_is_backend_prompt_prefers_current_workspace_architecture_summary(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- An unrelated old session said the backend used persona scoring."
        )
        ai = TransportErrorAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("what is the backend?")

        self.assertIn("kernel.py", result.final_response or "")
        self.assertIn("web.py", result.final_response or "")
        self.assertIn("routing.py", result.final_response or "")
        self.assertNotIn("persona scoring", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")

    def test_repo_summary_prompt_uses_opencode_to_summarize_bounded_local_evidence_when_enabled(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Devenv AI is a local-first coding agent runtime with memory, local tools, and OpenCode-backed routing.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
                )
            ]
        )
        ai.opencode_enabled = True

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text(
                "# Devenv AI\n\nDevenv AI is a local-first coding agent foundation for running project-aware workflows on your machine.\n",
                encoding="utf-8",
            )
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("summarize this repo", opencode_enabled=True)

        self.assertEqual(
            result.final_response,
            "Devenv AI is a local-first coding agent runtime with memory, local tools, and OpenCode-backed routing.",
        )
        self.assertEqual(len(ai.chat_calls), 1)
        self.assertEqual(ai.chat_calls[0]["tool_names"], [])
        self.assertIn("## Local Workspace Evidence", ai.chat_calls[0]["memory_context"] or "")
        self.assertIn("README.md", ai.chat_calls[0]["memory_context"] or "")
        self.assertIn("kernel.py", ai.chat_calls[0]["memory_context"] or "")

    def test_repo_summary_prompt_falls_back_to_local_answer_when_opencode_synthesis_fails(self) -> None:
        memory = FakeMemory()
        ai = TransportErrorAI([])
        ai.opencode_enabled = True

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            (Path(tempdir) / "README.md").write_text("# Devenv AI\n\nDevenv AI is a local-first coding agent foundation.\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(PeekLinesTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("summarize this repo", opencode_enabled=True)

        self.assertIn("README.md", result.final_response or "")
        self.assertIn("kernel.py", result.final_response or "")

    def test_readme_summary_is_humanized_for_repo_answers(self) -> None:
        summary = _summarize_local_text_file(
            "README.md",
            "\n".join(
                [
                    "# Devenv AI",
                    "",
                    "Devenv AI is a local-first coding agent foundation for running project-aware workflows on your machine.",
                    "It combines a runtime layer with a persistent Cognitive Memory Engine (CME).",
                ]
            ),
        )

        self.assertIn("local-first coding agent foundation", summary or "")
        self.assertIn("persistent Cognitive Memory Engine", summary or "")
        self.assertNotIn("Preview:", summary or "")

    def test_repo_work_prompt_routes_to_local_architecture_summary(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does the repo work?")

        self.assertIn("main orchestrator", result.final_response or "")
        self.assertIn("local web backend", result.final_response or "")
        self.assertNotIn("OpenCode backend access is not granted right now", result.final_response or "")
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_local_knowledge_route_can_inspect_workspace_without_ai(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            backend_path = Path(tempdir) / "rvidia1a"
            backend_path.mkdir()
            (backend_path / "server.py").write_text("print('server')", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("tell me about rvidia")

        self.assertIn("I inspected", result.final_response)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "list_directory")

    def test_local_knowledge_route_prefers_named_project_inspection_for_code_level_question(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            getgit_path = Path(tempdir) / "getgit"
            getgit_path.mkdir()
            (getgit_path / "core.py").write_text("def answer_query():\n    pass\n\ndef main():\n    pass\n", encoding="utf-8")
            (getgit_path / "README.md").write_text(
                "# GetGit\n\nRepository intelligence system using RAG.\n",
                encoding="utf-8",
            )
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does getgit decide what content to send to ai?")

        self.assertIn("README.md", result.final_response or "")
        self.assertIn("core.py", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")
        self.assertEqual([step.tool_name for step in result.steps], ["list_directory", "read_file", "inspect_symbols"])

    def test_local_knowledge_route_rejects_schema_fragment_memory_for_code_level_question(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context=(
                '## Retrieved Memory\n'
                '- [episode] ",\\"type\\":\\"string\\"}},\\"required\\":[\\"content\\",\\"content\\",\\"\\"],\\"type\\":\\"object\\"}]"'
            )
        )
        ai = FakeAI(
            [
                AIResponse(
                    content="GetGit decides what content to send by chunking and retrieval.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 5},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            getgit_path = Path(tempdir) / "getgit"
            getgit_path.mkdir()
            (getgit_path / "core.py").write_text("def answer_query():\n    pass\n\ndef main():\n    pass\n", encoding="utf-8")
            (getgit_path / "README.md").write_text(
                "# GetGit\n\nRepository intelligence system using RAG.\n",
                encoding="utf-8",
            )
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("how does getgit decide what content to send to ai?")

        self.assertIn("README.md", result.final_response or "")
        self.assertIn("core.py", result.final_response or "")
        self.assertEqual(result.metadata["backend_used"], "local")
        self.assertEqual([step.tool_name for step in result.steps], ["list_directory", "read_file", "inspect_symbols"])

    def test_execute_turn_answers_tool_strategy_question_before_local_knowledge_routing(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(ReadFileTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("what tools would you use to answer how does backend work?")

        self.assertIn("Devenv should stay in charge of retrieval", result.final_response or "")
        self.assertIn("`list_directory`", result.final_response or "")
        self.assertIn("`inspect_symbols`", result.final_response or "")

    def test_summarize_directory_listing_parses_structured_payload_without_leaking_json(self) -> None:
        summary = _summarize_directory_listing(
            "/tmp/getgit",
            'list_directory completed for /tmp/getgit in recursive mode\n'
            '{"path":"/tmp/getgit","mode":"recursive","entries":['
            '{"relative_path":"rag","is_dir":true},'
            '{"relative_path":"core.py","is_dir":false},'
            '{"relative_path":"templates/index.html","is_dir":false}'
            "]}",
        )

        self.assertIn("rag, core.py, templates/index.html", summary)
        self.assertNotIn('{"relative_path"', summary)

    def test_answer_from_retrieved_memory_rejects_low_signal_directory_dump(self) -> None:
        answer = _answer_from_retrieved_memory(
            "how does rag work in getgit",
            "\n".join(
                [
                    "## Retrieved Memory",
                    '- [episode] I inspected `/tmp/getgit` locally. Relevant paths I found: .git"}, {"relative_path": "rag"}',
                ]
            ),
        )

        self.assertIsNone(answer)

    def test_answer_from_retrieved_memory_prefers_project_specific_recall(self) -> None:
        answer = _answer_from_retrieved_memory(
            "do you remember about that get-drip project?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Session 'Project work' matched via Codex history.",
                    "- Targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip for retrieval and prompt generation work.",
                    "- Assistant reported: get-drip was the project where we were improving retrieval quality across stored sessions.",
                    "- Assistant reported: Sharmil's reviews needed follow-up.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("Yes", answer or "")
        self.assertIn("get-drip", answer or "")
        self.assertNotIn("Sharmil", answer or "")

    def test_answer_from_retrieved_memory_prefers_issue_summary_over_session_path_for_named_project(self) -> None:
        answer = _answer_from_retrieved_memory(
            "hey, do you remember about get-drip project?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Session 'Fix 7 bugs' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip.",
                    "- User asked: Create Workspace -> accept the https link and convert it internally.",
                    "- User asked: DRIP pipeline chat does not work and test/publish should be reachable after approvals.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("get-drip", answer or "")
        self.assertIn("Create Workspace accepting https links", answer or "")
        self.assertNotIn("Session 'Fix 7 bugs'", answer or "")

    def test_answer_from_retrieved_memory_uses_working_memory_for_follow_up(self) -> None:
        answer = _answer_from_retrieved_memory(
            "we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
            "\n".join(
                [
                    "## Working Memory",
                    "- user: hey, do you remember about get-drip project?",
                    "- assistant: Yes. Session 'rollout-1' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip.",
                    "## External Session Context",
                    "- Assistant reported: get-drip had review feedback to support workspace creation links and fix the DRIP pipeline chat flow.",
                    "- Assistant reported: get-drip still had bugs around root URL redirects and Convex generated imports.",
                    '- {"type": "function", "name": "list_directory", "parameters": {"path": "/Users/samarthnaik/Desktop/LoopedIn/get-dri", "mode": "recursive"}}',
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("Yes", answer or "")
        self.assertIn("get-drip", answer or "")
        self.assertIn("Create Workspace", answer or "")
        self.assertIn("pipeline chat", answer or "")
        self.assertNotIn('"type": "function"', answer or "")

    def test_answer_from_retrieved_memory_follow_up_prefers_workspace_subject_over_generic_hyphenated_token(self) -> None:
        answer = _answer_from_retrieved_memory(
            "we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
            "\n".join(
                [
                    "## Working Memory",
                    "- assistant: Yes. get-drip came up in prior sessions.",
                    "## External Session Context",
                    "- Assistant reported: I’m grounding in the app first so we can turn those 7 bug notes into a decision-complete fix plan.",
                    "- Session 'Fix 7 bugs' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip.",
                    "- User asked: Create Workspace -> accept the https link and convert it internally.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("In get-drip", answer or "")
        self.assertNotIn("decision-complete", answer or "")

    def test_answer_from_retrieved_memory_rejects_progress_status_for_named_project_recall(self) -> None:
        answer = _answer_from_retrieved_memory(
            "hey, do you remember about get-drip project?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: I’ve confirmed the Convex schema entrypoints. Next I’m reading the main schema and scanning code references.",
                    "- Session 'Fix 7 bugs' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip.",
                    "- User asked: Create Workspace -> accept the https link and convert it internally.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("get-drip", answer or "")
        self.assertIn("Create Workspace accepting https links", answer or "")
        self.assertNotIn("Convex schema entrypoints", answer or "")

    def test_answer_from_retrieved_memory_prefers_external_context_over_generic_retrieved_memory(self) -> None:
        answer = _answer_from_retrieved_memory(
            "we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] Episodic Memory 8ef3e3bc: we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
                    "## External Session Context",
                    "- User asked: Create Workspace -> accept the https link and convert it internally.",
                    "- User asked: DRIP pipeline chat does not work and test/publish should be reachable after approvals.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("Create Workspace accepting https links", answer or "")
        self.assertIn("DRIP pipeline chat flow not working", answer or "")
        self.assertNotIn("Episodic Memory", answer or "")
        self.assertNotIn("User asked:", answer or "")

    def test_answer_from_retrieved_memory_formats_bug_list_as_markdown_sections(self) -> None:
        answer = _answer_from_retrieved_memory(
            "give get-drip bug list",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: get-drip had review feedback to support workspace creation links and fix the DRIP pipeline chat flow.",
                    "- Assistant reported: get-drip still had bugs around root URL redirects and Convex generated imports.",
                    "- Assistant reported: ISSUE-001 Critical Security Authentication bypass.",
                    "- Assistant reported: ISSUE-002 Critical Security Open email relay.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("In get-drip, the recalled bug list was:", answer or "")
        self.assertIn("**Core product bugs**", answer or "")
        self.assertIn("- authentication bypass", answer or "")
        self.assertIn("**PR review security findings**", answer or "")

    def test_execute_turn_uses_direct_memory_answer_for_bug_list_prompt(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: get-drip had review feedback to support workspace creation links and fix the DRIP pipeline chat flow.",
                    "- Assistant reported: get-drip still had bugs around root URL redirects and Convex generated imports.",
                ]
            )
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("give get-drip bug list", local_only=False)

        self.assertIn("recalled bug list", result.final_response or "")
        self.assertIn("Core product bugs", result.final_response or "")

    def test_exact_logged_answer_does_not_reuse_file_list_answer_for_bug_list_prompt(self) -> None:
        class FakeStore:
            def search_logs_for_external_query(self, query: str, limit: int = 5) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="bad-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "The strongest clues point to src/convex-types.ts, src/convex-api.ts, and src/routes/workspace.$workspaceId.tsx.",
                                "metadata": {"external_context_query": "infer the parts of the app in get-drip"},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("give get-drip bug list")

        self.assertIsNone(result)

    def test_exact_logged_answer_sanitizes_replay_json_into_readable_answer(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return [
                    "\n".join(
                        [
                            json.dumps({"type": "step_start", "timestamp": 1}),
                            json.dumps(
                                {
                                    "type": "text",
                                    "part": {
                                        "type": "text",
                                        "text": "Based on the codebase, here's the get-drip bug list:\n\n**Core product bugs**\n- The DRIP pipeline chat flow not working",
                                    },
                                }
                            ),
                            json.dumps(
                                {
                                    "type": "text",
                                    "part": {
                                        "type": "text",
                                        "text": "Based on the codebase, here's the get-drip bug list:\n\n**Core product bugs**\n- The DRIP pipeline chat flow not working",
                                    },
                                }
                            ),
                        ]
                    )
                ]

            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                return []

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("give get-drip bug list")

        self.assertEqual(
            result,
            "Based on the codebase, here's the get-drip bug list:\n\n**Core product bugs**\n- The DRIP pipeline chat flow not working",
        )

    def test_retrieve_memory_context_uses_recent_conversation_for_follow_up_matching(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "02"
            sessions_dir.mkdir(parents=True)
            session_id = "session-get-drip"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Fix 7 bugs", "updated_at": "2026-07-02T11:52:11Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-02T13-12-12-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-02T13:12:12Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:13Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "get-drip still had bugs around root URL redirects and Convex generated imports."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {"role": "user", "content": "hey, do you remember about get-drip project?"},
                {"role": "assistant", "content": "Yes. Session 'rollout-1' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip."},
            ]
            kernel.context_builder = ContextBuilderService(
                tempdir,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            memory_context, metadata = kernel._retrieve_memory_context(
                "we had a few reviews and bugs to be fixed? can you tell exactly what were those?"
            )

        self.assertIn("get-drip", memory_context)
        self.assertEqual(metadata["external_context_state"], "reused_prior_context")
        self.assertIn("Referenced context:", metadata["external_context_query"])
        self.assertIn("get-drip", metadata["external_context_query"])

    def test_retrieve_memory_context_does_not_treat_named_project_question_as_follow_up(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "24"
            sessions_dir.mkdir(parents=True)
            session_id = "session-codeguide"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Integrate CodeGuide with GetGit", "updated_at": "2026-04-24T19:16:23Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-04-24T19-16-23-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-04-24T19:16:23Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/work/hirex-frontend"}}),
                        json.dumps({"timestamp": "2026-04-24T19:16:24Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "CodeGuide was about integrating its flow with GetGit and ai_services without duplicating logic."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {"role": "user", "content": "hey, do you remember about get-drip project?"},
                {"role": "assistant", "content": "Yes. Session 'rollout-1' targeted workspace /Users/samarthnaik/Desktop/LoopedIn/get-drip."},
            ]
            kernel.context_builder = ContextBuilderService(
                tempdir,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            _memory_context, metadata = kernel._retrieve_memory_context("do you remember codeguide? what was it about?")

        self.assertEqual(metadata["external_context_state"], "reused_prior_context")
        self.assertEqual(metadata["external_context_query"], "do you remember codeguide? what was it about?")

    def test_local_only_follow_up_memory_question_stays_in_direct_answer_mode(self) -> None:
        memory = EmptyMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "02"
            sessions_dir.mkdir(parents=True)
            session_id = "session-get-drip"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Fix 7 bugs", "updated_at": "2026-07-02T11:52:11Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-02T13-12-12-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-02T13:12:12Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:13Z", "type": "event_msg", "payload": {"type": "user_message", "message": "Create Workspace should accept the https link and convert it internally."}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:14Z", "type": "event_msg", "payload": {"type": "user_message", "message": "DRIP pipeline chat does not work and test/publish should be reachable after approvals."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                tempdir,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )
            kernel.execute_turn("hey, do you remember about get-drip project?", local_only=True)
            result = kernel.execute_turn(
                "we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
                local_only=True,
            )

        self.assertIsNotNone(result.final_response)
        self.assertIn("Create Workspace accepting https links", result.final_response or "")
        self.assertIn("DRIP pipeline chat flow not working", result.final_response or "")

    def test_memory_recall_and_follow_up_use_direct_answer_mode_in_normal_app_flow(self) -> None:
        memory = EmptyMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            codex_root = Path(tempdir) / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "02"
            sessions_dir.mkdir(parents=True)
            session_id = "session-get-drip"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Fix 7 bugs", "updated_at": "2026-07-02T11:52:11Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-02T13-12-12-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-02T13:12:12Z", "type": "session_meta", "payload": {"id": session_id, "cwd": "/Users/samarthnaik/Desktop/LoopedIn/get-drip"}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:13Z", "type": "event_msg", "payload": {"type": "user_message", "message": "Create Workspace should accept the https link and convert it internally. Salesforce should be marked coming soon or disabled."}}),
                        json.dumps({"timestamp": "2026-07-02T13:12:14Z", "type": "event_msg", "payload": {"type": "user_message", "message": "DRIP pipeline chat does not work and test/publish should be reachable after approvals."}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.context_builder = ContextBuilderService(
                tempdir,
                provider_configs=(
                    ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                ),
            )

            first = kernel.execute_turn("hey, do you remember about get-drip project?", local_only=False)
            second = kernel.execute_turn(
                "we had a few reviews and bugs to be fixed? can you tell exactly what were those?",
                local_only=False,
            )

        self.assertIn("get-drip", first.final_response or "")
        self.assertIn("Create Workspace", second.final_response or "")
        self.assertIn("DRIP pipeline chat", second.final_response or "")

    def test_answer_from_retrieved_memory_humanizes_project_summary(self) -> None:
        answer = _answer_from_retrieved_memory(
            "do you remember codeguide? what was it about?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: The first sweep shows there is already a candidate-practice CodeGuide backend and worker plumbing in place, including a `task_practice_code_evaluate` path that calls `task_getgit_checkpoints`. I’m narrowing in on what’s incomplete rather than layering on a parallel one.",
                ]
            ),
        )

        self.assertIsNotNone(answer)
        self.assertIn("It was about", answer or "")
        self.assertIn("candidate-practice CodeGuide backend", answer or "")
        self.assertNotIn("The first sweep shows", answer or "")
        self.assertNotIn("I’m narrowing", answer or "")

    def test_answer_from_retrieved_memory_humanizes_explain_it_follow_up(self) -> None:
        answer = _answer_from_retrieved_memory(
            "can you explain about it?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: get-drip was about cleaning up schema-related retrieval and app flow issues.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "Yes. It was about cleaning up schema-related retrieval and app flow issues.",
        )

    def test_shape_logged_project_answer_humanizes_getdrip_bug_list(self) -> None:
        memory = FakeMemory()
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what do you know about get-drip bugs":
                    return [
                        "\n".join(
                            [
                                "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:",
                                "",
                                "1. Create Workspace accepting https links and converting them internally",
                                "2. Salesforce being marked as coming soon or disabled",
                                "3. The DRIP pipeline chat flow not working",
                                "4. test/publish staying reachable after approvals",
                                "5. root URL redirects",
                                "6. Convex generated imports",
                                "7. authentication bypass (critical)",
                            ]
                        )
                    ]
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return []

        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about get-drip bugs")

        self.assertIn("In get-drip, the recalled bug list was:", result or "")
        self.assertNotIn("Based on the code in", result or "")

    def test_answer_from_retrieved_memory_explains_bug_fix_follow_up(self) -> None:
        answer = _answer_from_retrieved_memory(
            "how did we fix those bugs",
            "\n".join(
                [
                    "## Working Memory",
                    "- assistant: Yes. get-drip came up in prior sessions.",
                    "## External Session Context",
                    "- User asked: Create Workspace -> accept the https link and convert it internally.",
                    "- Assistant reported: get-drip still had bugs around root URL redirects and Convex generated imports.",
                    "- Assistant reported: the DRIP pipeline chat flow did not work and Salesforce was marked as coming soon.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "Yes. In get-drip, we fixed those bugs by addressing Create Workspace accepting https links and converting them internally, Salesforce being marked as coming soon or disabled, the DRIP pipeline chat flow not working, root URL redirects, and Convex generated imports.",
        )

    def test_clean_memory_follow_up_rejects_transcript_dump_lines(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what were those",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: Q. do you know about get-drip bugs A. Yes. Because we fixed correctness/review issues. Devenv status Tool trace 14s OpenCode Prepared the final answer.",
                    "- Assistant reported: Create Workspace -> accept the https link and convert it internally.",
                    "- Assistant reported: the DRIP pipeline chat flow did not work.",
                ]
            ),
        )

        self.assertNotIn("Devenv status", answer or "")
        self.assertNotIn("Tool trace", answer or "")

    def test_follow_up_prefers_recent_working_memory_summary_over_unrelated_retrieved_memory(self) -> None:
        answer = _answer_from_retrieved_memory(
            "can you explain about it",
            "\n".join(
                [
                    "## Working Memory",
                    "- assistant: The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
                    "## Retrieved Memory",
                    "- [episode] can you explain how the main.py works | The `main.py` file is a Python script that sets up a simple HTTP server.",
                ]
            ),
        )

        self.assertIn("It was mainly about root URL redirects", answer or "")
        self.assertNotIn("main.py", answer or "")

    def test_retrieve_memory_context_anchors_lexical_follow_up_to_recent_subject(self) -> None:
        class FakeStore:
            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                if "get-drip" in terms:
                    return [
                        EpisodicLog(
                            log_id="anchored-1",
                            timestamp=2.0,
                            associated_node_id=None,
                            raw_interaction=json.dumps(
                                {
                                    "user": "what do you know about clean up schema of get-drip",
                                    "agent": "The get-drip cleanup was mainly about root URL redirects and Convex generated imports.",
                                }
                            ),
                        )
                    ]
                return [
                    EpisodicLog(
                        log_id="unanchored-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "can you explain how the main.py works",
                                "agent": "The `main.py` file is a Python script that sets up a simple HTTP server.",
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            kernel.ephemeral_history = [
                {"role": "user", "content": "what do you know about clean up schema of get-drip"},
                {"role": "assistant", "content": "The get-drip cleanup was mainly about root URL redirects and Convex generated imports."},
            ]
            context, _metadata = kernel._retrieve_memory_context("can you explain about it")

        self.assertIn("get-drip cleanup", context)
        self.assertNotIn("main.py", context)

    def test_answer_from_retrieved_memory_rejects_single_file_memory_for_repo_summary_prompt(self) -> None:
        answer = _answer_from_retrieved_memory(
            "Explain the repo",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] can you explain how the main.py works | The `main.py` file is a Python script that sets up a simple HTTP server to serve the demo calendar app.",
                ]
            ),
        )

        self.assertIsNone(answer)

    def test_execute_turn_clarifies_ambiguous_follow_up_without_memory_lookup(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("can you explain about it")

        self.assertEqual(
            result.final_response,
            "What should I explain? I don't have a clear prior subject in this thread yet.",
        )

    def test_compose_external_memory_query_anchors_explain_it_follow_up_to_prior_subject(self) -> None:
        query = _compose_external_memory_query(
            "can you explain about it?",
            [
                {"role": "user", "content": "what do you know about clean up schema of get-drip"},
                {"role": "assistant", "content": "Yes. get-drip was about cleaning up schema-related retrieval and app flow issues."},
            ],
        )

        self.assertIn("can you explain about it?", query)
        self.assertIn("Referenced context:", query)
        self.assertIn("get-drip", query)

    def test_compose_external_memory_query_prefers_recent_user_subject_over_assistant_file_clues(self) -> None:
        query = _compose_external_memory_query(
            "can you explain about it",
            [
                {"role": "user", "content": "what do you know about clean up schrema og get-drip"},
                {
                    "role": "assistant",
                    "content": "Yes. The strongest clues point to src/convex-types.ts, src/convex-api.ts, src/routes/workspace.$workspaceId.campaigns.$campaignId.test-activate.tsx, src/routes/workspace.$workspaceId.campaigns.$campaignId.pipeline.tsx, src/routes/workspace.$workspaceId.tsx.",
                },
            ],
        )

        self.assertIn("Referenced context:", query)
        self.assertIn("get-drip", query)
        self.assertIn("Referenced context: get-drip", query)

    def test_answer_from_retrieved_memory_does_not_duplicate_yes_prefix(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what do you know about get-drip?",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: Yes. get-drip was the project where we were improving retrieval quality across stored sessions.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "Yes. get-drip was the project where we were improving retrieval quality across stored sessions.",
        )

    def test_sanitize_logged_answer_strips_tool_output_dump_and_qa_wrapper(self) -> None:
        cleaned = _sanitize_logged_answer(
            "\n".join(
                [
                    "Q. do you know about get-drip bugs",
                    "A. Yes.",
                    "",
                    "Here are the reviewer answers, based on the current code in [`reviews.md`](/tmp/reviews.md):",
                    "",
                    "1. The review findings are still open.",
                    "",
                    "Tool output: v.literal(\"rejected\"), v.literal(\"needs_review\")",
                ]
            )
        )

        self.assertTrue(cleaned.startswith("Yes."))
        self.assertIn("Here are the reviewer answers", cleaned)
        self.assertNotIn("Tool output:", cleaned)
        self.assertNotIn("Q. do you know", cleaned)
        self.assertNotIn("<proposed_plan>", cleaned)

    def test_sanitize_logged_answer_strips_inline_ui_trace_noise(self) -> None:
        cleaned = _sanitize_logged_answer(
            "Yes. Because we fixed correctness issues. Devenv status Tool trace 14s OpenCode Prepared the final answer."
        )

        self.assertEqual(cleaned, "Yes. Because we fixed correctness issues.")

    def test_sanitize_logged_answer_collapses_wrapped_duplicate_blocks(self) -> None:
        cleaned = _sanitize_logged_answer(
            "\n\n".join(
                [
                    "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                    "Yes.\n\nYes. Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                ]
            )
        )

        self.assertEqual(cleaned, "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.")

    def test_sanitize_logged_answer_extracts_last_assistant_block_from_ui_transcript_dump(self) -> None:
        cleaned = _sanitize_logged_answer(
            "\n".join(
                [
                    "You",
                    "",
                    "do you know about get-drip bugs",
                    "",
                    "Devenv status",
                    "Tool trace",
                    "14s",
                    "OpenCode",
                    "⚡",
                    "Prepared the final answer",
                    "TracePrepared the final answer",
                    "Devenv",
                    "",
                    "Yes. Because we fixed correctness/review issues.",
                    "",
                    "You",
                    "",
                    "what do you know about clean up schrema og get-drip",
                    "",
                    "Devenv status",
                    "Tool trace",
                    "2s",
                    "OpenCode",
                    "⚡",
                    "Prepared the final answer",
                    "TracePrepared the final answer",
                    "Devenv",
                    "",
                    "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                ]
            )
        )

        self.assertEqual(cleaned, "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.")

    def test_runtime_turn_result_sanitizes_final_response_and_error_message(self) -> None:
        result = RuntimeTurnResult(
            final_response=(
                "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.\n\n"
                "Yes.\n\n"
                "Yes. Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts."
            ),
            error_message="OpenCode server failed: bad request. Devenv status Tool trace 14s",
        )

        self.assertEqual(
            result.final_response,
            "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
        )
        self.assertEqual(result.error_message, "OpenCode server failed: bad request.")

    def test_sanitize_logged_answer_strips_leaked_proposed_plan_block(self) -> None:
        cleaned = _sanitize_logged_answer(
            "\n".join(
                [
                    "Yes.",
                    "",
                    "Here are the reviewer answers, based on the current code in [`reviews.md`](/tmp/reviews.md).",
                    "",
                    "<proposed_plan>",
                    "# Schema Cleanup Plan",
                    "</proposed_plan>",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "Yes.\n\nHere are the reviewer answers, based on the current code in [`reviews.md`](/tmp/reviews.md).",
        )

    def test_exact_logged_answer_does_not_reuse_file_clue_answer_for_generic_getdrip_recall(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="clue-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "The strongest clues point to src/convex-types.ts, src/convex-api.ts, src/routes/workspace.$workspaceId.tsx.",
                                "metadata": {"external_context_query": "infer the parts of the app in get-drip"},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about get-drip?")

        self.assertIsNone(result)

    def test_exact_logged_answer_does_not_reuse_file_clue_answer_for_schema_cleanup_prompt(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="clue-2",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "The strongest clues point to src/convex-types.ts, src/convex-api.ts, src/routes/workspace.$workspaceId.tsx.",
                                "metadata": {"external_context_query": "infer the parts of the app in get-drip"},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about clean up schema of get-drip?")

        self.assertIsNone(result)

    def test_exact_logged_answer_does_not_reuse_same_prompt_broad_log_replay(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="self-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schema of get-drip",
                                "agent": "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about clean up schema of get-drip")

        self.assertIsNone(result)

    def test_exact_logged_answer_reuses_same_prompt_when_logged_answer_is_usable(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="self-usable-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schema of get-drip",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. root URL redirects\n2. Convex generated imports\n3. authentication bypass (critical)",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about clean up schrema og get-drip")

        self.assertEqual(
            result,
            "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_answer_from_retrieved_memory_rejects_file_clue_answer_for_schema_cleanup_prompt(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what do you know about clean up schema of get-drip",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] what do you know about clean up schrema og get-drip | Yes. The strongest clues point to `src/convex-types.ts`, `src/convex-api.ts`.",
                ]
            ),
        )

        self.assertIsNone(answer)

    def test_lookup_exact_logged_answer_humanizes_schema_cleanup_bug_summary(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what do you know about clean up schema of get-drip":
                    return [
                        "\n".join(
                            [
                                "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:",
                                "",
                                "1. root URL redirects",
                                "2. Convex generated imports",
                                "3. authentication bypass (critical)",
                            ]
                        )
                    ]
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return []

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about clean up schema of get-drip")

        self.assertEqual(
            result,
            "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_lookup_exact_logged_answer_humanizes_generic_getdrip_recall(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what do you know about get-drip":
                    return [
                        "\n".join(
                            [
                                "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:",
                                "",
                                "1. Create Workspace accepting https links and converting them internally",
                                "2. Salesforce being marked as coming soon or disabled",
                                "3. The DRIP pipeline chat flow not working",
                            ]
                        )
                    ]
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return []

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about get-drip?")

        self.assertEqual(
            result,
            "Yes. In get-drip, the main issues were Create Workspace accepting https links and converting them internally, Salesforce being marked as coming soon or disabled, and the DRIP pipeline chat flow not working.",
        )

    def test_lookup_exact_logged_answer_normalizes_schema_cleanup_typo_variant(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what do you know about clean up schema of get-drip":
                    return [
                        "\n".join(
                            [
                                "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:",
                                "",
                                "1. root URL redirects",
                                "2. Convex generated imports",
                                "3. authentication bypass (critical)",
                            ]
                        )
                    ]
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return []

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what do you know about clean up schrema og get-drip")

        self.assertEqual(
            result,
            "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_cleanup_follow_up_stays_narrow_to_cleanup_specific_issues(self) -> None:
        answer = _answer_from_retrieved_memory(
            "can you explain about it",
            "\n".join(
                [
                    "## Working Memory",
                    "- assistant: The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "Yes. It was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_answer_from_retrieved_memory_humanizes_cleanup_narrative_summary(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what do you know about clean up schrema og get-drip",
            "\n".join(
                [
                    "## External Session Context",
                    "- Assistant reported: The repo points to a conservative default: there are very few whole tables that are obviously dead, but there are some strong legacy-column cleanup candidates in `campaigns` and `crmCustomers`, plus a few audit/cache/helper tables that are easy to misclassify as unused because they have narrow call sites.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "The get-drip cleanup was mainly about removing legacy columns and duplicated state, focusing the schema pass on `campaigns` and `crmCustomers`, keeping the cleanup conservative instead of deleting whole tables, and keeping audit, cache, and helper tables that still support live flows.",
        )

    def test_execute_turn_answers_cleanup_explain_follow_up_from_recent_conversation(self) -> None:
        memory = FailingMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.ephemeral_history = [
                {
                    "role": "assistant",
                    "content": "The get-drip cleanup was mainly about removing legacy columns and duplicated state, focusing the schema pass on `campaigns` and `crmCustomers`, keeping the cleanup conservative instead of deleting whole tables, and keeping audit, cache, and helper tables that still support live flows.",
                }
            ]
            result = kernel.execute_turn("can you explain about it")

        self.assertIn("removing legacy columns and duplicated state", result.final_response or "")
        self.assertNotIn("main.py", result.final_response or "")

    def test_typo_cleanup_prompt_is_treated_as_cleanup_specific(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what do you know about clean up schrema og get-drip",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] what do you know about clean up schema of get-drip | Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. root URL redirects\n2. Convex generated imports\n3. authentication bypass (critical)",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_memory_context_sections_preserve_multiline_retrieved_entries(self) -> None:
        sections = _memory_context_sections(
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] schema cleanup | Based on the code review:",
                    "",
                    "1. root URL redirects",
                    "2. Convex generated imports",
                ]
            )
        )

        self.assertEqual(
            sections["retrieved"],
            [
                "[episode] schema cleanup | Based on the code review:\n1. root URL redirects\n2. Convex generated imports",
            ],
        )

    def test_retrieve_memory_context_skips_vector_lookup_for_session_history_questions(self) -> None:
        class RecallOnlyMemory(FakeMemory):
            def __init__(self) -> None:
                super().__init__()
                self.store = type(
                    "Store",
                    (),
                    {
                        "search_logs": lambda self, terms, limit=20: [],
                    },
                )()

            def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
                raise AssertionError("session-history recall should not hit vector retrieval")

        memory = RecallOnlyMemory()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            context, metadata = kernel._retrieve_memory_context("what was the last merge conflict we solved")

        self.assertEqual(context, "")
        self.assertEqual(metadata["external_context_state"], "new_context")

    def test_answer_from_retrieved_memory_handles_merge_conflict_recall(self) -> None:
        answer = _answer_from_retrieved_memory(
            "what was the last merge conflict we solved",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] we fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling.",
                ]
            ),
        )

        self.assertEqual(
            answer,
            "we fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling.",
        )

    def test_execute_turn_returns_memory_only_fallback_for_unresolved_session_history_question(self) -> None:
        class RecallOnlyMemory(FakeMemory):
            def __init__(self) -> None:
                super().__init__()
                self.store = type(
                    "Store",
                    (),
                    {
                        "search_logs": lambda self, terms, limit=20: [],
                    },
                )()

            def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
                raise AssertionError("session-history recall should not hit vector retrieval")

        memory = RecallOnlyMemory()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("what was the last merge conflict we solved")

        self.assertEqual(
            result.final_response,
            "I couldn't recover a reliable note about the last merge conflict we solved.",
        )
        self.assertEqual(memory.logs, [])

    def test_lookup_exact_logged_answer_supports_session_history_recall(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what was the last merge conflict we solved":
                    return [
                        "We fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling."
                    ]
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return []

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel._lookup_exact_logged_answer("what was the last merge conflict we solved")

        self.assertEqual(
            result,
            "We fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling.",
        )

    def test_execute_turn_uses_pre_retrieval_fast_path_for_session_history_recall(self) -> None:
        class RecallOnlyMemory(FakeMemory):
            def __init__(self) -> None:
                super().__init__()
                self.store = type(
                    "Store",
                    (),
                    {
                        "search_agent_responses_for_external_query": lambda self, query, limit=8: [
                            "We fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling."
                        ]
                        if query == "what was the last merge conflict we solved"
                        else [],
                        "search_logs": lambda self, terms, limit=20: [],
                    },
                )()

            def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
                raise AssertionError("session-history exact recall should skip retrieval")

        memory = RecallOnlyMemory()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("what was the last merge conflict we solved")

        self.assertEqual(
            result.final_response,
            "We fixed the last merge conflict by reconciling the OpenCode fallback retry path with the structured output handling.",
        )
        self.assertEqual(memory.working_memory_calls, [])

    def test_do_you_know_about_getdrip_bugs_uses_bug_list_recall(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="bug-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about get-drip bugs",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. Create Workspace accepting https links and converting them internally\n2. Salesforce being marked as coming soon or disabled\n3. The DRIP pipeline chat flow not working\n4. test/publish staying reachable after approvals\n5. root URL redirects\n6. Convex generated imports\n7. authentication bypass (critical)",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("do you know about get-drip bugs")

        self.assertIn("In get-drip, the recalled bug list was:", result.final_response or "")

    def test_execute_turn_uses_pre_retrieval_fast_path_for_do_you_know_bug_variant(self) -> None:
        class RecallOnlyMemory(FakeMemory):
            def __init__(self) -> None:
                super().__init__()
                self.store = type(
                    "Store",
                    (),
                    {
                        "search_agent_responses_for_external_query": lambda self, query, limit=8: [
                            "\n".join(
                                [
                                    "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:",
                                    "",
                                    "1. Create Workspace accepting https links and converting them internally",
                                    "2. Salesforce being marked as coming soon or disabled",
                                    "3. The DRIP pipeline chat flow not working",
                                ]
                            )
                        ]
                        if query == "what do you know about get-drip bugs"
                        else [],
                        "search_logs": lambda self, terms, limit=20: [],
                    },
                )()

            def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
                raise AssertionError("bug-list exact recall should skip retrieval")

        memory = RecallOnlyMemory()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("do you know about get-drip bugs")

        self.assertIn("In get-drip, the recalled bug list was:", result.final_response or "")
        self.assertEqual(memory.working_memory_calls, [])

    def test_retrieve_lexical_memory_context_compacts_cleanup_prompt_once_answer_is_supported(self) -> None:
        class FakeStore:
            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="cleanup-compact-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schema of get-drip",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. root URL redirects\n2. Convex generated imports\n3. authentication bypass (critical)",
                            }
                        ),
                    ),
                    EpisodicLog(
                        log_id="cleanup-compact-2",
                        timestamp=2.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schema of get-drip",
                                "agent": "Yes. The strongest clues point to `src/convex-types.ts`, `src/convex-api.ts`.",
                            }
                        ),
                    ),
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            context = kernel._retrieve_lexical_memory_context("what do you know about clean up schrema og get-drip")

        self.assertIn("root URL redirects", context)
        self.assertNotIn("strongest clues point to", context.lower())

    def test_retrieve_lexical_memory_context_compacts_bug_prompt_once_answer_is_supported(self) -> None:
        class FakeStore:
            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="bug-compact-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about get-drip bugs",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n1. Create Workspace accepting https links and converting them internally\n2. Salesforce being marked as coming soon or disabled\n3. The DRIP pipeline chat flow not working",
                            }
                        ),
                    ),
                    EpisodicLog(
                        log_id="bug-compact-2",
                        timestamp=2.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about get-drip bugs",
                                "agent": "Based on the code in `core/runtime/kernel.py:2940-2980`, the 7 bugs tracked for get-drip are:\n\n4. test/publish staying reachable after approvals\n5. root URL redirects",
                            }
                        ),
                    ),
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            context = kernel._retrieve_lexical_memory_context("do you know about get-drip bugs")

        self.assertIn("Create Workspace", context)
        self.assertNotIn("test/publish", context)

    def test_execute_turn_prefers_structured_project_answer_before_generic_memory_recall(self) -> None:
        class FakeStore:
            def search_agent_responses_for_external_query(self, query: str, limit: int = 8) -> list[str]:
                if query == "what do you know about clean up schema of get-drip":
                    return [
                        "The cleanup was about root URL redirects, Convex generated imports, and review follow-ups."
                    ]
                return []

            def search_logs_for_external_query(self, query: str, limit: int = 8) -> list[EpisodicLog]:
                return []

            def search_logs(self, terms: list[str], limit: int = 20) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="clue-3",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "user": "what do you know about clean up schrema og get-drip",
                                "agent": "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                                "metadata": {},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("what do you know about clean up schema of get-drip")

        self.assertEqual(
            result.final_response,
            "The get-drip cleanup was mainly about root URL redirects and Convex generated imports.",
        )

    def test_local_directory_summary_is_not_persisted_to_episodic_memory(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            backend_path = Path(tempdir) / "rvidia1a"
            backend_path.mkdir()
            (backend_path / "server.py").write_text("print('server')", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = type(
                "Router",
                (),
                {
                    "decide": lambda self, prompt: LocalRouteDecision(
                        use_local_knowledge=True,
                        confidence=0.7,
                        knowledge_score=0.8,
                        remote_score=0.1,
                        reason="test",
                    )
                },
            )()
            kernel.register_tool(ListDirectoryTool())
            kernel.execute_turn("tell me about rvidia")

        self.assertEqual(memory.logs, [])

    def test_backend_entrypoint_summary_is_not_persisted_to_episodic_memory(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            runtime_dir = Path(tempdir) / "core" / "runtime"
            runtime_dir.mkdir(parents=True)
            ai_dir = Path(tempdir) / "core" / "ai"
            ai_dir.mkdir(parents=True)
            (runtime_dir / "kernel.py").write_text("def execute_turn():\n    pass\n", encoding="utf-8")
            (runtime_dir / "web.py").write_text("class DevenvWebApp:\n    pass\n", encoding="utf-8")
            (ai_dir / "routing.py").write_text("class RoutingAICore:\n    pass\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            kernel.register_tool(InspectSymbolsTool())
            result = kernel.execute_turn("what is the backend?")

        self.assertIn("I inspected the backend entry points locally.", result.final_response or "")
        self.assertEqual(memory.logs, [])

    def test_repair_directory_path_fixes_missing_inline_guess(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            (Path(tempdir) / "rvidia").mkdir()
            kernel = DevenvKernel(tempdir, memory=FakeMemory(), ai=FakeAI([]))

            repaired = kernel._repair_directory_path(f"{tempdir}/rvidia1a")

        self.assertTrue(repaired.endswith("rvidia"))

    def test_execute_turn_runs_registered_tool(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_1",
                            tool_name="read_file",
                            arguments={"path": "note.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 7},
                ),
                AIResponse(
                    content="I read the file.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"completion_tokens": 4},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            note_path = f"{tempdir}/note.txt"
            with open(note_path, "w", encoding="utf-8") as handle:
                handle.write("hello runtime")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ReadFileTool())
            result = kernel.execute_turn("Read note.txt")

        self.assertEqual(result.final_response, "I read the file.")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "read_file")
        self.assertTrue(result.steps[0].success)
        self.assertIn("read_file completed", result.steps[0].output)
        self.assertIn("hello runtime", result.steps[0].output)
        second_call_messages = ai.chat_calls[1]["messages"]
        self.assertEqual(second_call_messages[-1]["role"], "tool")
        self.assertIn("read_file", ai.chat_calls[1]["tool_names"])
        self.assertEqual(result.total_usage["prompt_tokens"], 7)
        self.assertEqual(result.total_usage["completion_tokens"], 4)

    def test_follow_up_turn_does_not_replay_old_tool_output_history(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_1",
                            tool_name="read_file",
                            arguments={"path": "note.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 7},
                ),
                AIResponse(
                    content="I read the file.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"completion_tokens": 4},
                ),
                AIResponse(
                    content="Follow-up answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 3},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            note_path = f"{tempdir}/note.txt"
            with open(note_path, "w", encoding="utf-8") as handle:
                handle.write("hello runtime")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            kernel.register_tool(ReadFileTool())
            kernel.execute_turn("Read note.txt")
            kernel.execute_turn("What happened?")

        follow_up_messages = ai.chat_calls[2]["messages"]
        self.assertEqual([message["role"] for message in follow_up_messages], ["system", "user"])

    def test_execute_turn_flags_sandbox_violation(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_unsafe",
                            tool_name="read_file",
                            arguments={"path": "../secrets.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content="I could not access that path.",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            result = kernel.execute_turn("Read ../secrets.txt")

        self.assertTrue(result.steps[0].is_sandboxed_violation)
        self.assertIn("Sandbox violation", result.steps[0].output)
        self.assertEqual(result.final_response, "I could not access that path.")

    def test_execute_turn_caps_tool_iterations(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(call_id="call_1", tool_name="missing_tool", arguments={}),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(call_id="call_2", tool_name="missing_tool", arguments={}),
                    ),
                    finish_reason="tool_calls",
                    usage={},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            result = kernel.execute_turn("Loop", max_consecutive_tools=1)

        self.assertEqual(result.error_message, "Direct tool limit reached before the request could be completed.")
        self.assertEqual(result.blueprint.tasks[0].child_checkpoint_ids, (2, 3))
        self.assertEqual(result.blueprint.tasks[1].description, "Gather the context required for: Loop")

    def test_execute_turn_continues_when_memory_retrieval_fails(self) -> None:
        memory = FailingMemory()
        ai = FakeAI(
            [
                AIResponse(
                    content="Fallback answer",
                    tool_calls=(),
                    finish_reason="stop",
                    usage={"prompt_tokens": 2},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            result = kernel.execute_turn("Explain the repo")

        self.assertEqual(result.final_response, "Fallback answer")
        self.assertEqual(ai.chat_calls[0]["memory_context"], "")

    def test_execute_turn_returns_local_repo_summary_when_opencode_access_is_denied(self) -> None:
        memory = EmptyMemory()
        ai = AccessDeniedAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            repo_root = Path(tempdir)
            (repo_root / "src").mkdir()
            (repo_root / "README.md").write_text("# Demo repo\n", encoding="utf-8")
            (repo_root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("Explain the repo")

        self.assertIsNotNone(result.final_response)
        self.assertIn("Relevant paths I found", result.final_response or "")
        self.assertIsNone(result.error_message)

    def test_execute_turn_returns_clear_message_when_opencode_access_is_denied_for_non_repo_prompt(self) -> None:
        memory = EmptyMemory()
        ai = AccessDeniedAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Why does this fail?")

        self.assertEqual(
            result.final_response,
            "What is failing? Share the command, error message, file, or step that is breaking so I can trace it accurately.",
        )
        self.assertIsNone(result.error_message)

    def test_execute_turn_returns_local_candidate_summary_when_opencode_access_is_denied_for_named_folder_prompt(self) -> None:
        memory = EmptyMemory()
        ai = AccessDeniedAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            candidate_root = Path(tempdir) / "rvidia1a"
            candidate_root.mkdir()
            (candidate_root / "server.py").write_text("print('server')\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("tell me about rvidia")

        self.assertIn("OpenCode backend access is not granted right now", result.final_response or "")
        self.assertIn("Relevant paths I found", result.final_response or "")
        self.assertIn("server.py", result.final_response or "")
        self.assertEqual(result.error_message, "OpenCode backend access has not been granted.")
        self.assertEqual(memory.logs, [])

    def test_execute_turn_returns_local_candidate_summary_when_opencode_access_is_denied_for_code_level_named_project_prompt(self) -> None:
        memory = EmptyMemory()
        ai = AccessDeniedAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            project_root = Path(tempdir) / "getgit"
            project_root.mkdir()
            (project_root / "core.py").write_text("print('core')\n", encoding="utf-8")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("how does getgit decide what content to send to ai?")

        self.assertIn("Relevant paths I found", result.final_response or "")
        self.assertIn("core.py", result.final_response or "")
        self.assertIsNone(result.error_message)

    def test_execute_turn_returns_local_repo_summary_when_opencode_transport_fails(self) -> None:
        memory = EmptyMemory()
        ai = TransportErrorAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            Path(tempdir, "README.md").write_text("# Demo repo\n", encoding="utf-8")
            Path(tempdir, "src").mkdir()
            Path(tempdir, "src", "app.py").write_text("print('hi')\n", encoding="utf-8")
            kernel.register_tool(ListDirectoryTool())
            result = kernel.execute_turn("Explain the repo")

        self.assertIn("Relevant paths I found", result.final_response or "")
        self.assertIn("OpenCode server request failed with status 400.", result.error_message or "")

    def test_execute_turn_clarifies_underspecified_troubleshooting_prompt_without_memory_or_ai(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))
        ai = ExplodingAI([])

        class CountingBuilder:
            def __init__(self) -> None:
                self.calls = 0

            def build_runtime_memory_context(self, task: str):
                self.calls += 1
                return "", (), {}

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            builder = CountingBuilder()
            kernel.context_builder = builder
            result = kernel.execute_turn("why does this fail?")

        self.assertEqual(
            result.final_response,
            "What is failing? Share the command, error message, file, or step that is breaking so I can trace it accurately.",
        )
        self.assertEqual(builder.calls, 0)
        self.assertEqual(result.steps, [])

    def test_answer_from_retrieved_memory_does_not_answer_generic_why_does_prompt_from_unrelated_memory(self) -> None:
        answer = _answer_from_retrieved_memory(
            "Why does this fail?",
            "\n".join(
                [
                    "## Retrieved Memory",
                    "- [episode] can you explain how the main.py works | The `main.py` file is a Python script that sets up a simple HTTP server.",
                    "- [episode] what architecture was getgit | GetGit looks like a Flask/Python RAG app.",
                ]
            ),
        )

        self.assertIsNone(answer)

    def test_execute_turn_returns_partial_success_when_follow_up_ai_call_fails(self) -> None:
        memory = FakeMemory()
        ai = RateLimitedAI(
            [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call_1",
                            tool_name="read_file",
                            arguments={"path": "note.txt"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 7},
                ),
                RuntimeError("Groq chat completion failed with HTTP 429"),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            note_path = f"{tempdir}/note.txt"
            with open(note_path, "w", encoding="utf-8") as handle:
                handle.write("hello runtime")
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())
            result = kernel.execute_turn("Read note.txt")

        self.assertEqual(len(result.steps), 1)
        self.assertTrue(result.steps[0].success)
        self.assertIn("applied", result.final_response or "")
        self.assertIn("HTTP 429", result.final_response or "")

    def test_local_only_direct_mode_answers_from_memory_without_ai(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- [episode] The calendar project backend used FastAPI."
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("What was the calendar project backend?", local_only=True)

        self.assertIn("FastAPI", result.final_response or "")

    def test_local_only_turn_does_not_require_remote_ai_initialization(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- [episode] GetGit used a Flask backend with server.py and a rag folder."
        )
        previous = os.environ.get("GROQ_API_KEY")
        os.environ.pop("GROQ_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                kernel = DevenvKernel(tempdir, memory=memory, ai=None)
                result = kernel.execute_turn("What architecture did GetGit use?", local_only=True)
        finally:
            if previous is not None:
                os.environ["GROQ_API_KEY"] = previous

        self.assertIn("Flask", result.final_response or "")

    def test_local_only_direct_turn_skips_lazy_tooling_helpers(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- [episode] GetGit used a Flask backend with server.py and a rag folder."
        )

        with tempfile.TemporaryDirectory() as tempdir:
            with mock.patch("core.runtime.kernel.ContextBuilderService", side_effect=AssertionError("context builder should stay lazy")):
                with mock.patch.object(DevenvKernel, "_build_tool_client", side_effect=AssertionError("tool client should stay lazy")):
                    kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
                    result = kernel.execute_turn("What architecture did GetGit use?", local_only=True)

        self.assertIn("Flask", result.final_response or "")

    def test_local_only_direct_turn_can_skip_retrieval_with_exact_logged_answer(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.calls = 0

            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                self.calls += 1
                return [
                    EpisodicLog(
                        log_id="good-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components.",
                                "metadata": {"external_context_query": "What architecture did GetGit use?"},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("What architecture did GetGit use?", local_only=True)

        self.assertEqual(
            result.final_response,
            "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components.",
        )
        self.assertEqual(memory.logs, [])
        self.assertEqual(memory.working_memory_calls, [])

    def test_local_only_exact_logged_answer_cache_skips_repeat_store_lookup(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.calls = 0

            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                self.calls += 1
                return [
                    EpisodicLog(
                        log_id="good-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components.",
                                "metadata": {"external_context_query": "What architecture did GetGit use?"},
                            }
                        ),
                    )
                ]

        memory = FakeMemory()
        store = FakeStore()
        memory.store = store
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            first = kernel.execute_turn("What architecture did GetGit use?", local_only=True)
            second = kernel.execute_turn("What architecture did GetGit use?", local_only=True)

        self.assertIn("Flask", first.final_response or "")
        self.assertEqual(second.final_response, first.final_response)
        self.assertEqual(store.calls, 1)

    def test_local_only_exact_logged_answer_prefers_exact_query_store_lookup(self) -> None:
        class FakeStore:
            def search_logs_for_external_query(self, query: str, limit: int = 5) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="good-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app.",
                                "metadata": {"external_context_query": "Does get-drip use the same architecture as GetGit?"},
                            }
                        ),
                    )
                ]

            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                raise AssertionError("broad search_logs should not be needed when exact query lookup succeeds")

        memory = FakeMemory()
        memory.store = FakeStore()
        memory.retrieve_context = lambda current_prompt, top_k=5: (_ for _ in ()).throw(AssertionError("retrieve_context should be skipped"))

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ExplodingAI([]))
            result = kernel.execute_turn("Does get-drip use the same architecture as GetGit?", local_only=True)

        self.assertEqual(
            result.final_response,
            "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app.",
        )

    def test_sqlite_store_search_logs_for_external_query_finds_newly_inserted_log(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(f"{tempdir}/memory.db")
            store.insert_log(
                EpisodicLog(
                    log_id="log-1",
                    timestamp=1.0,
                    associated_node_id=None,
                    raw_interaction=json.dumps(
                        {
                            "user": "Does get-drip use the same architecture as GetGit?",
                            "agent": "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app.",
                            "metadata": {"external_context_query": "Does get-drip use the same architecture as GetGit?"},
                        }
                    ),
                )
            )

            rows = store.search_logs_for_external_query("Does get-drip use the same architecture as GetGit?")

        self.assertEqual(len(rows), 1)
        self.assertIn("Convex-backed app", rows[0].raw_interaction)

    def test_sqlite_store_search_logs_for_external_query_finds_legacy_log_without_indexed_value(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(f"{tempdir}/memory.db")
            raw_interaction = json.dumps(
                {
                    "user": "What architecture did GetGit use?",
                    "agent": "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components.",
                    "metadata": {"external_context_query": "What architecture did GetGit use?"},
                }
            )
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction, external_context_query)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("legacy-1", 1.0, None, raw_interaction, None),
                )

            rows = store.search_logs_for_external_query("What architecture did GetGit use?")

        self.assertEqual(len(rows), 1)
        self.assertIn("server.py", rows[0].raw_interaction)

    def test_sqlite_store_search_logs_for_external_query_backfills_legacy_row(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(f"{tempdir}/memory.db")
            query = "What architecture did GetGit use?"
            raw_interaction = json.dumps(
                {
                    "user": query,
                    "agent": "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components.",
                    "metadata": {"external_context_query": query},
                }
            )
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction, external_context_query)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("legacy-2", 1.0, None, raw_interaction, None),
                )

            first_rows = store.search_logs_for_external_query(query)
            with store.transaction() as connection:
                backfilled = connection.execute(
                    "SELECT external_context_query FROM episodic_logs WHERE log_id = ?",
                    ("legacy-2",),
                ).fetchone()
            second_rows = store.search_logs_for_external_query(query)

        self.assertEqual(len(first_rows), 1)
        self.assertEqual(str(backfilled["external_context_query"]), query)
        self.assertEqual(len(second_rows), 1)

    def test_sqlite_store_search_agent_responses_for_external_query_backfills_legacy_row(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(f"{tempdir}/memory.db")
            query = "What architecture did GetGit use?"
            response = "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components."
            raw_interaction = json.dumps(
                {
                    "user": query,
                    "agent": response,
                    "metadata": {"external_context_query": query},
                }
            )
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction, external_context_query, agent_response)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("legacy-3", 1.0, None, raw_interaction, None, None),
                )

            first_responses = store.search_agent_responses_for_external_query(query)
            with store.transaction() as connection:
                backfilled = connection.execute(
                    "SELECT external_context_query, agent_response FROM episodic_logs WHERE log_id = ?",
                    ("legacy-3",),
                ).fetchone()
            second_responses = store.search_agent_responses_for_external_query(query)

        self.assertEqual(first_responses, [response])
        self.assertEqual(str(backfilled["external_context_query"]), query)
        self.assertEqual(str(backfilled["agent_response"]), response)
        self.assertEqual(second_responses, [response])

    def test_sqlite_store_search_agent_responses_for_external_query_uses_stored_response_without_json_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = SQLiteMemoryStore(f"{tempdir}/memory.db")
            query = "What architecture did GetGit use?"
            response = "GetGit was described as a Flask backend, with a `server.py` entrypoint, and RAG-related components."
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO episodic_logs (log_id, timestamp, associated_node_id, raw_interaction, external_context_query, agent_response)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("indexed-1", 1.0, None, "{not-json", query, response),
                )

            responses = store.search_agent_responses_for_external_query(query)

        self.assertEqual(responses, [response])

    def test_direct_memory_answer_handles_architecture_question_without_ai(self) -> None:
        memory = FakeMemory()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(
            markdown_context="## Retrieved Memory\n- [episode] GetGit used a Flask backend with server.py and a rag folder."
        )
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("What architecture did GetGit use?")

        self.assertIn("Flask", result.final_response or "")
        self.assertIn("server.py", result.final_response or "")

    def test_local_only_prefers_clean_exact_logged_project_answer(self) -> None:
        class FakeStore:
            def search_logs(self, terms: list[str], limit: int = 5) -> list[EpisodicLog]:
                return [
                    EpisodicLog(
                        log_id="bad-1",
                        timestamp=2.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "`README.md` references Flask, RAG. Preview: # GetGit ...",
                                "metadata": {"external_context_query": "Does get-drip use the same architecture as GetGit?"},
                            }
                        ),
                    ),
                    EpisodicLog(
                        log_id="good-1",
                        timestamp=1.0,
                        associated_node_id=None,
                        raw_interaction=json.dumps(
                            {
                                "agent": "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app.",
                                "metadata": {"external_context_query": "Does get-drip use the same architecture as GetGit?"},
                            }
                        ),
                    ),
                ]

        memory = FakeMemory()
        memory.store = FakeStore()
        memory.retrieve_context = lambda current_prompt, top_k=5: FakeRetrievalResult(markdown_context="")
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            result = kernel.execute_turn("Does get-drip use the same architecture as GetGit?", local_only=True)

        self.assertEqual(
            result.final_response,
            "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app.",
        )

    def test_local_only_planning_executes_without_ai(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(WriteFileTool())
            first = kernel.execute_turn("create a frontend folder in calendar with html css and js", planning_mode=PlanningMode.FORCE_PLAN, local_only=True)
            second = kernel.execute_turn(
                "create a frontend folder in calendar with html css and js",
                planning_mode=PlanningMode.FORCE_PLAN,
                continue_plan=True,
                local_only=True,
            )

            html_path = Path(tempdir) / "calendar" / "frontend" / "index.html"
            css_path = Path(tempdir) / "calendar" / "frontend" / "styles.css"
            html_exists = html_path.exists()
            css_exists = css_path.exists()
            html_content = html_path.read_text(encoding="utf-8")

        self.assertIsNotNone(first.blueprint)
        self.assertEqual(len(first.blueprint.tasks), 3)
        self.assertTrue(html_exists)
        self.assertTrue(css_exists)
        self.assertIn("calendar-weekdays", html_content)
        self.assertIn("calendar styling", second.final_response or "")

    def test_local_only_dark_theme_calendar_writes_dark_css(self) -> None:
        memory = FakeMemory()
        ai = ExplodingAI([])

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(WriteFileTool())
            first = kernel.execute_turn(
                "create a dark theme frontend folder in calendar with html css and js",
                planning_mode=PlanningMode.FORCE_PLAN,
                local_only=True,
            )
            second = kernel.execute_turn(
                "create a dark theme frontend folder in calendar with html css and js",
                planning_mode=PlanningMode.FORCE_PLAN,
                continue_plan=True,
                local_only=True,
            )

            css_path = Path(tempdir) / "calendar" / "frontend" / "styles.css"
            css_content = css_path.read_text(encoding="utf-8")

        self.assertIsNotNone(first.blueprint)
        self.assertIn("color-scheme: dark", css_content)
        self.assertIn("--bg: #11161d", css_content)
        self.assertIn("calendar styling", second.final_response or "")

    def test_memory_persists_across_kernel_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first_memory = MemoryEngine(
                db_path=f"{tempdir}/memory.db",
                vector_dir=f"{tempdir}/vectors",
                embedder=HashingEmbedder(dimension=8),
                vector_index=InMemoryVectorIndex(),
            )
            first_ai = FakeAI(
                [
                    AIResponse(
                        content="The calendar project used a Python backend and React frontend.",
                        tool_calls=(),
                        finish_reason="stop",
                        usage={"prompt_tokens": 3},
                    )
                ]
            )
            first_kernel = DevenvKernel(tempdir, memory=first_memory, ai=first_ai)
            first_kernel.local_router = _disabled_router()
            first_kernel.execute_turn("Remember the calendar project stack")

            second_memory = MemoryEngine(
                db_path=f"{tempdir}/memory.db",
                vector_dir=f"{tempdir}/vectors",
                embedder=HashingEmbedder(dimension=8),
                vector_index=InMemoryVectorIndex(),
            )
            second_ai = FakeAI(
                [
                    AIResponse(
                        content="I found prior context.",
                        tool_calls=(),
                        finish_reason="stop",
                        usage={"prompt_tokens": 2},
                    )
                ]
            )
            second_kernel = DevenvKernel(tempdir, memory=second_memory, ai=second_ai)
            second_kernel.local_router = _disabled_router()
            result = second_kernel.execute_turn("What was the calendar project backend?")

        self.assertEqual(result.final_response, "I found prior context.")
        self.assertIn("calendar project", second_ai.chat_calls[0]["memory_context"].lower())
        self.assertIn("python backend", second_ai.chat_calls[0]["memory_context"].lower())

    def test_session_budget_blocks_future_turns_after_limit_is_reached(self) -> None:
        memory = FakeMemory()
        ai = FakeAI(
            [
                AIResponse(content="First answer", tool_calls=(), finish_reason="stop", usage={"total_tokens": 8}),
                AIResponse(content="Second answer", tool_calls=(), finish_reason="stop", usage={"total_tokens": 4}),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.local_router = _disabled_router()
            first = kernel.execute_turn("Explain the repo", session_budget_tokens=10)
            second = kernel.execute_turn("Explain the repo again", session_budget_tokens=6)

        self.assertEqual(first.final_response, "First answer")
        self.assertEqual(first.metadata["budget_state"]["used"], 8)
        self.assertTrue(second.metadata["budget_state"]["blocked"])
        self.assertIn("budget", second.error_message.lower())
        self.assertEqual(second.total_usage["total_tokens"], 8)


def _disabled_router():
    return type(
        "Router",
        (),
        {
            "decide": lambda self, prompt: LocalRouteDecision(
                use_local_knowledge=False,
                confidence=0.0,
                knowledge_score=0.0,
                remote_score=1.0,
                reason="test",
            )
        },
    )()


if __name__ == "__main__":
    unittest.main()
