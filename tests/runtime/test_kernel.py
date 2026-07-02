from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse, ToolCallRequest
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex
from core.runtime import DevenvKernel
from core.runtime.context_builder import ContextBuilderService
from core.runtime.kernel import _answer_from_retrieved_memory, _summarize_directory_listing
from core.runtime.local_router import LocalRouteDecision
from core.runtime.models import ExternalSessionProviderConfig, PlanningMode
from core.tools.list_directory import ListDirectoryTool
from core.tools.read_file import ReadFileTool
from core.tools.write_file import WriteFileTool


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


class DevenvKernelTest(unittest.TestCase):
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

    def test_default_memory_paths_are_scoped_under_project_root(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)

        repo_root = Path(__file__).resolve().parents[2]
        self.assertEqual(kernel.db_path, str((repo_root / "memory.db").resolve()))
        self.assertEqual(kernel.vector_dir, str((repo_root / "vectors").resolve()))

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
        self.assertEqual(ai.chat_calls, [])

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

    def test_local_knowledge_route_defers_code_level_question_without_memory_answer(self) -> None:
        memory = FakeMemory()
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
            (getgit_path / "core.py").write_text("print('core')", encoding="utf-8")
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
            result = kernel.execute_turn("how does getgit decide what content to send to ai?")

        self.assertEqual(result.final_response, "GetGit decides what content to send by chunking and retrieval.")
        self.assertEqual(len(ai.chat_calls), 1)
        self.assertEqual(result.steps, [])

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
