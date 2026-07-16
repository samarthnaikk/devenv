from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from core.ai.models import AIExecutedToolStep, AIResponse, ToolCallRequest
from core.runtime.models import CheckpointTask, ExecutionBlueprint
from core.runtime.models import ExternalSessionProviderConfig, PlanningMode, RunConfig
from core.runtime.setup import inspect_setup
from core.runtime.web import DevenvWebApp


@dataclass(frozen=True)
class FakeRetrievalResult:
    markdown_context: str


class FakeMemory:
    def record_working_memory(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        return None

    def retrieve_context(self, current_prompt: str, top_k: int = 5) -> FakeRetrievalResult:
        return FakeRetrievalResult(markdown_context="")

    def add_episodic_log(
        self,
        user_prompt: str,
        agent_response: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "log-1"


class FakeAI:
    def __init__(self) -> None:
        self.model = "fake-opencode-model"
        self.provider_label = "OpenCode CLI"
        self.preferred_backend = "opencode"
        self.last_backend_used = "opencode"
        self.reset_session_calls = 0
        self.responses = [
            AIResponse(content="Website response", tool_calls=(), finish_reason="stop", usage={"prompt_tokens": 3}),
        ]
        self.registered_tools: list[str] = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def status(self) -> dict[str, object]:
        from core.ai.models import AIBackendStatus

        return {
            "opencode": AIBackendStatus(
                name="opencode",
                available=True,
                enabled=True,
                model=self.model,
                detail="Server reachable",
                metadata={
                    "server": {
                        "reachable": True,
                        "healthy": True,
                        "version": "1.3.3",
                        "detail": "OpenCode server reachable: 1.3.3",
                        "base_url": "http://127.0.0.1:4096",
                        "started_by_manager": False,
                    }
                },
            ),
            "ollama": AIBackendStatus(
                name="ollama",
                available=True,
                enabled=True,
                model="qwen2.5:3b",
                detail="Ollama reachable",
                metadata={
                    "models": ["qwen2.5:3b", "codellama:7b"],
                    "base_url": "http://127.0.0.1:11434",
                },
            ),
            "codex": AIBackendStatus(
                name="codex",
                available=True,
                enabled=True,
                model="gpt-5-codex",
                detail="Configured",
                metadata={
                    "transport": "responses_mcp",
                    "mcp_server": {
                        "reachable": True,
                        "base_url": "http://127.0.0.1:8765/mcp",
                    },
                },
            ),
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names=None,
    ) -> AIResponse:
        return self.responses.pop(0)

    def reset_session(self) -> None:
        self.reset_session_calls += 1

    def set_backend_model(self, backend: str, model: str) -> None:
        self.model = model


class CapturingFakeAI(FakeAI):
    def __init__(self) -> None:
        super().__init__()
        self.chat_calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names=None,
    ) -> AIResponse:
        self.chat_calls.append(
            {
                "messages": messages,
                "memory_context": memory_context,
                "temperature": temperature,
                "tool_names": list(tool_names or []),
            }
        )
        return super().chat(
            messages=messages,
            memory_context=memory_context,
            temperature=temperature,
            tool_names=tool_names,
        )


class DevenvWebAppTest(unittest.TestCase):
    def test_health_payload_reuses_cached_setup_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            readiness = inspect_setup(RunConfig(workspace_path=tempdir), include_optional=True)
            with mock.patch("core.runtime.web.inspect_setup", return_value=readiness) as inspect_mock:
                first = app.build_health_payload()
                second = app.build_health_payload()

        self.assertTrue(first["setup"]["checked_at"])
        self.assertEqual(first["setup"], second["setup"])
        self.assertEqual(inspect_mock.call_count, 1)

    def test_payload_helpers_expose_workspace_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("hello", encoding="utf-8")
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            health = app.build_health_payload()
            files = app.build_files_payload()
            file_payload = app.build_file_payload("README.md")

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["ai_provider"], "OpenCode CLI")
        self.assertEqual(health["ai_model"], "fake-opencode-model")
        self.assertIn("fake-opencode-model", health["available_models"])
        self.assertTrue(health["context_builder_enabled"])
        self.assertIn("context_sources", health)
        self.assertTrue(health["opencode_server"]["reachable"])
        self.assertEqual(health["performance_mode"], "low")
        self.assertFalse(health["privacy"]["no_memory"])
        self.assertFalse(health["privacy"]["incognito"])
        self.assertIn("setup", health)
        self.assertIn("tool_readiness", health)
        self.assertIn("web_search", health["tool_readiness"])
        self.assertIn("mcp_server", health)
        self.assertIn("codex_backend", health)
        self.assertEqual(health["codex_backend"]["transport"], "responses_mcp")
        self.assertEqual(files["entries"][0]["name"], "README.md")
        self.assertEqual(file_payload["content"], "hello")
        self.assertEqual(file_payload["kind"], "text")

    def test_health_payload_exposes_runtime_contract_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(
                    workspace_path=tempdir,
                    performance_mode="high",
                    no_memory=True,
                    incognito=True,
                ),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            health = app.build_health_payload()

        self.assertEqual(health["performance_mode"], "high")
        self.assertTrue(health["privacy"]["no_memory"])
        self.assertTrue(health["privacy"]["incognito"])
        self.assertFalse(health["setup"]["ready"] is None)
        self.assertEqual(health["tool_readiness"]["generate_prompt"]["ready"], True)

    def test_setup_inspection_exposes_shared_readiness_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            readiness = inspect_setup(
                RunConfig(
                    workspace_path=tempdir,
                    performance_mode="medium",
                ),
                include_optional=True,
            )

        self.assertEqual(readiness.required_checks[0].name, "workspace")
        self.assertEqual(readiness.required_checks[0].status, "ready")
        self.assertEqual(readiness.optional_checks[0].name, "opencode_server")
        self.assertIsNotNone(readiness.checked_at)

    def test_file_payload_supports_image_preview(self) -> None:
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "pixel.png").write_bytes(png_bytes)
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            payload = app.build_file_payload("pixel.png")

        self.assertEqual(payload["kind"], "image")
        self.assertEqual(payload["content_type"], "image/png")
        self.assertTrue(str(payload["content"]).startswith("data:image/png;base64,"))

    def test_file_payload_supports_pdf_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "report.pdf").write_bytes(b"%PDF-1.4\n%preview\n")
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            payload = app.build_file_payload("report.pdf")

        self.assertEqual(payload["kind"], "pdf")
        self.assertEqual(payload["content_type"], "application/pdf")
        self.assertTrue(str(payload["content"]).startswith("data:application/pdf;base64,"))

    def test_health_payload_marks_generate_pdf_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            health = app.build_health_payload()

        self.assertTrue(health["tool_readiness"]["generate_pdf"]["ready"])

    def test_web_app_exposes_turn_and_error_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("hello", encoding="utf-8")
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            health = app.build_health_payload()
            files = app.build_files_payload()
            turn = app.run_turn("show me the readme")

            self.assertEqual(health["status"], "ok")
            self.assertEqual(files["entries"][0]["name"], "README.md")
            self.assertEqual(turn["final_response"], "Website response")
            self.assertEqual(turn["blueprint"]["tasks"][0]["description"], "show me the readme")
        with self.assertRaises(PermissionError):
            app.build_file_payload("../secrets.txt")

    def test_run_turn_accepts_explicit_planning_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            captured: dict[str, object] = {}
            app.kernel.execute_turn = lambda prompt, max_consecutive_tools=5, planning_mode=PlanningMode.AUTO, continue_plan=False, local_only=False: captured.update(
                {
                    "prompt": prompt,
                    "planning_mode": planning_mode,
                    "continue_plan": continue_plan,
                    "local_only": local_only,
                }
            ) or type("Result", (), {"to_dict": lambda self: {"final_response": "ok"}})()
            result = app.run_turn("hello", planning_mode=PlanningMode.FORCE_PLAN, continue_plan=True, local_only=True)

        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(captured["prompt"], "hello")
        self.assertEqual(captured["planning_mode"], PlanningMode.FORCE_PLAN)
        self.assertTrue(captured["continue_plan"])
        self.assertTrue(captured["local_only"])

    def test_run_turn_forwards_selected_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            captured: dict[str, object] = {}

            def fake_execute_turn(
                prompt,
                max_consecutive_tools=5,
                planning_mode=PlanningMode.AUTO,
                continue_plan=False,
                local_only=False,
                selected_tools=None,
            ):
                captured.update(
                    {
                        "prompt": prompt,
                        "selected_tools": selected_tools,
                    }
                )
                return type("Result", (), {"to_dict": lambda self: {"final_response": "ok"}})()

            app.kernel.execute_turn = fake_execute_turn
            result = app.run_turn("search this", selected_tools=["web_search", "read_file"])

        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(captured["prompt"], "search this")
        self.assertEqual(captured["selected_tools"], ["web_search", "read_file"])

    def test_reset_plan_state_clears_active_kernel_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.active_plan_prompt = "Build a planner"
            app.kernel.active_blueprint = object()

            result = app.reset_plan_state()

        self.assertTrue(result["cleared"])
        self.assertEqual(result["state"], "PLANNING")
        self.assertIsNone(app.kernel.active_plan_prompt)
        self.assertIsNone(app.kernel.active_blueprint)

    def test_set_model_updates_health_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            app.update_backend_access("opencode", True)
            app.kernel.ai.last_backend_used = "opencode"
            payload = app.set_model("llama-3.1-8b-instant")
            health = app.build_health_payload()

        self.assertEqual(payload["ai_model"], "llama-3.1-8b-instant")
        self.assertEqual(health["ai_model"], "llama-3.1-8b-instant")
        self.assertEqual(health["ai_provider"], "OpenCode CLI")
        self.assertIn("llama-3.1-8b-instant", health["available_models"])

    def test_context_builder_payload_helpers_expose_sessions_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            (workspace / "README.md").write_text("Prompt builder workspace.", encoding="utf-8")
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "06" / "28"
            sessions_dir.mkdir(parents=True)
            session_id = "web-session-1"
            (codex_root / "session_index.jsonl").write_text(
                '{"id":"web-session-1","thread_name":"Prompt builder","updated_at":"2026-06-28T12:00:00Z"}\n',
                encoding="utf-8",
            )
            (codex_root / "history.jsonl").write_text(
                '{"session_id":"web-session-1","ts":1,"text":"Prepare a prompt for Codex."}\n',
                encoding="utf-8",
            )
            (sessions_dir / "rollout-2026-06-28T12-00-00-web-session-1.jsonl").write_text(
                '{"timestamp":"2026-06-28T12:00:01Z","type":"event_msg","payload":{"type":"agent_message","message":"Keep this manual and copy-paste only."}}\n',
                encoding="utf-8",
            )
            app = DevenvWebApp(
                RunConfig(
                    workspace_path=tempdir,
                    external_session_configs=(
                        ExternalSessionProviderConfig(
                            provider="codex",
                            root_path=str(codex_root),
                            index_path="session_index.jsonl",
                        ),
                    ),
                ),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            sources = app.build_context_sources_payload()
            app.update_session_access("codex", True)
            sessions = app.build_context_sessions_payload("codex")
            detail = app.build_context_session_payload("codex", session_id)
            prepared = app.build_prepared_prompt_payload(
                {
                    "task": "Prepare a prompt with minimal changes.",
                    "provider": "codex",
                    "session_ids": [session_id],
                    "include_workspace_scan": True,
                    "include_prior_context": True,
                    "output_format": "compact",
                }
            )

        self.assertEqual(sources["sources"][0]["provider"], "codex")
        self.assertEqual(sessions["sessions"][0]["session_id"], session_id)
        self.assertEqual(detail["summary"]["session_id"], session_id)
        self.assertIn("Task:", prepared["prompt"])

    def test_access_endpoints_update_server_side_consent_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            session_payload = app.update_session_access("codex", True)
            backend_payload = app.update_backend_access("opencode", True)
            ollama_backend_payload = app.update_backend_access("ollama", True)
            codex_backend_payload = app.update_backend_access("codex", True)
            health = app.build_health_payload()

        self.assertTrue(session_payload["session_access"]["codex"])
        self.assertTrue(backend_payload["backend_access"]["opencode"])
        self.assertTrue(ollama_backend_payload["backend_access"]["ollama"])
        self.assertTrue(codex_backend_payload["backend_access"]["codex"])
        self.assertTrue(health["access_policy"]["session_access"]["codex"])
        self.assertTrue(health["access_policy"]["backend_access"]["opencode"])
        self.assertTrue(health["access_policy"]["backend_access"]["ollama"])
        self.assertTrue(health["access_policy"]["backend_access"]["codex"])

    def test_health_payload_exposes_model_catalog_by_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            health = app.build_health_payload()

        self.assertIn("available_models_by_backend", health)
        self.assertIn("ollama", health["available_models_by_backend"])
        self.assertIn("qwen2.5:3b", health["available_models_by_backend"]["ollama"])
        self.assertEqual(health["selected_models_by_backend"]["ollama"], "qwen2.5:3b")

    def test_set_model_can_target_specific_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )

            payload = app.set_model("qwen2.5:3b", "ollama")

        self.assertEqual(payload["ai_model"], "qwen2.5:3b")
        self.assertEqual(payload["selected_models_by_backend"]["ollama"], "qwen2.5:3b")

    def test_run_turn_forwards_codex_backend_access_and_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.update_backend_access("codex", True)
            captured: dict[str, object] = {}

            def fake_execute_turn(
                prompt,
                max_consecutive_tools=5,
                planning_mode=PlanningMode.AUTO,
                continue_plan=False,
                local_only=False,
                selected_tools=None,
                backend_preference="opencode",
                opencode_enabled=False,
                codex_enabled=False,
            ):
                captured.update(
                    {
                        "prompt": prompt,
                        "backend_preference": backend_preference,
                        "opencode_enabled": opencode_enabled,
                        "codex_enabled": codex_enabled,
                    }
                )
                return type("Result", (), {"to_dict": lambda self: {"final_response": "ok"}})()

            app.kernel.execute_turn = fake_execute_turn
            result = app.run_turn("hello", backend_preference="codex")

        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(captured["backend_preference"], "codex")
        self.assertFalse(captured["opencode_enabled"])
        self.assertTrue(captured["codex_enabled"])

    def test_run_turn_forwards_ollama_backend_access_and_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.update_backend_access("ollama", True)
            captured: dict[str, object] = {}

            def fake_execute_turn(
                prompt,
                max_consecutive_tools=5,
                planning_mode=PlanningMode.AUTO,
                continue_plan=False,
                local_only=False,
                selected_tools=None,
                backend_preference="opencode",
                opencode_enabled=False,
                ollama_enabled=False,
                codex_enabled=False,
            ):
                captured.update(
                    {
                        "prompt": prompt,
                        "backend_preference": backend_preference,
                        "opencode_enabled": opencode_enabled,
                        "ollama_enabled": ollama_enabled,
                        "codex_enabled": codex_enabled,
                    }
                )
                return type("Result", (), {"to_dict": lambda self: {"final_response": "ok"}})()

            app.kernel.execute_turn = fake_execute_turn
            result = app.run_turn("hello", backend_preference="ollama")

        self.assertEqual(result["final_response"], "ok")
        self.assertEqual(captured["backend_preference"], "ollama")
        self.assertFalse(captured["opencode_enabled"])
        self.assertTrue(captured["ollama_enabled"])
        self.assertFalse(captured["codex_enabled"])

    def test_reset_thread_clears_kernel_conversation_and_ai_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.kernel.ephemeral_history = [{"role": "user", "content": "hello"}]
            app.kernel.session_usage_totals = {"total_tokens": 12}
            previous_session_id = app.kernel.session_id

            payload = app.reset_thread()

        self.assertEqual(payload["state"], "PLANNING")
        self.assertEqual(payload["usage"], {})
        self.assertEqual(ai.reset_session_calls, 1)
        self.assertEqual(app.kernel.ephemeral_history, [])
        self.assertNotEqual(previous_session_id, app.kernel.session_id)

    def test_performance_mode_updates_health_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir, performance_mode="medium"),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            payload = app.update_performance_mode("high")
            health = app.build_health_payload()

        self.assertEqual(payload["performance_mode"], "high")
        self.assertEqual(health["performance_mode"], "high")
        self.assertEqual(app.context_builder.performance_mode, "high")

    def test_privacy_mode_updates_health_payload_and_turn_args(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            captured: dict[str, object] = {}
            def fake_execute_turn(
                prompt,
                *,
                max_consecutive_tools=5,
                planning_mode=PlanningMode.AUTO,
                continue_plan=False,
                local_only=False,
                backend_preference="auto",
                opencode_enabled=False,
                session_budget_tokens=None,
                no_memory=False,
                incognito=False,
            ):
                captured.update(
                    {
                        "no_memory": no_memory,
                        "incognito": incognito,
                    }
                )
                return type("Result", (), {"to_dict": lambda self: {"final_response": "ok", "total_usage": {}, "metadata": {}}})()

            app.kernel.execute_turn = fake_execute_turn

            privacy = app.update_privacy_mode(no_memory=False, incognito=True)
            health = app.build_health_payload()
            app.run_turn("hello")

        self.assertTrue(privacy["privacy"]["incognito"])
        self.assertTrue(privacy["privacy"]["no_memory"])
        self.assertTrue(health["privacy"]["incognito"])
        self.assertTrue(health["privacy"]["no_memory"])
        self.assertTrue(captured["incognito"])
        self.assertTrue(captured["no_memory"])

    def test_session_payload_requires_explicit_provider_access(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            with self.assertRaises(PermissionError):
                app.build_context_sessions_payload("codex")

    def test_run_turn_exposes_top_level_backend_and_usage_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": "ok",
                        "total_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                        "elapsed_ms": 12,
                    }
                },
            )()

            payload = app.run_turn("hello")

        self.assertEqual(payload["backend_used"], "opencode")
        self.assertEqual(payload["usage_sample"]["total_tokens"], 5)
        self.assertFalse(payload["budget_state"]["blocked"])

    def test_run_tool_executes_registered_runtime_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            payload = app.run_tool("generate_prompt", {"task": "Prepare a migration plan"})

        self.assertTrue(payload["success"])
        self.assertEqual(payload["tool_name"], "generate_prompt")
        self.assertIn("Prepare a migration plan", payload["data"]["prompt"])

    def test_run_turn_preserves_partial_blueprint_when_execution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, max_consecutive_tools=5, planning_mode=PlanningMode.AUTO, continue_plan=False, local_only=False: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": None,
                        "steps": [],
                        "total_usage": {},
                        "ai_logs": [],
                        "system_logs": ["Execution failed: Execution tool limit reached before the checkpoint completed."],
                        "state": "EXECUTING",
                        "blueprint": ExecutionBlueprint(
                            raw_plan_markdown="- [ ] Build frontend\n- [ ] Add HTML",
                            tasks=[
                                CheckpointTask(task_id=1, description="Build frontend", is_completed=True),
                                CheckpointTask(task_id=2, description="Add HTML"),
                            ],
                            active_task_pointer=1,
                        ).to_dict(),
                        "error_message": "Execution tool limit reached before the checkpoint completed.",
                    }
                },
            )()
            result = app.run_turn("hello", planning_mode=PlanningMode.FORCE_PLAN)

        self.assertEqual(result["error_message"], "Execution tool limit reached before the checkpoint completed.")
        self.assertEqual(result["blueprint"]["tasks"][0]["description"], "Build frontend")

    def test_run_plan_returns_valid_blueprint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-web", "description": "Inspect core/runtime/web.py for planning hooks and generate_pdf readiness exposure", "level": 0},
                                {"task_id": "review-tooling", "description": "Review core/runtime/tooling.py to confirm where generate_pdf should be registered", "level": 0},
                                {"task_id": "design-tool", "description": "Define the generate_pdf tool contract and LaTeX-backed output shape in core/tools", "level": 1},
                                {"task_id": "wire-runtime", "description": "Wire the generate_pdf tool through the runtime entrypoints and setup/readiness surfaces", "level": 2},
                                {"task_id": "verify-runtime", "description": "Verify the integration with tests/runtime/test_web.py and related tool coverage", "level": 3},
                            ],
                            "edges": [
                                {"from": "inspect-web", "to": "design-tool"},
                                {"from": "review-tooling", "to": "design-tool"},
                                {"from": "design-tool", "to": "wire-runtime"},
                                {"from": "wire-runtime", "to": "verify-runtime"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                    backend="ollama",
                )
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan(
                "plan on how to add pdf generation tool to this codebase",
                backend_preference="ollama",
            )

        self.assertEqual(result["backend_used"], "ollama")
        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][0]["task_id"], "inspect-web")
        self.assertEqual(result["usage_sample"]["total_tokens"], 10)

    def test_run_plan_blocks_mutation_tools_and_recovers_with_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            ai.responses = [
                AIResponse(
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            call_id="call-1",
                            tool_name="write_file",
                            arguments={"path": "sample-test/styles.css", "content": "bad", "mode": "overwrite"},
                        ),
                    ),
                    finish_reason="tool_calls",
                    usage={"total_tokens": 3},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-ui", "description": "Inspect the plan rendering path", "level": 0},
                                {"task_id": "patch-schema", "description": "Patch the planner JSON schema handling", "level": 1},
                            ],
                            "edges": [{"from": "inspect-ui", "to": "patch-schema"}],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 5},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan how to fix planner JSON")

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][1]["task_id"], "patch-schema")
        self.assertEqual(result["steps"], [])
        self.assertTrue(
            any("Blocked planning tool call: write_file" in log for log in result["system_logs"])
        )

    def test_run_plan_repairs_invalid_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            ai.responses = [
                AIResponse(
                    content="Here is a plan:\n1. inspect\n2. patch",
                    finish_reason="stop",
                    usage={"total_tokens": 2},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": 1, "description": "Inspect the tool entrypoints", "level": 0},
                                {"task_id": 2, "description": "Implement PDF generation support", "level": 1},
                            ]
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 4},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan pdf generation support")

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][0]["task_id"], "task-1")
        self.assertEqual(result["blueprint"]["tasks"][0]["description"], "Here is a plan:")
        self.assertEqual(result["blueprint"]["edges"][0]["from"], "task-1")

    def test_run_plan_coerces_string_steps_json_into_blueprint(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "steps": [
                                "Inspect the runtime tool registry",
                                "Add a generate_pdf tool contract",
                                "Wire the tool into the web runtime",
                            ]
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 4},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan PDF tool integration")

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][0]["task_id"], "task-1")
        self.assertEqual(
            result["blueprint"]["tasks"][1]["description"],
            "Add a generate_pdf tool contract",
        )

    def test_run_plan_coerces_plain_text_list_into_blueprint(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ai = FakeAI()
            ai.responses = [
                AIResponse(
                    content="1. Inspect the codebase\n2. Add the PDF tool\n3. Test the integration",
                    finish_reason="stop",
                    usage={"total_tokens": 4},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan PDF tool integration")

        self.assertIsNone(result["error_message"])
        self.assertEqual(len(result["blueprint"]["tasks"]), 3)
        self.assertEqual(result["blueprint"]["tasks"][0]["description"], "Inspect the codebase")

    def test_run_plan_attaches_repo_grounding_and_zero_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("PDF generation support lives near runtime tools.", encoding="utf-8")
            Path(tempdir, "core/runtime").mkdir(parents=True)
            Path(tempdir, "core/runtime/web.py").write_text("def run_plan():\n    return 'generate_pdf'\n", encoding="utf-8")
            Path(tempdir, "core/runtime/tooling.py").write_text("def build_runtime_tools():\n    return ['generate_pdf']\n", encoding="utf-8")
            ai = CapturingFakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-runtime", "description": "Inspect runtime tools", "level": 0},
                                {"task_id": "wire-pdf", "description": "Wire PDF support", "level": 1},
                            ],
                            "edges": [{"from": "inspect-runtime", "to": "wire-pdf"}],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 6},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-runtime", "description": "Inspect core/runtime/web.py and planning entrypoints", "level": 0},
                                {"task_id": "review-tooling", "description": "Review core/runtime/tooling.py and core/runtime/setup.py for registration and readiness hooks", "level": 1},
                                {"task_id": "add-pdf-tool", "description": "Add the PDF generation tool implementation and register it", "level": 2},
                                {"task_id": "wire-contract", "description": "Wire the runtime/web contract and health metadata for the new PDF tool", "level": 3},
                                {"task_id": "test-runtime", "description": "Add runtime tests covering planning and execution for the PDF tool", "level": 4},
                            ],
                            "edges": [
                                {"from": "inspect-runtime", "to": "review-tooling"},
                                {"from": "review-tooling", "to": "add-pdf-tool"},
                                {"from": "add-pdf-tool", "to": "wire-contract"},
                                {"from": "wire-contract", "to": "test-runtime"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 7},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan on how to add pdf generation tool to this codebase")

        self.assertIsNone(result["error_message"])
        self.assertEqual(ai.chat_calls[0]["temperature"], 0.0)
        system_messages = [message["content"] for message in ai.chat_calls[0]["messages"] if message.get("role") == "system"]
        self.assertTrue(any("Repository grounding:" in content for content in system_messages))
        combined = "\n".join(system_messages)
        self.assertIn("core/runtime/web.py", combined)
        self.assertIn("core/runtime/tooling.py", combined)

    def test_run_plan_refines_generic_repo_plan_before_returning(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("Runtime planning touches core/runtime/web.py and tests.", encoding="utf-8")
            ai = CapturingFakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "t1", "description": "Inspect the workspace for existing PDF generation tools", "level": 0},
                                {"task_id": "t2", "description": "Create a new file for the PDF generation tool if no existing one is found", "level": 1},
                            ],
                            "edges": [{"from": "t1", "to": "t2"}],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 2},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-runtime", "description": "Inspect core/runtime/web.py and tool registration for existing PDF hooks", "level": 0},
                                {"task_id": "review-tooling", "description": "Review core/runtime/tooling.py and core/tools for the new generate_pdf integration point", "level": 1},
                                {"task_id": "add-tool", "description": "Add the PDF generation tool implementation and register it with the runtime", "level": 2},
                                {"task_id": "wire-ui", "description": "Wire setup/readiness and any UI-facing runtime metadata for the new tool", "level": 3},
                                {"task_id": "test-flow", "description": "Add or update runtime tests covering PDF tool planning and execution flow", "level": 4},
                            ],
                            "edges": [
                                {"from": "inspect-runtime", "to": "review-tooling"},
                                {"from": "review-tooling", "to": "add-tool"},
                                {"from": "add-tool", "to": "wire-ui"},
                                {"from": "wire-ui", "to": "test-flow"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 5},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan on how to add pdf generation tool to this codebase")

        self.assertIsNone(result["error_message"])
        self.assertEqual(len(result["blueprint"]["tasks"]), 5)
        self.assertEqual(len(ai.chat_calls), 2)

    def test_run_plan_refines_even_when_task_count_is_high_but_repo_anchors_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("PDF support is wired through runtime tooling and setup readiness.", encoding="utf-8")
            Path(tempdir, "core/runtime").mkdir(parents=True)
            Path(tempdir, "core/runtime/web.py").write_text("generate_pdf = False\n", encoding="utf-8")
            Path(tempdir, "core/runtime/tooling.py").write_text("def build_runtime_tools():\n    return []\n", encoding="utf-8")
            Path(tempdir, "core/runtime/setup.py").write_text("def inspect_setup():\n    return 'latex_pdf_toolchain'\n", encoding="utf-8")
            ai = CapturingFakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "t1", "description": "Inspect the workspace for PDF support", "level": 0},
                                {"task_id": "t2", "description": "Search for relevant libraries", "level": 1},
                                {"task_id": "t3", "description": "Create a new file for the integration", "level": 2},
                                {"task_id": "t4", "description": "Integrate the chosen library", "level": 3},
                                {"task_id": "t5", "description": "Test the integration", "level": 4},
                            ],
                            "edges": [
                                {"from": "t1", "to": "t2"},
                                {"from": "t2", "to": "t3"},
                                {"from": "t3", "to": "t4"},
                                {"from": "t4", "to": "t5"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 4},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-web", "description": "Inspect core/runtime/web.py for existing generate_pdf planning and readiness hooks", "level": 0},
                                {"task_id": "review-tooling", "description": "Review core/runtime/tooling.py to decide how the generate_pdf tool should be registered", "level": 1},
                                {"task_id": "review-setup", "description": "Check core/runtime/setup.py for latex_pdf_toolchain readiness and optional dependency reporting", "level": 2},
                                {"task_id": "implement-tool", "description": "Add the PDF tool implementation and wire its runtime contract into the identified integration points", "level": 3},
                                {"task_id": "test-runtime", "description": "Add tests covering planner output and runtime readiness for the new PDF path", "level": 4},
                            ],
                            "edges": [
                                {"from": "inspect-web", "to": "review-tooling"},
                                {"from": "review-tooling", "to": "review-setup"},
                                {"from": "review-setup", "to": "implement-tool"},
                                {"from": "implement-tool", "to": "test-runtime"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 6},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan on how to add pdf generation tool to this codebase")

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][0]["task_id"], "inspect-web")
        self.assertEqual(len(ai.chat_calls), 2)

    def test_run_plan_refines_tool_assisted_generic_blueprint_before_returning(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "README.md").write_text("PDF support is wired through runtime tooling and setup readiness.", encoding="utf-8")
            ai = CapturingFakeAI()
            ai.responses = [
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "t1-ep9596de99", "description": "Inspect for existing PDF generation tools in the workspace", "level": 0},
                                {"task_id": "t2-ls-root", "description": "List directories at root level to find potential places for a new pdf_generator.py file", "level": 1},
                                {"task_id": "t3-inspect-codebase", "description": "Inspect the codebase for existing PDF generation tools or relevant files", "level": 2},
                                {"task_id": "t4-find-existing-tools", "description": "Identify if a pdf_generator.py file already exists in the workspace", "level": 3},
                                {"task_id": "t5-create-new-tool", "description": "Create a new pdf_generator.py file for PDF generation tool implementation", "level": 4},
                                {"task_id": "t6-integrate-pdf-generator", "description": "Integrate the newly created pdf_generator.py into existing workflows or add necessary dependencies", "level": 5},
                                {"task_id": "t7-test-pdf-generation", "description": "Test the PDF generation tool to ensure it works as expected", "level": 6},
                            ],
                            "edges": [
                                {"from": "t1-ep9596de99", "to": "t2-ls-root"},
                                {"from": "t2-ls-root", "to": "t3-inspect-codebase"},
                                {"from": "t3-inspect-codebase", "to": "t4-find-existing-tools"},
                                {"from": "t4-find-existing-tools", "to": "t5-create-new-tool"},
                                {"from": "t5-create-new-tool", "to": "t6-integrate-pdf-generator"},
                                {"from": "t6-integrate-pdf-generator", "to": "t7-test-pdf-generation"},
                            ],
                        }
                    ),
                    executed_steps=(
                        AIExecutedToolStep(
                            step_id="step-1",
                            tool_name="list_directory",
                            arguments={"path": ".", "mode": "recursive", "max_depth": 2},
                            output="listed",
                            success=True,
                            data={},
                        ),
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 5},
                    backend="ollama",
                ),
                AIResponse(
                    content=json.dumps(
                        {
                            "tasks": [
                                {"task_id": "inspect-web", "description": "Inspect core/runtime/web.py for planning and generate_pdf readiness hooks", "level": 0},
                                {"task_id": "review-tooling", "description": "Review core/runtime/tooling.py for the runtime registration path", "level": 1},
                                {"task_id": "review-setup", "description": "Check core/runtime/setup.py for latex_pdf_toolchain readiness reporting", "level": 2},
                                {"task_id": "implement-tool", "description": "Add the PDF tool and wire its runtime contract into the discovered integration points", "level": 3},
                                {"task_id": "test-runtime", "description": "Add runtime tests that validate planner output and PDF readiness", "level": 4},
                            ],
                            "edges": [
                                {"from": "inspect-web", "to": "review-tooling"},
                                {"from": "review-tooling", "to": "review-setup"},
                                {"from": "review-setup", "to": "implement-tool"},
                                {"from": "implement-tool", "to": "test-runtime"},
                            ],
                        }
                    ),
                    finish_reason="stop",
                    usage={"total_tokens": 6},
                    backend="ollama",
                ),
            ]
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=ai,
            )
            app.update_backend_access("ollama", True)

            result = app.run_plan("plan on how to add pdf generation tool to this codebase")

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["blueprint"]["tasks"][0]["task_id"], "inspect-web")
        self.assertEqual(len(ai.chat_calls), 2)

    def test_run_turn_sanitizes_replay_json_error_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": "\n".join(
                            [
                                json.dumps({"type": "step_start", "timestamp": 1}),
                                json.dumps(
                                    {
                                        "type": "tool_use",
                                        "part": {
                                            "type": "tool",
                                            "tool": "invalid",
                                            "state": {"input": {"error": "Model tried to call unavailable tool 'search_text'."}},
                                        },
                                    }
                                ),
                                json.dumps(
                                    {
                                        "type": "error",
                                        "error": {
                                            "name": "UnknownError",
                                            "data": {"message": "The user rejected permission to use this specific tool call."},
                                        },
                                    }
                                ),
                            ]
                        ),
                        "total_usage": {},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                    }
                },
            )()

            result = app.run_turn("hello")

        self.assertEqual(result["final_response"], "Permission to use a required tool call was denied.")

    def test_run_turn_collapses_repeated_final_response_blocks(self) -> None:
        repeated = (
            "Confidently, get-drip was described as a Convex-backed app, and the work focused on Create Workspace "
            "accepting https links and converting them internally and Salesforce being marked as coming soon or disabled, "
            "and the DRIP pipeline chat flow not working. What remains unclear is a cleaner one-line architecture summary beyond those clues."
        )
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": f"{repeated}\n\n{repeated}\n\n{repeated}",
                        "total_usage": {},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                    }
                },
            )()

            result = app.run_turn("hello")

        self.assertEqual(result["final_response"], repeated)

    def test_run_turn_collapses_affirmative_wrapped_duplicate_blocks(self) -> None:
        repeated = "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts."
        wrapped = f"Yes.\n\n{repeated}"
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": f"{repeated}\n\n{wrapped}\n\nYes. Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
                        "total_usage": {},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                    }
                },
            )()

            result = app.run_turn("hello")

        self.assertEqual(result["final_response"], repeated)

    def test_run_turn_trims_tool_output_noise_from_plain_text_response(self) -> None:
        noisy = (
            "Yes. The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.\n\n"
            "Devenv status\n\n"
            "Tool trace\n\n"
            "OpenCode\n\n"
            "Prepared the final answer\n\n"
            "Tool output: return lowered.endswith(\"?\") def _lexical_memory_terms(user_prompt: str) -> list[str]: ..."
        )
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": noisy,
                        "total_usage": {},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                    }
                },
            )()

            result = app.run_turn("hello")

        self.assertEqual(
            result["final_response"],
            "Yes. The get-drip cleanup was mainly about root URL redirects, Convex generated imports, and authentication bypass.",
        )

    def test_run_turn_extracts_assistant_answer_from_ui_transcript_dump(self) -> None:
        noisy = "\n".join(
            [
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
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.kernel.execute_turn = lambda prompt, **kwargs: type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {
                        "final_response": noisy,
                        "total_usage": {},
                        "metadata": {"backend_used": "opencode", "budget_state": {"blocked": False}},
                    }
                },
            )()

            result = app.run_turn("hello")

        self.assertEqual(
            result["final_response"],
            "Yes. The strongest clues point to src/convex-types.ts and src/convex-api.ts.",
        )

    def test_health_payload_exposes_indexing_progress_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            codex_root = workspace / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "07" / "04"
            sessions_dir.mkdir(parents=True)
            session_id = "health-session-1"
            (codex_root / "session_index.jsonl").write_text(
                json.dumps({"id": session_id, "thread_name": "Health payload session", "updated_at": "2026-07-04T10:00:00Z"}) + "\n",
                encoding="utf-8",
            )
            (sessions_dir / f"rollout-2026-07-04T10-00-00-{session_id}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "2026-07-04T10:00:00Z", "type": "session_meta", "payload": {"id": session_id, "cwd": str(workspace)}}),
                        json.dumps({"timestamp": "2026-07-04T10:00:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "remember this startup progress work"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app = DevenvWebApp(
                RunConfig(
                    workspace_path=tempdir,
                    external_session_configs=(
                        ExternalSessionProviderConfig(provider="codex", root_path=str(codex_root), index_path="session_index.jsonl"),
                    ),
                ),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            app.update_session_access("codex", True)
            for _ in range(40):
                if app.context_builder.indexing_status()["completed"]:
                    break
                time.sleep(0.05)
            health = app.build_health_payload()

        self.assertIn("indexing", health)
        self.assertEqual(health["indexing"]["providers"], ["codex"])
        self.assertTrue(health["indexing"]["completed"])
        self.assertEqual(health["indexing"]["total_sessions"], 1)

    def test_health_payload_tolerates_unreadable_opencode_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            bad_opencode_path = Path(tempdir) / "missing-opencode.db"
            app = DevenvWebApp(
                RunConfig(
                    workspace_path=tempdir,
                    external_session_configs=(
                        ExternalSessionProviderConfig(provider="opencode", root_path=str(bad_opencode_path)),
                    ),
                ),
                memory=FakeMemory(),
                ai=FakeAI(),
            )
            health = app.build_health_payload()

        self.assertEqual(len(health["context_sources"]), 1)
        self.assertEqual(health["context_sources"][0]["provider"], "opencode")
        self.assertFalse(health["context_sources"][0]["available"])


if __name__ == "__main__":
    unittest.main()
