from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse
from core.runtime.models import CheckpointTask, ExecutionBlueprint
from core.runtime.models import ExternalSessionProviderConfig, PlanningMode, RunConfig
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
        self.model = "fake-groq-model"
        self.responses = [
            AIResponse(content="Website response", tool_calls=(), finish_reason="stop", usage={"prompt_tokens": 3}),
        ]
        self.registered_tools: list[str] = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
        tool_names=None,
    ) -> AIResponse:
        return self.responses.pop(0)


class DevenvWebAppTest(unittest.TestCase):
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
        self.assertEqual(health["ai_provider"], "Groq")
        self.assertEqual(health["ai_model"], "fake-groq-model")
        self.assertIn("fake-groq-model", health["available_models"])
        self.assertTrue(health["context_builder_enabled"])
        self.assertIn("context_sources", health)
        self.assertEqual(files["entries"][0]["name"], "README.md")
        self.assertEqual(file_payload["content"], "hello")
        self.assertEqual(file_payload["kind"], "text")

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
            turn = app.run_turn("hello")

            self.assertEqual(health["status"], "ok")
            self.assertEqual(files["entries"][0]["name"], "README.md")
            self.assertEqual(turn["final_response"], "Website response")
            self.assertEqual(turn["blueprint"]["tasks"][0]["description"], "hello")
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

    def test_set_model_updates_health_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            app = DevenvWebApp(
                RunConfig(workspace_path=tempdir),
                memory=FakeMemory(),
                ai=FakeAI(),
            )

            app.update_backend_access("opencode", True)
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
            health = app.build_health_payload()

        self.assertTrue(session_payload["session_access"]["codex"])
        self.assertTrue(backend_payload["backend_access"]["opencode"])
        self.assertTrue(health["access_policy"]["session_access"]["codex"])
        self.assertTrue(health["access_policy"]["backend_access"]["opencode"])

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


if __name__ == "__main__":
    unittest.main()
