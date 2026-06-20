from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ai.models import AIResponse
from core.runtime.models import CheckpointTask, ExecutionBlueprint
from core.runtime.models import PlanningMode, RunConfig
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
