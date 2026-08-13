from __future__ import annotations

import tempfile
import unittest

from core.runtime.models import RunConfig, RuntimeTurnResult
from core.runtime.tui import DevenvTUIController


class FakeMemory:
    def record_working_memory(self, messages, active_state):
        return None

    def retrieve_context(self, current_prompt: str, top_k: int = 5):
        return type("Retrieval", (), {"markdown_context": ""})()

    def add_episodic_log(self, user_prompt, agent_response, node_id=None, metadata=None):
        return "log-1"


class FakeAI:
    def __init__(self) -> None:
        self.model = "fake-opencode-model"
        self.provider_label = "OpenCode CLI"
        self.preferred_backend = "opencode"
        self.last_backend_used = "opencode"
        self.backend_models = {
            "opencode": "fake-opencode-model",
            "ollama": "qwen2.5:3b",
            "llama_cpp": "qwen2.5-coder.gguf",
            "codex": "gpt-5-codex",
        }
        self.enabled_flags = {}
        self.registered_tools = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)

    def set_backend_preference(self, backend: str, **kwargs) -> None:
        self.preferred_backend = backend
        self.enabled_flags = dict(kwargs)

    def set_model(self, model: str) -> None:
        self.model = model
        self.backend_models[self.preferred_backend] = model

    def set_backend_model(self, backend: str, model: str) -> None:
        self.backend_models[backend] = model
        if backend == self.preferred_backend:
            self.model = model

    def status(self):
        return {
            backend: type(
                "Status",
                (),
                {"model": model},
            )()
            for backend, model in self.backend_models.items()
        }


class FakeKernel:
    def __init__(self) -> None:
        self.memory = FakeMemory()
        self.ai = FakeAI()
        self.tools = {}
        self.context_builder = None
        self.execute_turn_calls = []
        self.reset_calls = 0

    def register_tool(self, tool) -> None:
        self.tools[tool.name] = tool
        if hasattr(self.ai, "register_tool"):
            self.ai.register_tool(tool)

    def execute_turn(self, prompt: str, **kwargs):
        self.execute_turn_calls.append((prompt, kwargs))
        return RuntimeTurnResult(final_response="stub response")

    def reset_conversation(self) -> str:
        self.reset_calls += 1
        return "session-reset"

    def close(self) -> None:
        return None


class DevenvTUITest(unittest.TestCase):
    def test_permission_command_updates_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir, performance_mode="medium"),
                kernel=FakeKernel(),
            )

            backend_result = controller.handle_command("/permission backend opencode on")
            provider_result = controller.handle_command("/permission provider codex on")

        self.assertIn("now on", backend_result.message)
        self.assertIn("now on", provider_result.message)
        self.assertTrue(controller.access_policy.can_use_backend("opencode"))
        self.assertTrue(controller.access_policy.can_access_provider("codex"))
        self.assertEqual(controller.context_builder.runtime_allowed_providers, {"codex"})
        self.assertTrue(controller.kernel.ai.enabled_flags["opencode_enabled"])

    def test_backend_command_warns_when_backend_is_still_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/backend codex")

        self.assertIn("still blocked", result.message)
        self.assertEqual(controller.preferred_backend, "codex")

    def test_model_command_sets_backend_specific_model(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/model codex gpt-5-codex-high")

        self.assertIn("gpt-5-codex-high", result.message)
        self.assertEqual(controller.kernel.ai.backend_models["codex"], "gpt-5-codex-high")

    def test_run_prompt_passes_backend_and_permission_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir, max_consecutive_tools=7, no_memory=True),
                kernel=kernel,
            )
            controller.handle_command("/permission backend opencode on")
            controller.handle_command("/permission provider opencode on")

            controller.run_prompt("what were the bugs we found in get-drip")

        self.assertEqual(len(kernel.execute_turn_calls), 1)
        prompt, kwargs = kernel.execute_turn_calls[0]
        self.assertEqual(prompt, "what were the bugs we found in get-drip")
        self.assertEqual(kwargs["backend_preference"], "opencode")
        self.assertTrue(kwargs["opencode_enabled"])
        self.assertFalse(kwargs["codex_enabled"])
        self.assertTrue(controller.context_builder.runtime_allowed_providers == {"opencode"})
        self.assertTrue(kwargs["no_memory"])

    def test_clear_command_resets_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=kernel,
            )

            result = controller.handle_command("/clear")

        self.assertIn("session-reset", result.message)
        self.assertEqual(kernel.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
