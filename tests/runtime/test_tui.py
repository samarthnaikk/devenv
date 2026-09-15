from __future__ import annotations

import tempfile
import unittest

from core.runtime.models import RunConfig, RuntimeTurnResult, StageTrace, ToolExecutionStep
from core.runtime.tui import (
    DevenvTUIController,
    RetrievalOutcome,
    _format_retrieval_result_lines,
    _format_turn_result_lines,
    _retrieval_plain_text,
)


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
        metadata_by_backend = {
            "opencode": {"models": ["opencode/claude-sonnet-4", "opencode/gpt-5-codex"]},
            "ollama": {"models": ["qwen2.5:3b", "qwen2.5-coder:7b"]},
            "llama_cpp": {"models": ["qwen2.5-coder.gguf", "deepseek-coder.gguf"]},
            "codex": {"models": ["gpt-5-codex", "gpt-5-codex-high"]},
        }
        return {
            backend: type(
                "Status",
                (),
                {"model": model, "metadata": metadata_by_backend.get(backend, {})},
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


class FakeContextBuilder:
    def __init__(self, outcome: RetrievalOutcome | None = None) -> None:
        self.runtime_allowed_providers: set[str] | None = None
        self.requests: list[tuple[str, int]] = []
        self.outcome = outcome or RetrievalOutcome(query="stub")
        self.index_ready = True

    def set_runtime_allowed_providers(self, providers) -> None:
        self.runtime_allowed_providers = set(providers or set())

    def build_runtime_memory_context(self, query: str, *, provider_name=None, max_lines: int = 6):
        self.requests.append((query, max_lines))
        return self.outcome.context, self.outcome.session_ids, dict(self.outcome.metadata)

    def indexing_status(self) -> dict[str, object]:
        return {"active": False, "completed": self.index_ready, "message": "ready", "percent": 100}


def _sample_outcome() -> RetrievalOutcome:
    context = "\n".join(
        [
            "## External Session Context",
            "- Session 'Retrieval engine work' targeted workspace /repo/devenv.",
            "- Assistant reported: The retrieval engine fuses lexical and semantic recall before selecting chunks.",
            "- User asked: How does the retrieval engine work?",
        ]
    )
    return RetrievalOutcome(
        query="retrieval engine",
        context=context,
        session_ids=("session-1",),
        metadata={"context_match_providers": ["codex"], "index_ready": True},
    )


class PromptFeeder:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("PromptFeeder ran out of answers")
        return self.answers.pop(0)


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

    def test_permission_command_opens_interactive_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            feeder = PromptFeeder(["1", "1", "1"])
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir, performance_mode="medium"),
                kernel=FakeKernel(),
                prompt_input=feeder,
            )

            result = controller.handle_command("/permission")

        self.assertIn("Backend `opencode` permission is now on.", result.message)
        self.assertTrue(controller.access_policy.can_use_backend("opencode"))

    def test_backend_command_opens_interactive_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            feeder = PromptFeeder(["4"])
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
                prompt_input=feeder,
            )

            result = controller.handle_command("/backend")

        self.assertIn("Preferred backend set to `codex`", result.message)
        self.assertEqual(controller.preferred_backend, "codex")

    def test_model_command_opens_interactive_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            feeder = PromptFeeder(["4", "2"])
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
                prompt_input=feeder,
            )

            result = controller.handle_command("/model")

        self.assertIn("gpt-5-codex-high", result.message)
        self.assertEqual(controller.kernel.ai.backend_models["codex"], "gpt-5-codex-high")

    def test_palette_entries_include_toggle_and_model_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            entries = controller.palette_entries("codex")

        labels = [entry.label for entry in entries]
        self.assertTrue(any("Toggle backend codex" in label for label in labels))
        self.assertTrue(any("Use backend codex" in label for label in labels))
        self.assertTrue(any("Set codex model to gpt-5-codex-high" in label for label in labels))

    def test_model_options_use_backend_reported_models(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            models = controller.model_options_for_backend("codex")

        self.assertIn("gpt-5-codex", models)
        self.assertIn("gpt-5-codex-high", models)

    def test_tui_state_persists_across_controller_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            first.handle_command("/permission backend opencode on")
            first.handle_command("/permission provider codex on")
            first.handle_command("/backend codex")
            first.handle_command("/model codex gpt-5-codex-high")
            first.close()

            second_kernel = FakeKernel()
            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=second_kernel,
            )

            self.assertTrue(second.access_policy.can_use_backend("opencode"))
            self.assertTrue(second.access_policy.can_access_provider("codex"))
            self.assertEqual(second.preferred_backend, "codex")
            self.assertEqual(second.kernel.ai.backend_models["codex"], "gpt-5-codex-high")
            self.assertEqual(second.context_builder.runtime_allowed_providers, {"codex"})

    def test_run_retrieval_delegates_to_context_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            fake = FakeContextBuilder(outcome=_sample_outcome())
            controller.context_builder = fake
            controller.handle_command("/permission provider codex on")

            result = controller.run_retrieval("how does retrieval work?")

        self.assertEqual(fake.requests, [("how does retrieval work?", 12)])
        self.assertEqual(result.session_ids, ("session-1",))
        self.assertIn("fuses lexical and semantic recall", result.context)
        self.assertEqual(fake.runtime_allowed_providers, {"codex"})

    def test_retrieve_command_returns_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder(outcome=_sample_outcome())

            command_result = controller.handle_command("/retrieve retrieval engine")

        self.assertIn("Retrieval engine work", command_result.message)
        self.assertIn("fuses lexical and semantic recall", command_result.message)

    def test_mode_command_switches_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            result = first.handle_command("/mode solve")
            first.close()

            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

        self.assertIn("in progress", result.message)
        self.assertEqual(second.mode, "solve")

    def test_set_source_enabled_updates_runtime_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder()

            message = controller.set_source_enabled("opencode", True)
            first_close = controller.enabled_sources()
            controller.close()

            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

        self.assertIn("enabled", message)
        self.assertEqual(first_close, ["opencode"])
        self.assertTrue(second.access_policy.can_access_provider("opencode"))

    def test_format_retrieval_result_lines_renders_context(self) -> None:
        lines = _format_retrieval_result_lines(_sample_outcome())
        joined = "\n".join(lines)

        self.assertIn("Retrieval engine work", joined)
        self.assertIn("fuses lexical and semantic recall", joined)
        self.assertIn("retrieval engine", joined)
        self.assertIn("session-1", joined)

    def test_format_retrieval_result_lines_handles_empty(self) -> None:
        outcome = RetrievalOutcome(
            query="unknown topic",
            metadata={"context_match_reason": "No strong prior-session match was found."},
        )

        lines = _format_retrieval_result_lines(outcome)
        joined = "\n".join(lines)

        self.assertIn("no matches", joined)
        self.assertIn("No strong prior-session match was found.", joined)

    def test_format_retrieval_result_lines_escapes_markup(self) -> None:
        outcome = RetrievalOutcome(
            query="q",
            context="\n".join(
                [
                    "## External Session Context",
                    "- Session 'weird [red] title' targeted workspace /repo.",
                    "- Assistant reported: use [bold]carefully[/]",
                ]
            ),
            session_ids=("session-1",),
            metadata={},
        )

        joined = "\n".join(_format_retrieval_result_lines(outcome))

        self.assertIn("\\[red]", joined)
        self.assertIn("\\[bold]", joined)

    def test_retrieval_plain_text_lists_sessions(self) -> None:
        text = _retrieval_plain_text(_sample_outcome())

        self.assertIn("Retrieved context", text)
        self.assertIn("session-1", text)

    def test_format_turn_result_lines_includes_thinking_and_system_logs(self) -> None:
        result = RuntimeTurnResult(
            final_response="stub response",
            ai_logs=["Assistant produced direct response"],
            system_logs=["Direct memory chars sent: 42"],
            stage_traces=[StageTrace(stage="brain", success=True, summary="Answered directly", logs=["Used focused memory"])],
            steps=[
                ToolExecutionStep(
                    step_id="1",
                    tool_name="read_file",
                    arguments={"path": "note.txt"},
                    output="ok",
                    success=True,
                    is_sandboxed_violation=False,
                )
            ],
        )

        lines = _format_turn_result_lines(result)

        self.assertTrue(any("thinking" in line and "Assistant produced direct response" in line for line in lines))
        self.assertTrue(any("system" in line and "Direct memory chars sent: 42" in line for line in lines))
        self.assertTrue(any("Used focused memory" in line for line in lines))
        self.assertTrue(any("Assistant" in line and "stub response" in line for line in lines))

    def test_format_turn_result_lines_shows_empty_response_message(self) -> None:
        lines = _format_turn_result_lines(RuntimeTurnResult(final_response=None))

        self.assertEqual(lines, ["[yellow]Assistant[/] The runtime completed without producing a visible response."])


if __name__ == "__main__":
    unittest.main()
