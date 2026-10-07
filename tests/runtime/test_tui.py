from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.ai.model_catalog import OpenCodeModelInfo
from core.runtime.models import (
    CheckpointTask,
    ExecutionBlueprint,
    ExecutionMode,
    PlanningMode,
    RunConfig,
    RuntimeTurnResult,
    StageTrace,
    ToolExecutionStep,
)
from core.runtime.tui import (
    DevenvTUIController,
    RetrievalOutcome,
    TUILogBridge,
    _format_log_line,
    _format_plan_result_lines,
    _format_retrieval_result_lines,
    _format_retrieval_result_lines_plain,
    _format_turn_result_lines,
    _plan_plain_text,
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

    def chat(self, messages=None, **kwargs):
        self.chat_messages = messages
        return type("Response", (), {"content": "**Answer**\n- formatted from evidence"})()

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
        elapsed_ms=42,
    )


def _sample_blueprint() -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown="# Plan\n\n- [ ] Inspect the theme module\n- [ ] Add a dark-mode toggle",
        original_objective="add dark mode",
        tasks=[
            CheckpointTask(task_id=1, description="Inspect the theme module"),
            CheckpointTask(task_id=2, description="Add a dark-mode toggle"),
        ],
    )


class BlueprintKernel(FakeKernel):
    def __init__(self, blueprint: ExecutionBlueprint | None = None) -> None:
        super().__init__()
        self.blueprint = blueprint or _sample_blueprint()

    def execute_turn(self, prompt: str, **kwargs):
        self.execute_turn_calls.append((prompt, kwargs))
        return RuntimeTurnResult(
            final_response=self.blueprint.raw_plan_markdown,
            blueprint=self.blueprint,
            execution_mode=ExecutionMode.PLAN_ONLY.value,
            system_logs=[
                "Planning tool scope size: 2",
                "Planning tool scope: read_file, search_text",
            ],
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
    def setUp(self) -> None:
        # These tests exercise command wiring with arbitrary model ids; disable
        # provider-prefix normalization so they do not depend on the machine's
        # live OpenCode catalog/account. Normalization has its own tests.
        self._pref_patch = mock.patch(
            "core.ai.model_catalog._preferred_opencode_provider", lambda: ""
        )
        self._pref_patch.start()
        self.addCleanup(self._pref_patch.stop)

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

    def test_plan_command_requires_query(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=kernel,
            )

            result = controller.handle_command("/plan")

        self.assertEqual(result.message, "Usage: /plan <query>")
        self.assertEqual(kernel.execute_turn_calls, [])

    def test_plan_command_runs_plan_only_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=kernel,
            )

            controller.handle_command("/plan add dark mode")

        self.assertEqual(len(kernel.execute_turn_calls), 1)
        prompt, kwargs = kernel.execute_turn_calls[0]
        self.assertEqual(prompt, "add dark mode")
        self.assertTrue(kwargs["plan_only"])
        self.assertEqual(kwargs["planning_mode"], PlanningMode.FORCE_PLAN)

    def test_plan_command_renders_blueprint_read_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=BlueprintKernel(),
            )

            result = controller.handle_command("/plan add dark mode")

        self.assertIn("plan-only", result.message)
        self.assertIn("Checkpoints  2", result.message)
        self.assertIn("read_file, search_text", result.message)
        self.assertIn("Add a dark-mode toggle", result.message)

    def test_format_plan_result_lines_without_blueprint_falls_back_to_response(self) -> None:
        result = RuntimeTurnResult(final_response="- [ ] do the thing", blueprint=None)

        lines = _format_plan_result_lines(result)

        self.assertIn("Mode  plan-only (read-only, nothing executed)", lines)
        self.assertIn("- [ ] do the thing", lines)

    def test_plan_plain_text_marks_read_only(self) -> None:
        text = _plan_plain_text(
            RuntimeTurnResult(
                final_response="# Plan",
                blueprint=_sample_blueprint(),
            )
        )

        self.assertIn("plan-only", text)
        self.assertIn("Checkpoints  2", text)

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

    def test_selector_model_command_sets_selector_model(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/model selector opencode/claude-haiku-4-5")

        self.assertIn("Retrieval-selector model set", result.message)
        self.assertEqual(controller.get_selector_model(), "opencode/claude-haiku-4-5")

    def test_defaults_use_longcat_for_both_models_and_enable_selector(self) -> None:
        from core.runtime.tui import (
            DEFAULT_ASSISTANT_MODEL,
            DEFAULT_TUI_SELECTOR_MODEL,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tempdir:
                controller = DevenvTUIController(
                    RunConfig(workspace_path=tempdir),
                    kernel=FakeKernel(),
                )

        self.assertEqual(
            controller.kernel.ai.backend_models["opencode"], DEFAULT_ASSISTANT_MODEL
        )
        self.assertEqual(controller.get_selector_model(), DEFAULT_TUI_SELECTOR_MODEL)
        self.assertEqual(DEFAULT_ASSISTANT_MODEL, "opencode-go/longcat-2.5-preview-free")
        self.assertEqual(DEFAULT_TUI_SELECTOR_MODEL, "opencode-go/longcat-2.5-preview-free")

    def test_defaults_do_not_override_persisted_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first = DevenvTUIController(
                RunConfig(workspace_path=tempdir), kernel=FakeKernel()
            )
            first.handle_command("/model selector opencode/claude-haiku-4-5")
            first.close()

            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir), kernel=FakeKernel()
            )

        self.assertEqual(second.get_selector_model(), "opencode/claude-haiku-4-5")

    def test_assistant_model_command_sets_both_models(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command(
                "/assistant-model opencode/longcat-2.5-preview-free opencode/deepseek-v4.1-flash"
            )

        self.assertIn("opencode/longcat-2.5-preview-free", result.message)
        self.assertIn("Retrieval-selector model set", result.message)
        self.assertEqual(
            controller.kernel.ai.backend_models["opencode"], "opencode/longcat-2.5-preview-free"
        )
        self.assertEqual(controller.get_selector_model(), "opencode/deepseek-v4.1-flash")

    def test_assistant_model_command_answer_only_keeps_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.handle_command("/model selector opencode/claude-haiku-4-5")

            result = controller.handle_command("/assistant-model opencode/claude-sonnet-4")

        self.assertIn("opencode/claude-sonnet-4", result.message)
        self.assertEqual(controller.get_selector_model(), "opencode/claude-haiku-4-5")

    def test_assistant_model_show_reports_both(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.handle_command("/assistant-model opencode/claude-sonnet-4 opencode/claude-haiku-4-5")

            result = controller.handle_command("/assistant-model show")

        self.assertIn("answer model", result.message)
        self.assertIn("selector model", result.message)
        self.assertIn("opencode/claude-haiku-4-5", result.message)

    def test_assistant_model_palette_entry_present(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            entries = controller.palette_entries("assistant")

        commands = [entry.command for entry in entries]
        self.assertIn("/assistant-model", commands)

    def test_models_command_lists_models(self) -> None:
        models = [
            OpenCodeModelInfo("opencode", "claude-sonnet-4", name="Claude Sonnet 4", cost_input=3.0),
            OpenCodeModelInfo("anthropic", "claude-opus-4-6", name="Claude Opus 4.6"),
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch.object(controller, "available_models", return_value=models):
                result = controller.handle_command("/models")

        self.assertIn("OpenCode models (2)", result.message)
        self.assertIn("opencode (1)", result.message)
        self.assertIn("anthropic (1)", result.message)
        self.assertIn("claude-sonnet-4", result.message)

    def test_models_command_filters_by_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch.object(
                controller,
                "available_models",
                side_effect=lambda provider=None, **kwargs: (
                    [OpenCodeModelInfo("anthropic", "claude-opus-4-6")]
                    if provider == "anthropic"
                    else []
                ),
            ):
                result = controller.handle_command("/models unknown")

        self.assertIn("No models found for provider `unknown`", result.message)

    def test_model_refresh_command_reports_count(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch.object(
                controller,
                "available_models",
                return_value=[OpenCodeModelInfo("opencode", "x")],
            ) as patched:
                result = controller.handle_command("/model refresh")

        patched.assert_called_once_with(refresh=True)
        self.assertIn("Refreshed OpenCode model list (1 models)", result.message)

    def test_palette_entries_include_selector_and_models_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            models = [OpenCodeModelInfo("opencode", "claude-sonnet-4", name="Claude Sonnet 4")]
            with mock.patch.object(controller, "available_models", return_value=models):
                entries = controller.palette_entries("model")

        commands = [entry.command for entry in entries]
        self.assertIn("/models", commands)
        self.assertIn("/model refresh", commands)
        self.assertIn("/model selector", commands)
        self.assertIn("/model selector opencode/claude-sonnet-4", commands)

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

    def test_run_answer_formats_retrieved_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder(outcome=_sample_outcome())

            answer = controller.run_answer("how does retrieval work?")

        self.assertEqual(answer, "**Answer**\n- formatted from evidence")
        self.assertEqual(controller.last_answer_text, answer)

    def test_run_answer_abstains_without_evidence(self) -> None:
        from core.runtime.tui import INSUFFICIENT_EVIDENCE_MESSAGE

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder(
                outcome=RetrievalOutcome(query="empty", context="", session_ids=())
            )

            answer = controller.run_answer("anything?")

        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(controller.last_answer_text, INSUFFICIENT_EVIDENCE_MESSAGE)

    def test_run_answer_from_outcome_reuses_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            outcome = _sample_outcome()

            answer = controller.run_answer_from_outcome("how does retrieval work?", outcome)

        self.assertIsNotNone(answer)
        self.assertIn("formatted from evidence", answer or "")

    def test_run_retrieval_includes_local_memory_context(self) -> None:
        class _MemoryWithContext(FakeMemory):
            def retrieve_context(self, current_prompt: str, top_k: int = 5):
                return type(
                    "Retrieval",
                    (),
                    {"markdown_context": "## Retrieved Memory\n- inj_note: use the nonce XJ-42"},
                )()

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.kernel.memory = _MemoryWithContext()
            controller.context_builder = FakeContextBuilder(
                outcome=RetrievalOutcome(
                    query="what is inj_note",
                    context="## External Session Context\n- external line",
                    session_ids=("session-1",),
                    metadata={},
                )
            )

            outcome = controller.run_retrieval("what is inj_note")

        self.assertIn("inj_note", outcome.context)
        self.assertIn("external line", outcome.context)
        self.assertTrue(outcome.metadata.get("local_memory_context"))

    def test_ask_command_returns_formatted_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder(outcome=_sample_outcome())

            result = controller.handle_command("/ask how does retrieval work?")

        self.assertIn("formatted from evidence", result.message)

    def test_ask_command_requires_query(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/ask")

        self.assertIn("Usage: /ask", result.message)

    def test_stale_selector_prefix_is_normalized_on_load(self) -> None:
        catalog = ["opencode-go/longcat-2.5-preview-free", "opencode/longcat-2.5-preview-free"]
        with tempfile.TemporaryDirectory() as tempdir:
            state_dir = Path(tempdir) / ".devenv"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "tui_state.json").write_text(
                json.dumps({"selector_model": "opencode/longcat-2.5-preview-free"}),
                encoding="utf-8",
            )
            with mock.patch.object(
                DevenvTUIController,
                "_available_model_ids",
                lambda self: catalog,
            ), mock.patch(
                "core.ai.model_catalog._preferred_opencode_provider",
                lambda: "opencode-go",
            ):
                controller = DevenvTUIController(
                    RunConfig(workspace_path=tempdir),
                    kernel=FakeKernel(),
                )

        self.assertEqual(controller.selector_model, "opencode-go/longcat-2.5-preview-free")

    def test_normalize_keeps_zen_only_model(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch.object(
                DevenvTUIController,
                "_available_model_ids",
                lambda self: ["opencode/claude-fable-5"],
            ):
                normalized = controller._normalize_model_prefix("opencode/claude-fable-5")

        # claude-fable-5 exists under opencode/ (Zen) only, so it is unchanged.
        self.assertEqual(normalized, "opencode/claude-fable-5")

    def test_plan_is_persisted_and_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=BlueprintKernel(),
            )
            result = controller.run_plan("add dark mode")
            listing = controller.handle_command("/plans list")

        self.assertIn("plan_id", result.metadata)
        self.assertIn("Saved plans", listing.message)
        self.assertIn("dark mode", listing.message)

    def test_plans_show_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=BlueprintKernel(),
            )
            result = controller.run_plan("add dark mode")
            plan_id = result.metadata["plan_id"]
            shown = controller.handle_command(f"/plans show {plan_id}")
            exported = controller.handle_command(f"/plans export {plan_id}")

        self.assertIn("Inspect the theme module", shown.message)
        self.assertIn("Exported", exported.message)

    def test_plans_list_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            result = controller.handle_command("/plans")

        self.assertIn("No saved plans", result.message)

    def test_tag_and_untag_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            class _TagBuilder:
                def __init__(self) -> None:
                    self.tags: dict[str, list[str]] = {}

                def set_session_tag(self, unified_id: str, tag: str) -> None:
                    self.tags.setdefault(unified_id, []).append(tag.lower())

                def remove_session_tag(self, unified_id: str, tag: str) -> None:
                    self.tags.get(unified_id, []).remove(tag.lower())

                def list_session_tags(self, unified_id: str):
                    return [(tag, "user") for tag in self.tags.get(unified_id, [])]

                def list_sessions(self, provider: str):
                    return []

            builder = _TagBuilder()
            controller.context_builder = builder

            tagged = controller.handle_command("/tag opencode:s1 deploy")
            untagged = controller.handle_command("/untag opencode:s1 deploy")

        self.assertIn("Tagged", tagged.message)
        self.assertEqual(builder.tags["opencode:s1"], [])
        self.assertIn("Removed", untagged.message)

    def test_tag_command_requires_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            result = controller.handle_command("/tag opencode:s1")

        self.assertIn("Usage: /tag", result.message)

    def test_exclude_tag_command_sets_env(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch.dict(os.environ, {"DEVENV_EXCLUDE_TAGS": ""}):
                result = controller.handle_command("/exclude-tag derived")
                self.assertIn("derived", os.environ.get("DEVENV_EXCLUDE_TAGS", ""))

        self.assertIn("Excluded tags", result.message)

    def test_selector_model_persists_across_controller_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            first.handle_command("/model selector opencode/claude-haiku-4-5")
            first.close()

            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

        self.assertEqual(second.get_selector_model(), "opencode/claude-haiku-4-5")

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

    def test_format_retrieval_result_lines_includes_metrics(self) -> None:
        lines = _format_retrieval_result_lines(_sample_outcome())
        joined = "\n".join(lines)

        self.assertIn("providers: codex", joined)
        self.assertIn("42 ms", joined)
        self.assertIn("index: ready", joined)

    def test_plain_retrieval_lines_contain_no_markup(self) -> None:
        outcome = RetrievalOutcome(
            query="q",
            context="\n".join(
                [
                    "## External Session Context",
                    "- Tool output: backend-1 | [SQL: UPDATE alembic_version ... = '0011_add_qualifly_track']",
                    "- Assistant reported: use [bold]carefully[/] and [/]",
                ]
            ),
            session_ids=("session-1",),
            metadata={},
        )

        joined = "\n".join(_format_retrieval_result_lines_plain(outcome))

        # Plain output carries no escape backslashes, so markup parsing never runs.
        self.assertNotIn("\\[", joined)
        self.assertIn("[SQL:", joined)
        self.assertIn("session-1", joined)

    def test_plain_retrieval_lines_render_without_markup_errors(self) -> None:
        import io

        from rich.console import Console

        context = "\n".join(
            ["## External Session Context"]
            + [f"- Tool output: line {i} [SQL: x = '0011_add_qualifly_track'] [/]" for i in range(30)]
        )
        outcome = RetrievalOutcome(query="q", context=context, session_ids=("s",), metadata={})
        joined = "\n".join(_format_retrieval_result_lines_plain(outcome))

        console = Console(file=io.StringIO(), markup=False)
        console.print(joined)  # must not raise

    def test_format_turn_result_lines_escapes_markup(self) -> None:
        import io

        from rich.console import Console

        result = RuntimeTurnResult(
            final_response="answer with [bold]tags[/] and orphan [/] and [red]x",
            system_logs=["log with [red] markup"],
        )

        joined = "\n".join(_format_turn_result_lines(result))

        # Must render as markup without raising on adversarial content.
        Console(file=io.StringIO(), markup=True).print(joined)
        self.assertIn("\\[bold]", joined)
        self.assertIn("\\[/]", joined)

    def test_retrieval_plain_text_lists_sessions(self) -> None:
        text = _retrieval_plain_text(_sample_outcome())

        self.assertIn("Retrieved context", text)
        self.assertIn("session-1", text)

    def test_run_retrieval_records_elapsed_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            controller.context_builder = FakeContextBuilder(outcome=_sample_outcome())

            result = controller.run_retrieval("anything")

        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_format_log_line_includes_level_and_source(self) -> None:
        line = _format_log_line(0.0, logging.WARNING, "core.runtime.tui", "boom [x]")

        self.assertIn("WARNING", line)
        self.assertIn("core.runtime.tui", line)
        self.assertIn("boom", line)
        self.assertIn("\\[x]", line)

    def test_tui_log_bridge_forwards_records_to_queue(self) -> None:
        captured: "queue.Queue[tuple[float, int, str, str]]" = queue.Queue()
        bridge = TUILogBridge(captured)
        logger = logging.getLogger("tests.tui.log_bridge")
        logger.addHandler(bridge)
        logger.setLevel(logging.INFO)
        try:
            logger.info("hello %s", "world")
        finally:
            logger.removeHandler(bridge)

        created, levelno, name, message = captured.get_nowait()
        self.assertIsInstance(created, float)
        self.assertEqual(levelno, logging.INFO)
        self.assertEqual(name, "tests.tui.log_bridge")
        self.assertEqual(message, "hello world")

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

    def test_ai_list_command_lists_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/ai list")

        self.assertIn("opencode", result.message)
        self.assertIn("OpenCode", result.message)
        self.assertIsNone(result.agent)

    def test_ai_command_connects_to_available_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/opencode"):
                result = controller.handle_command("/ai opencode")

        self.assertEqual(result.agent, "opencode")
        self.assertEqual(controller.preferred_agent, "opencode")

    def test_ai_command_reports_unavailable_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch("core.ai.agents.shutil.which", return_value=None):
                result = controller.handle_command("/ai opencode")

        self.assertIsNone(result.agent)
        self.assertIn("unavailable", result.message)

    def test_ai_command_rejects_unknown_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            result = controller.handle_command("/ai not-an-agent")

        self.assertIn("Unknown AI agent", result.message)

    def test_ai_preference_persists_across_controller_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            first = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            with mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/opencode"):
                first.handle_command("/ai opencode")
            first.close()

            second = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

        self.assertEqual(second.preferred_agent, "opencode")

    def test_palette_entries_include_agent_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            entries = controller.palette_entries("opencode")

        self.assertTrue(any(entry.command == "/ai opencode" for entry in entries))

    def test_help_text_mentions_ai_command(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            help_text = controller.help_text()

        self.assertIn("/ai", help_text)

    def test_create_agent_session_builds_acp_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )

            session = controller.create_agent_session("opencode")

        self.assertEqual(session.spec.name, "opencode")
        self.assertEqual(session.workspace_path, str(Path(tempdir).resolve()))


_TEXTUAL_AVAILABLE = importlib.util.find_spec("textual") is not None

if _TEXTUAL_AVAILABLE:
    from core.runtime.tui import DevenvTextualApp


@unittest.skipIf(not _TEXTUAL_AVAILABLE, "textual is not installed")
class DevenvTextualAppTest(unittest.IsolatedAsyncioTestCase):
    async def test_sidebar_exposes_agents_section(self) -> None:
        import io

        from rich.console import Console

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                buffer = io.StringIO()
                Console(file=buffer, force_terminal=False, width=80).print(
                    app.query_one("#agents-list").render()
                )
                rendered = buffer.getvalue()
                agent_names = [option.spec.name for option in controller.available_agent_options()]

        for name in agent_names:
            self.assertIn(name, rendered)

    async def test_open_agents_action_pushes_picker(self) -> None:
        from core.runtime.tui_widgets import SelectionScreen

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_open_agents()
                await pilot.pause()
                self.assertIsInstance(app.screen, SelectionScreen)

    async def test_agents_binding_is_registered(self) -> None:
        bindings = {}
        for binding in DevenvTextualApp.BINDINGS:
            if isinstance(binding, tuple):
                bindings[binding[0]] = binding[1]
            else:
                bindings[binding.key] = binding.action
        self.assertEqual(bindings.get("f6"), "open_agents")

    def test_question_mark_binding_is_registered(self) -> None:
        bindings = {
            binding[0]: binding[1]
            for binding in DevenvTextualApp.BINDINGS
            if isinstance(binding, tuple)
        }
        self.assertEqual(bindings.get("question_mark"), "show_help")

    def test_command_palette_provider_registered(self) -> None:
        from core.runtime.tui_commands import DevenvCommandProvider

        self.assertIn(DevenvCommandProvider, DevenvTextualApp.COMMANDS)

    async def test_help_command_pushes_overlay(self) -> None:
        from core.runtime.tui_widgets import HelpOverlay

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/help")
                await pilot.pause()
                self.assertIsInstance(app.screen, HelpOverlay)

    async def test_backend_command_without_args_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/backend")
                await pilot.pause()
                self.assertEqual(controller.preferred_backend, "opencode")

    async def test_plan_command_without_query_shows_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=kernel,
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/plan")
                await pilot.pause()

        self.assertEqual(kernel.execute_turn_calls, [])

    async def test_plan_command_dispatches_plan_only_turn_from_app(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = FakeKernel()
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=kernel,
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/plan add dark mode")
                await app.workers.wait_for_complete()
                await pilot.pause()

        self.assertEqual(len(kernel.execute_turn_calls), 1)
        prompt, kwargs = kernel.execute_turn_calls[0]
        self.assertEqual(prompt, "add dark mode")
        self.assertTrue(kwargs["plan_only"])

    def test_slash_candidates_filter_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            state = type("State", (), {"text": "/mode", "cursor_position": 5})()
            values = [item.value for item in app._slash_candidates(state)]

        self.assertTrue(values)
        self.assertTrue(all(value.startswith("/") for value in values))

    def test_slash_candidates_ignore_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            state = type("State", (), {"text": "how does retrieval work", "cursor_position": 23})()

        self.assertEqual(app._slash_candidates(state), [])

    def test_themes_are_registered(self) -> None:
        from core.runtime.tui_theme import LIGHT_THEME_NAME, build_themes

        names = {theme.name for theme in build_themes()}
        self.assertEqual(names, {"devenv", LIGHT_THEME_NAME})

    async def test_app_uses_devenv_theme_and_can_switch(self) -> None:
        from core.runtime.tui_theme import LIGHT_THEME_NAME

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertEqual(app.theme, "devenv")
                app.theme = LIGHT_THEME_NAME
                await pilot.pause()
                self.assertEqual(app.theme, LIGHT_THEME_NAME)

    async def test_workspace_tabs_present(self) -> None:
        from textual.widgets import TabbedContent, TabPane

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                tab_ids = {pane.id for pane in app.query(TabPane)}
                self.assertEqual(
                    tab_ids, {"tab-retrieve", "tab-sessions", "tab-memory", "tab-logs"}
                )
                self.assertEqual(app.query_one("#workspace-tabs", TabbedContent).active, "tab-retrieve")

    async def test_tab_command_switches_active_tab(self) -> None:
        from textual.widgets import TabbedContent

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/tab logs")
                await pilot.pause()
                self.assertEqual(app.query_one("#workspace-tabs", TabbedContent).active, "tab-logs")

    async def test_toggle_logs_action_round_trips(self) -> None:
        from textual.widgets import TabbedContent

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.action_toggle_logs()
                await pilot.pause()
                self.assertEqual(app.query_one("#workspace-tabs", TabbedContent).active, "tab-logs")
                app.action_toggle_logs()
                await pilot.pause()
                self.assertEqual(app.query_one("#workspace-tabs", TabbedContent).active, "tab-retrieve")

    def test_backend_locality_classification(self) -> None:
        from core.runtime.tui import backend_locality

        with mock.patch.dict(os.environ, {"OPENCODE_SERVER_URL": "http://127.0.0.1:4096"}):
            self.assertEqual(backend_locality("opencode"), "local")
        with mock.patch.dict(os.environ, {"OPENCODE_SERVER_URL": "https://opencode.example.com"}):
            self.assertEqual(backend_locality("opencode"), "remote")
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}):
            self.assertEqual(backend_locality("codex"), "remote")

    async def test_receipts_command_pushes_overlay(self) -> None:
        from core.runtime.tui_widgets import TextOverlay

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/receipts")
                await pilot.pause()
                self.assertIsInstance(app.screen, TextOverlay)

    async def test_help_overlay_escape_dismisses_without_crashing(self) -> None:
        from core.runtime.tui_widgets import HelpOverlay

        with tempfile.TemporaryDirectory() as tempdir:
            controller = DevenvTUIController(
                RunConfig(workspace_path=tempdir),
                kernel=FakeKernel(),
            )
            app = DevenvTextualApp(controller)
            async with app.run_test() as pilot:
                await pilot.pause()
                app.run_command_line("/help")
                await pilot.pause()
                self.assertIsInstance(app.screen, HelpOverlay)
                # Any keypress must resolve bindings without crashing.
                await pilot.press("a")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, HelpOverlay)

    def test_style_honors_no_color(self) -> None:
        from core.runtime.tui import Ansi, _style

        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(_style("hi", Ansi.BOLD), "hi")
        with mock.patch.dict(os.environ, {"NO_COLOR": ""}):
            self.assertIn(Ansi.BOLD, _style("hi", Ansi.BOLD))

    def test_cli_flags_forward_to_run_tui(self) -> None:
        from core.runtime import tui as tui_module

        with mock.patch.object(tui_module, "run_tui", return_value=0) as run_tui_mock:
            with mock.patch.object(sys, "argv", ["devenv-run", ".", "--no-alt-screen", "--no-mouse"]):
                tui_module.main()

        _, kwargs = run_tui_mock.call_args
        self.assertTrue(kwargs["inline"])
        self.assertFalse(kwargs["mouse"])


if __name__ == "__main__":
    unittest.main()
