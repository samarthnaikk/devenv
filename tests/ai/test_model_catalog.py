from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.ai.model_catalog import (
    ModelCatalogError,
    OpenCodeModelInfo,
    discover_opencode_models,
    list_opencode_model_ids,
    parse_model_list,
    resolve_model_id,
)

PLAIN_OUTPUT = "\n".join(
    [
        "opencode/claude-sonnet-4",
        "opencode/gpt-5-codex",
        "anthropic/claude-opus-4-6",
        "",
    ]
)

VERBOSE_OUTPUT = "\n".join(
    [
        "opencode/big-pickle",
        "{",
        '  "id": "big-pickle",',
        '  "providerID": "opencode",',
        '  "name": "Big Pickle",',
        '  "family": "big-pickle",',
        '  "cost": {"input": 0.5, "output": 1.5}',
        "}",
        "opencode/claude-sonnet-4",
        "{",
        '  "id": "claude-sonnet-4",',
        '  "providerID": "opencode",',
        '  "name": "Claude Sonnet 4",',
        '  "family": "claude-sonnet",',
        '  "cost": {"input": 3, "output": 15}',
        "}",
    ]
)


class ParseModelListTest(unittest.TestCase):
    def test_parses_plain_output(self) -> None:
        models = parse_model_list(PLAIN_OUTPUT)
        self.assertEqual(
            [model.full_id for model in models],
            ["opencode/claude-sonnet-4", "opencode/gpt-5-codex", "anthropic/claude-opus-4-6"],
        )

    def test_parses_verbose_metadata(self) -> None:
        models = parse_model_list(VERBOSE_OUTPUT)
        self.assertEqual([model.full_id for model in models], ["opencode/big-pickle", "opencode/claude-sonnet-4"])
        first = models[0]
        self.assertEqual(first.name, "Big Pickle")
        self.assertEqual(first.family, "big-pickle")
        self.assertEqual(first.cost_input, 0.5)
        self.assertEqual(first.cost_output, 1.5)

    def test_skips_duplicates(self) -> None:
        models = parse_model_list("opencode/x\nopencode/x\nopencode/y\n")
        self.assertEqual([model.full_id for model in models], ["opencode/x", "opencode/y"])

    def test_ignores_noise_lines(self) -> None:
        models = parse_model_list("refreshing models...\n\nopencode/z\nunknown line\n")
        self.assertEqual([model.full_id for model in models], ["opencode/z"])

    def test_label_falls_back_to_model_id(self) -> None:
        self.assertEqual(OpenCodeModelInfo("opencode", "x").label, "x")


class ProviderPreferenceTest(unittest.TestCase):
    def test_preferred_provider_sorts_first(self) -> None:
        from core.ai.model_catalog import _filter_provider, OpenCodeModelInfo

        models = [
            OpenCodeModelInfo("anthropic", "x"),
            OpenCodeModelInfo("opencode", "y"),
            OpenCodeModelInfo("opencode-go", "z"),
        ]
        ordered = _filter_provider(models, None)
        self.assertEqual(ordered[0].provider_id, "opencode-go")
        self.assertEqual(ordered[1].provider_id, "opencode")

    def test_explicit_provider_filter_still_works(self) -> None:
        from core.ai.model_catalog import _filter_provider, OpenCodeModelInfo

        models = [OpenCodeModelInfo("opencode", "y"), OpenCodeModelInfo("opencode-go", "z")]
        filtered = _filter_provider(models, "opencode-go")
        self.assertEqual([m.provider_id for m in filtered], ["opencode-go"])

    def test_default_fallback_is_go_longcat(self) -> None:
        from core.ai.model_catalog import DEFAULT_FALLBACK_MODELS

        self.assertEqual(DEFAULT_FALLBACK_MODELS[0], "opencode-go/longcat-2.5-preview-free")


class DiscoverModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "models.json"

    def _runner(self, calls: list[list[str]], output: str = PLAIN_OUTPUT):
        def run(command, _timeout):
            calls.append(list(command))
            return output

        return run

    def test_fetches_and_caches(self) -> None:
        calls: list[list[str]] = []
        first = discover_opencode_models(
            runner=self._runner(calls),
            cache_path=self.cache_path,
            executable="opencode",
            now=1000.0,
        )
        self.assertEqual(len(first), 3)
        self.assertEqual(calls[0], ["opencode", "models", "--verbose"])

        second = discover_opencode_models(
            runner=self._runner(calls),
            cache_path=self.cache_path,
            executable="opencode",
            now=1000.0 + 60,
        )
        self.assertEqual([model.full_id for model in second], [model.full_id for model in first])
        self.assertEqual(len(calls), 1, "second call should be served from cache")

    def test_refresh_bypasses_cache(self) -> None:
        calls: list[list[str]] = []
        discover_opencode_models(runner=self._runner(calls), cache_path=self.cache_path, now=1000.0)
        discover_opencode_models(
            runner=self._runner(calls), cache_path=self.cache_path, refresh=True, now=1000.0
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ["opencode", "models", "--verbose", "--refresh"])

    def test_falls_back_to_stale_cache_on_failure(self) -> None:
        calls: list[list[str]] = []
        models = discover_opencode_models(
            runner=self._runner(calls), cache_path=self.cache_path, now=1000.0
        )
        self.assertEqual(len(models), 3)

        def failing(_command, _timeout):
            raise ModelCatalogError("opencode exploded")

        result = discover_opencode_models(
            runner=failing,
            cache_path=self.cache_path,
            refresh=True,
            now=1000.0 + 100000,
        )
        self.assertEqual([model.full_id for model in result], [model.full_id for model in models])

    def test_falls_back_to_static_models_without_cache(self) -> None:
        def failing(_command, _timeout):
            raise OSError("opencode not installed")

        result = discover_opencode_models(
            runner=failing,
            cache_path=self.cache_path,
            fallback=["opencode/fallback-a", "opencode/fallback-b"],
            now=1000.0,
        )
        self.assertEqual(
            [model.full_id for model in result],
            ["opencode/fallback-a", "opencode/fallback-b"],
        )

    def test_provider_filter(self) -> None:
        models = discover_opencode_models(
            runner=self._runner([]),
            cache_path=self.cache_path,
            provider="anthropic",
            now=1000.0,
        )
        self.assertEqual([model.full_id for model in models], ["anthropic/claude-opus-4-6"])

    def test_empty_output_falls_back(self) -> None:
        result = discover_opencode_models(
            runner=lambda _c, _t: "",
            cache_path=self.cache_path,
            fallback=["opencode/only"],
            now=1000.0,
        )
        self.assertEqual([model.full_id for model in result], ["opencode/only"])

    def test_cache_file_is_valid_json(self) -> None:
        discover_opencode_models(runner=self._runner([]), cache_path=self.cache_path, now=1000.0)
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["fetched_at"], 1000.0)
        self.assertEqual(len(payload["models"]), 3)

    def test_list_ids_wrapper(self) -> None:
        identifiers = list_opencode_model_ids(
            runner=self._runner([]),
            cache_path=self.cache_path,
            provider="opencode",
            now=1000.0,
        )
        self.assertEqual(identifiers, ["opencode/claude-sonnet-4", "opencode/gpt-5-codex"])


class ResolveModelIdTest(unittest.TestCase):
    def test_corrects_stale_prefix_to_go(self) -> None:
        available = ["opencode-go/longcat-2.5-preview-free", "opencode/longcat-2.5-preview-free"]
        resolved = resolve_model_id("opencode/longcat-2.5-preview-free", available=available)
        self.assertEqual(resolved, "opencode-go/longcat-2.5-preview-free")

    def test_keeps_valid_id(self) -> None:
        available = ["opencode-go/longcat-2.5-preview-free"]
        self.assertEqual(
            resolve_model_id("opencode-go/longcat-2.5-preview-free", available=available),
            "opencode-go/longcat-2.5-preview-free",
        )

    def test_keeps_zen_only_model(self) -> None:
        available = ["opencode/big-pickle"]
        self.assertEqual(resolve_model_id("opencode/big-pickle", available=available), "opencode/big-pickle")

    def test_unknown_model_unchanged(self) -> None:
        self.assertEqual(resolve_model_id("foo/bar", available=["opencode-go/x"]), "foo/bar")

    def test_empty_and_bare_ids(self) -> None:
        self.assertEqual(resolve_model_id("", available=["opencode-go/x"]), "")
        self.assertEqual(resolve_model_id("bare-model", available=["opencode-go/x"]), "bare-model")


if __name__ == "__main__":
    unittest.main()
