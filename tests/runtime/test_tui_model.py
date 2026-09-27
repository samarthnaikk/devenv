from __future__ import annotations

import unittest

from core.ai.model_catalog import OpenCodeModelInfo
from core.runtime.tui_model import filter_models, format_cost, format_model_label


def _model(full_id: str, *, name: str = "", **cost: float) -> OpenCodeModelInfo:
    provider_id, model_id = full_id.split("/", 1)
    return OpenCodeModelInfo(
        provider_id=provider_id,
        model_id=model_id,
        name=name,
        cost_input=cost.get("cost_input"),
        cost_output=cost.get("cost_output"),
    )


class FormatCostTest(unittest.TestCase):
    def test_blank_without_cost(self) -> None:
        self.assertEqual(format_cost(_model("opencode/x")), "")

    def test_formats_integer_prices(self) -> None:
        model = _model("opencode/x", cost_input=3, cost_output=15)
        self.assertEqual(format_cost(model), "$3/$15 per 1M")

    def test_formats_fractional_prices(self) -> None:
        model = _model("opencode/x", cost_input=0.5, cost_output=1.5)
        self.assertEqual(format_cost(model), "$0.5/$1.5 per 1M")


class FormatLabelTest(unittest.TestCase):
    def test_includes_name_and_cost(self) -> None:
        model = _model("opencode/claude-sonnet-4", name="Claude Sonnet 4", cost_input=3, cost_output=15)
        self.assertEqual(
            format_model_label(model),
            "opencode/claude-sonnet-4  ·  Claude Sonnet 4  ·  $3/$15 per 1M",
        )

    def test_skips_redundant_name(self) -> None:
        model = _model("opencode/big-pickle", name="big-pickle")
        self.assertEqual(format_model_label(model), "opencode/big-pickle")


class FilterModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.models = [
            _model("opencode/claude-sonnet-4", name="Claude Sonnet 4"),
            _model("opencode/gpt-5-codex", name="GPT-5 Codex"),
            _model("anthropic/claude-opus-4-6", name="Claude Opus 4.6"),
        ]

    def test_empty_query_returns_all(self) -> None:
        self.assertEqual(len(filter_models(self.models, "   ")), 3)

    def test_matches_provider(self) -> None:
        result = filter_models(self.models, "anthropic")
        self.assertEqual([model.full_id for model in result], ["anthropic/claude-opus-4-6"])

    def test_matches_name_case_insensitively(self) -> None:
        result = filter_models(self.models, "codex")
        self.assertEqual([model.full_id for model in result], ["opencode/gpt-5-codex"])

    def test_multiple_tokens_are_anded(self) -> None:
        result = filter_models(self.models, "claude sonnet")
        self.assertEqual([model.full_id for model in result], ["opencode/claude-sonnet-4"])

    def test_no_match(self) -> None:
        self.assertEqual(filter_models(self.models, "zzz"), [])


if __name__ == "__main__":
    unittest.main()
