"""Model-labelling helpers for the Devenv TUI.

The interactive model picker now lives in :mod:`core.runtime.tui_widgets` as the
shared :class:`~core.runtime.tui_widgets.SelectionScreen`. This module keeps the
pure formatting/filtering helpers used by the picker and the tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.ai.model_catalog import OpenCodeModelInfo


def format_cost(model: OpenCodeModelInfo) -> str:
    if model.cost_input is None and model.cost_output is None:
        return ""
    cost_in = "" if model.cost_input is None else _format_price(model.cost_input)
    cost_out = "" if model.cost_output is None else _format_price(model.cost_output)
    return f"${cost_in}/${cost_out} per 1M"


def _format_price(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def format_model_label(model: OpenCodeModelInfo) -> str:
    parts = [model.full_id]
    if model.name and model.name.lower() != model.model_id.lower():
        parts.append(model.name)
    cost = format_cost(model)
    if cost:
        parts.append(cost)
    return "  ·  ".join(parts)


def format_model_detail(model: OpenCodeModelInfo) -> str:
    """Secondary text for a model row: display name and cost, without the id."""
    parts: list[str] = []
    if model.name and model.name.lower() != model.model_id.lower():
        parts.append(model.name)
    cost = format_cost(model)
    if cost:
        parts.append(cost)
    return "  ·  ".join(parts)


def filter_models(
    models: Sequence[OpenCodeModelInfo],
    query: str,
) -> list[OpenCodeModelInfo]:
    tokens = [token for token in query.strip().lower().split() if token]
    if not tokens:
        return list(models)
    matched: list[OpenCodeModelInfo] = []
    for model in models:
        haystack = " ".join(
            [model.full_id, model.name, model.family, model.provider_id, model.model_id]
        ).lower()
        if all(token in haystack for token in tokens):
            matched.append(model)
    return matched


__all__ = [
    "filter_models",
    "format_cost",
    "format_model_detail",
    "format_model_label",
]
