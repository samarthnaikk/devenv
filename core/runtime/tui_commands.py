"""Command registry and Textual command-palette provider for the Devenv TUI.

The registry is a single source of truth derived from the controller's existing
``palette_entries`` plus a small set of presentation-only entries. Both the
command palette (``Ctrl+P``) and the help overlay read from it, so slash
commands stay discoverable without duplicating the command list.

This module contains no retrieval or model logic; it only maps labels to
existing controller commands and dispatches them back to the app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, AsyncIterator, Iterable

from textual.command import DiscoveryHit, Hit, Provider

try:  # rapidfuzz is a declared dependency, but stay defensive
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover

    _fuzz = None

if TYPE_CHECKING:  # pragma: no cover
    from .tui import DevenvTUIController


_CATEGORY_ORDER = ("App", "Backend", "Model", "Agents", "Sources")

# Presentation-only commands that are not part of the controller palette.
_EXTRA_COMMANDS: tuple[tuple[str, str, str, str, str], ...] = (
    ("help", "Show help & keyboard shortcuts", "/help", "help keys shortcuts bindings", "App"),
)


@dataclass(frozen=True)
class CommandSpec:
    """A single command exposed to the palette and the help overlay."""

    entry_id: str
    title: str
    command: str
    keywords: str
    category: str = "App"

    @property
    def help(self) -> str:
        return self.command


def _category_for(entry_id: str) -> str:
    eid = entry_id.lower()
    if eid.startswith(("toggle_backend:", "select_backend:")):
        return "Backend"
    if eid.startswith(("model:", "selector_model:", "models_")) or eid in {
        "model_selector_pick",
        "assistant_model_pick",
    }:
        return "Model"
    if eid.startswith("toggle_provider:"):
        return "Sources"
    if eid.startswith("agent:"):
        return "Agents"
    return "App"


def build_command_registry(controller: "DevenvTUIController") -> list[CommandSpec]:
    """Build the full, unfiltered command list for the given controller."""

    specs: list[CommandSpec] = []
    for entry_id, title, command, keywords, category in _EXTRA_COMMANDS:
        specs.append(CommandSpec(entry_id, title, command, keywords, category))
    for entry in controller.palette_entries(""):
        specs.append(
            CommandSpec(
                entry.entry_id,
                entry.label,
                entry.command,
                entry.keywords,
                _category_for(entry.entry_id),
            )
        )
    return specs


def group_by_category(
    specs: Iterable[CommandSpec],
) -> list[tuple[str, list[CommandSpec]]]:
    """Group specs by category in a stable display order."""

    grouped: dict[str, list[CommandSpec]] = {category: [] for category in _CATEGORY_ORDER}
    for spec in specs:
        grouped.setdefault(spec.category, []).append(spec)
    return [(category, grouped[category]) for category in _CATEGORY_ORDER if grouped.get(category)]


class DevenvCommandProvider(Provider):
    """Feeds the controller's commands into Textual's built-in command palette."""

    _CACHE_SECONDS = 5.0
    _MIN_SCORE = 55.0

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._specs: list[CommandSpec] = []
        self._built_at = 0.0

    def _specs_now(self) -> list[CommandSpec]:
        now = time.monotonic()
        if not self._specs or (now - self._built_at) > self._CACHE_SECONDS:
            controller = getattr(self.app, "controller", None)
            self._specs = build_command_registry(controller) if controller is not None else []
            self._built_at = now
        return self._specs

    @staticmethod
    def _score(query: str, spec: CommandSpec) -> float:
        haystack = f"{spec.title} {spec.command} {spec.keywords}"
        if _fuzz is not None:
            return float(_fuzz.WRatio(query, haystack))
        return 100.0 if query.lower() in haystack.lower() else 0.0

    async def search(self, query: str) -> AsyncIterator[Hit]:
        query = (query or "").strip()
        scored: list[tuple[float, CommandSpec]] = []
        for spec in self._specs_now():
            score = self._score(query, spec) if query else 100.0
            if not query or score >= self._MIN_SCORE:
                scored.append((score, spec))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, spec in scored[:40]:
            yield Hit(
                score / 100.0,
                spec.title,
                partial(self._dispatch, spec.command),
                help=spec.command,
            )

    async def discover(self) -> AsyncIterator[DiscoveryHit]:
        preferred = (
            "help",
            "status",
            "models_list",
            "model_selector_pick",
            "assistant_model_pick",
            "providers",
        )
        by_id = {spec.entry_id: spec for spec in self._specs_now()}
        for entry_id in preferred:
            spec = by_id.get(entry_id)
            if spec is not None:
                yield DiscoveryHit(
                    spec.title,
                    partial(self._dispatch, spec.command),
                    help=spec.command,
                )

    def _dispatch(self, command: str) -> None:
        runner = getattr(self.app, "run_command_line", None)
        if callable(runner):
            runner(command)


__all__ = [
    "CommandSpec",
    "DevenvCommandProvider",
    "build_command_registry",
    "group_by_category",
]
