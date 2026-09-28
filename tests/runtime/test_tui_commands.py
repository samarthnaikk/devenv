from __future__ import annotations

import unittest

from core.runtime.tui import PaletteEntry
from core.runtime.tui_commands import (
    CommandSpec,
    DevenvCommandProvider,
    build_command_registry,
    group_by_category,
)


class _FakeController:
    def palette_entries(self, query: str = "") -> list[PaletteEntry]:
        return [
            PaletteEntry("status", "Show status", "/status", "status summary"),
            PaletteEntry(
                "select_backend:opencode",
                "Use backend opencode [m]",
                "/backend opencode",
                "backend select",
            ),
            PaletteEntry("model:opencode:x", "Set opencode model to x", "/model opencode x", "model"),
            PaletteEntry(
                "agent:opencode",
                "Connect AI agent OpenCode [ready]",
                "/ai opencode",
                "ai agent",
            ),
            PaletteEntry(
                "toggle_provider:codex",
                "Toggle session provider codex [off]",
                "/permission provider codex on",
                "permission",
            ),
        ]


class RegistryTest(unittest.TestCase):
    def test_includes_palette_entries_and_help(self) -> None:
        specs = build_command_registry(_FakeController())
        commands = {spec.command for spec in specs}
        self.assertIn("/help", commands)
        self.assertIn("/status", commands)
        self.assertIn("/backend opencode", commands)

    def test_categories_are_derived(self) -> None:
        specs = {spec.entry_id: spec for spec in build_command_registry(_FakeController())}
        self.assertEqual(specs["select_backend:opencode"].category, "Backend")
        self.assertEqual(specs["model:opencode:x"].category, "Model")
        self.assertEqual(specs["agent:opencode"].category, "Agents")
        self.assertEqual(specs["toggle_provider:codex"].category, "Sources")
        self.assertEqual(specs["status"].category, "App")
        self.assertEqual(specs["help"].category, "App")

    def test_group_by_category_is_ordered(self) -> None:
        groups = group_by_category(build_command_registry(_FakeController()))
        names = [name for name, _ in groups]
        self.assertEqual(names, ["App", "Backend", "Model", "Agents", "Sources"])

    def test_provider_score_prefers_matches(self) -> None:
        spec = CommandSpec("x", "Show status", "/status", "status")
        hit = DevenvCommandProvider._score("status", spec)
        miss = DevenvCommandProvider._score("zzzzz", spec)
        self.assertGreater(hit, miss)


if __name__ == "__main__":
    unittest.main()
