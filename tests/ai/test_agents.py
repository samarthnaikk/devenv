from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.ai.agents import (
    AGENTS,
    AgentSpec,
    Launch,
    agent_availability,
    agent_registry,
    available_agents,
    load_custom_agents,
    resolve_agent,
    resolve_launch,
)


class AgentRegistryTest(unittest.TestCase):
    def test_registry_contains_expected_agents(self) -> None:
        self.assertEqual(set(AGENTS), {"opencode", "gemini", "claude", "codex"})

    def test_opencode_uses_native_acp(self) -> None:
        spec = resolve_agent("opencode")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.command, "opencode")
        self.assertEqual(spec.args, ("acp",))

    def test_gemini_uses_native_acp(self) -> None:
        spec = resolve_agent("gemini")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.command, "gemini")
        self.assertEqual(spec.args, ("--acp",))

    def test_claude_and_codex_have_npx_fallback(self) -> None:
        claude = resolve_agent("claude")
        codex = resolve_agent("codex")

        assert claude is not None and codex is not None
        self.assertEqual(claude.launches[-1].command, "npx")
        self.assertIn("@zed-industries/claude-code-acp", claude.launches[-1].args)
        self.assertEqual(codex.launches[-1].command, "npx")
        self.assertIn("@agentclientprotocol/codex-acp", codex.launches[-1].args)

    def test_resolve_agent_is_case_insensitive(self) -> None:
        self.assertIs(resolve_agent("OpenCode"), AGENTS["opencode"])
        self.assertIsNone(resolve_agent("does-not-exist"))

    def test_agent_availability_reports_executable(self) -> None:
        spec = AgentSpec(name="fake", title="Fake", launches=(Launch(command="fake-agent"),))
        with mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/fake-agent"):
            availability = agent_availability(spec)

        self.assertTrue(availability.available)
        self.assertEqual(availability.detail, "/usr/bin/fake-agent")
        self.assertEqual(availability.launch.command, "fake-agent")

    def test_agent_availability_reports_missing_binary(self) -> None:
        spec = AgentSpec(name="fake", title="Fake", launches=(Launch(command="fake-agent"),))
        with mock.patch("core.ai.agents.shutil.which", return_value=None):
            availability = agent_availability(spec)

        self.assertFalse(availability.available)
        self.assertIn("not found on PATH", availability.detail)

    def test_agent_availability_uses_install_hint_when_missing(self) -> None:
        spec = AgentSpec(
            name="fake",
            title="Fake",
            launches=(Launch(command="fake-agent"),),
            install_hint="npm install -g fake-agent",
        )
        with mock.patch("core.ai.agents.shutil.which", return_value=None):
            availability = agent_availability(spec)

        self.assertFalse(availability.available)
        self.assertEqual(availability.detail, "npm install -g fake-agent")

    def test_agent_availability_falls_back_to_second_launch(self) -> None:
        spec = AgentSpec(
            name="fake",
            title="Fake",
            launches=(
                Launch(command="missing-binary"),
                Launch(command="npx", args=("-y", "@fake/agent")),
            ),
        )

        def which(command: str) -> str | None:
            return "/usr/bin/npx" if command == "npx" else None

        with mock.patch("core.ai.agents.shutil.which", side_effect=which):
            availability = agent_availability(spec)

        self.assertTrue(availability.available)
        self.assertEqual(availability.launch.command, "npx")
        self.assertEqual(resolve_launch(spec).args, ("-y", "@fake/agent"))

    def test_available_agents_lists_every_registered_agent(self) -> None:
        with (
            mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/opencode"),
            mock.patch("core.ai.agents.custom_agents_path", return_value=Path("/nonexistent/agents.json")),
        ):
            names = [option.spec.name for option in available_agents()]

        self.assertEqual(names, list(AGENTS))


class CustomAgentConfigTest(unittest.TestCase):
    def test_load_custom_agents_parses_single_command(self) -> None:
        payload = {
            "agent_servers": {
                "my-agent": {
                    "title": "My Agent",
                    "command": "node",
                    "args": ["/tmp/agent.js", "--acp"],
                    "env": {"FOO": "bar"},
                }
            }
        }
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "agents.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            agents = load_custom_agents(path)

        self.assertIn("my-agent", agents)
        spec = agents["my-agent"]
        self.assertEqual(spec.title, "My Agent")
        self.assertEqual(spec.command, "node")
        self.assertEqual(spec.args, ("/tmp/agent.js", "--acp"))
        self.assertEqual(spec.env, {"FOO": "bar"})

    def test_load_custom_agents_parses_launches(self) -> None:
        payload = {
            "agent_servers": {
                "multi": {
                    "launches": [
                        {"command": "multi-acp"},
                        {"command": "npx", "args": ["-y", "multi-acp"], "label": "npx"},
                    ]
                }
            }
        }
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "agents.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            agents = load_custom_agents(path)

        self.assertEqual(len(agents["multi"].launches), 2)
        self.assertEqual(agents["multi"].launches[1].label, "npx")

    def test_load_custom_agents_missing_file_is_empty(self) -> None:
        self.assertEqual(load_custom_agents(Path("/nonexistent/agents.json")), {})

    def test_load_custom_agents_invalid_json_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "agents.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_custom_agents(path), {})

    def test_agent_registry_merges_and_overrides(self) -> None:
        payload = {
            "agent_servers": {
                "opencode": {"command": "opencode-custom", "args": ["acp"]},
                "brand-new": {"command": "brand-new-acp"},
            }
        }
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "agents.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch("core.ai.agents.custom_agents_path", return_value=path):
                registry = agent_registry()
                resolved = resolve_agent("brand-new")

        self.assertEqual(registry["opencode"].command, "opencode-custom")
        self.assertIn("brand-new", registry)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.command, "brand-new-acp")


if __name__ == "__main__":
    unittest.main()
