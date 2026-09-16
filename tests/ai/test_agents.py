from __future__ import annotations

import unittest
from unittest import mock

from core.ai.agents import (
    AGENTS,
    AgentSpec,
    agent_availability,
    available_agents,
    resolve_agent,
)


class AgentRegistryTest(unittest.TestCase):
    def test_registry_contains_opencode(self) -> None:
        spec = resolve_agent("opencode")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.command, "opencode")
        self.assertEqual(spec.args, ("acp",))

    def test_resolve_agent_is_case_insensitive(self) -> None:
        self.assertIs(resolve_agent("OpenCode"), AGENTS["opencode"])
        self.assertIsNone(resolve_agent("does-not-exist"))

    def test_agent_availability_reports_executable(self) -> None:
        spec = AgentSpec(name="fake", title="Fake", command="fake-agent", args=())
        with mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/fake-agent"):
            availability = agent_availability(spec)

        self.assertTrue(availability.available)
        self.assertEqual(availability.detail, "/usr/bin/fake-agent")

    def test_agent_availability_reports_missing_binary(self) -> None:
        spec = AgentSpec(name="fake", title="Fake", command="fake-agent", args=())
        with mock.patch("core.ai.agents.shutil.which", return_value=None):
            availability = agent_availability(spec)

        self.assertFalse(availability.available)
        self.assertIn("not found on PATH", availability.detail)

    def test_available_agents_lists_every_registered_agent(self) -> None:
        with mock.patch("core.ai.agents.shutil.which", return_value="/usr/bin/opencode"):
            names = [option.spec.name for option in available_agents()]

        self.assertEqual(names, list(AGENTS))


if __name__ == "__main__":
    unittest.main()
