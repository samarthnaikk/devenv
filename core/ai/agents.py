"""Registry of external AI agents Devenv can connect to natively.

Agents are launched by the runtime and spoken to over the Agent Client
Protocol (ACP). This module is intentionally tiny and dependency-free so it can
be imported from both the TUI and plain-Python fallbacks without pulling in the
ACP transport.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    """A launchable agent that speaks ACP over stdio."""

    name: str
    title: str
    command: str
    args: tuple[str, ...] = ()
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def launch_command(self) -> list[str]:
        return [self.command, *self.args]


AGENTS: dict[str, AgentSpec] = {
    "opencode": AgentSpec(
        name="opencode",
        title="OpenCode",
        command="opencode",
        args=("acp",),
        description="Native OpenCode agent over ACP (tools, permissions, MCP, AGENTS.md).",
    ),
}


@dataclass(frozen=True)
class AgentAvailability:
    spec: AgentSpec
    available: bool
    detail: str


def resolve_agent(name: str) -> AgentSpec | None:
    cleaned = str(name or "").strip().lower()
    return AGENTS.get(cleaned)


def agent_availability(spec: AgentSpec) -> AgentAvailability:
    executable = shutil.which(spec.command)
    if executable:
        return AgentAvailability(spec=spec, available=True, detail=executable)
    return AgentAvailability(
        spec=spec,
        available=False,
        detail=f"`{spec.command}` was not found on PATH.",
    )


def available_agents() -> list[AgentAvailability]:
    return [agent_availability(spec) for spec in AGENTS.values()]


__all__ = [
    "AGENTS",
    "AgentAvailability",
    "AgentSpec",
    "agent_availability",
    "available_agents",
    "resolve_agent",
]
