"""Registry of external AI agents Devenv can connect to natively.

Agents are launched by the runtime and spoken to over the Agent Client
Protocol (ACP). This module is intentionally tiny and dependency-free so it can
be imported from both the TUI and plain-Python fallbacks without pulling in the
ACP transport.

A spec may declare several launch strategies in priority order, such as a
globally-installed adapter binary first and an ``npx`` fallback second. Users can
extend the registry without touching code by declaring ``agent_servers`` in
``agents.json``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Launch:
    """A single way to launch an ACP agent process over stdio."""

    command: str
    args: tuple[str, ...] = ()
    label: str = ""

    def launch_command(self) -> list[str]:
        return [self.command, *self.args]


@dataclass(frozen=True)
class AgentSpec:
    """A launchable agent that speaks ACP over stdio."""

    name: str
    title: str
    launches: tuple[Launch, ...] = ()
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)
    install_hint: str = ""
    auth_hint: str = ""

    def __post_init__(self) -> None:
        if not self.launches:
            raise ValueError(f"AgentSpec `{self.name}` must define at least one launch strategy.")

    @property
    def primary_launch(self) -> Launch:
        return self.launches[0]

    @property
    def command(self) -> str:
        return self.launches[0].command

    @property
    def args(self) -> tuple[str, ...]:
        return self.launches[0].args

    def launch_command(self) -> list[str]:
        return self.launches[0].launch_command()


AGENTS: dict[str, AgentSpec] = {
    "opencode": AgentSpec(
        name="opencode",
        title="OpenCode",
        launches=(Launch(command="opencode", args=("acp",), label="native"),),
        description="Native OpenCode agent over ACP (tools, permissions, MCP, AGENTS.md).",
        install_hint="Install OpenCode from https://opencode.ai and ensure `opencode` is on PATH.",
        auth_hint="Authenticate with `opencode auth login`.",
    ),
    "gemini": AgentSpec(
        name="gemini",
        title="Gemini CLI",
        launches=(Launch(command="gemini", args=("--acp",), label="native"),),
        description="Google Gemini CLI over ACP (native `--acp` mode).",
        install_hint="Install Gemini CLI: `npm install -g @google/gemini-cli`.",
        auth_hint="Authenticate by running `gemini` once and completing the login flow.",
    ),
    "claude": AgentSpec(
        name="claude",
        title="Claude Code",
        launches=(
            Launch(command="claude-code-acp", label="global"),
            Launch(command="npx", args=("-y", "@zed-industries/claude-code-acp"), label="npx"),
        ),
        description="Claude Code via the Claude Agent SDK ACP adapter.",
        install_hint="Install: `npm install -g @zed-industries/claude-code-acp` (or use the npx fallback).",
        auth_hint="Authenticate with `claude login` or set ANTHROPIC_API_KEY.",
    ),
    "codex": AgentSpec(
        name="codex",
        title="Codex",
        launches=(
            Launch(command="codex-acp", label="global"),
            Launch(command="npx", args=("-y", "@agentclientprotocol/codex-acp"), label="npx"),
        ),
        description="OpenAI Codex via the Codex App Server ACP adapter.",
        install_hint="Install: `npm install -g @agentclientprotocol/codex-acp` (or use the npx fallback).",
        auth_hint="Authenticate with `codex login` or set OPENAI_API_KEY / CODEX_API_KEY.",
    ),
}


@dataclass(frozen=True)
class AgentAvailability:
    spec: AgentSpec
    available: bool
    detail: str
    launch: Launch | None = None


def resolve_launch(spec: AgentSpec) -> Launch | None:
    for launch in spec.launches:
        if shutil.which(launch.command):
            return launch
    return None


def agent_availability(spec: AgentSpec) -> AgentAvailability:
    for launch in spec.launches:
        executable = shutil.which(launch.command)
        if executable:
            return AgentAvailability(spec=spec, available=True, detail=executable, launch=launch)
    detail = spec.install_hint or f"`{spec.command}` was not found on PATH."
    return AgentAvailability(spec=spec, available=False, detail=detail, launch=None)


def custom_agents_path() -> Path:
    """Location of the user's custom ``agent_servers`` file."""

    explicit = os.getenv("DEVENV_AGENTS_PATH")
    if explicit:
        return Path(explicit).expanduser()
    home = os.getenv("DEVENV_HOME")
    if home:
        return Path(home).expanduser() / "agents.json"
    xdg = os.getenv("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "devenv" / "agents.json"


def load_custom_agents(path: Path | None = None) -> dict[str, AgentSpec]:
    """Load user-declared agents, merged over the built-in registry when used."""

    target = path or custom_agents_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring custom agents at %s: %s", target, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("Ignoring custom agents at %s: expected a JSON object.", target)
        return {}

    servers = raw.get("agent_servers")
    if not isinstance(servers, dict):
        return {}

    agents: dict[str, AgentSpec] = {}
    for raw_name, entry in servers.items():
        name = str(raw_name or "").strip().lower()
        if not name or not isinstance(entry, dict):
            continue
        spec = _spec_from_entry(name, entry)
        if spec is not None:
            agents[name] = spec
    return agents


def agent_registry() -> dict[str, AgentSpec]:
    """Built-in agents overridden/extended by user-declared agents."""

    registry = dict(AGENTS)
    registry.update(load_custom_agents())
    return registry


def resolve_agent(name: str) -> AgentSpec | None:
    cleaned = str(name or "").strip().lower()
    return agent_registry().get(cleaned)


def available_agents() -> list[AgentAvailability]:
    return [agent_availability(spec) for spec in agent_registry().values()]


def _spec_from_entry(name: str, entry: dict) -> AgentSpec | None:
    launches = _parse_launches(entry)
    if not launches:
        logger.warning("Ignoring custom agent `%s`: no command or launches.", name)
        return None
    title = str(entry.get("title") or name)
    description = str(entry.get("description") or "")
    install_hint = str(entry.get("installHint") or entry.get("install_hint") or "")
    auth_hint = str(entry.get("authHint") or entry.get("auth_hint") or "")
    env = {
        str(key): str(value)
        for key, value in (entry.get("env") or {}).items()
        if key is not None and value is not None
    }
    return AgentSpec(
        name=name,
        title=title,
        launches=launches,
        description=description,
        env=env,
        install_hint=install_hint,
        auth_hint=auth_hint,
    )


def _parse_launches(entry: dict) -> tuple[Launch, ...]:
    raw_launches = entry.get("launches")
    if isinstance(raw_launches, list):
        launches: list[Launch] = []
        for item in raw_launches:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command") or "").strip()
            if not command:
                continue
            args = item.get("args")
            launches.append(
                Launch(
                    command=command,
                    args=tuple(str(arg) for arg in args) if isinstance(args, list) else (),
                    label=str(item.get("label") or ""),
                )
            )
        if launches:
            return tuple(launches)

    command = str(entry.get("command") or "").strip()
    if not command:
        return ()
    args = entry.get("args")
    return (
        Launch(
            command=command,
            args=tuple(str(arg) for arg in args) if isinstance(args, list) else (),
        ),
    )


__all__ = [
    "AGENTS",
    "AgentAvailability",
    "AgentSpec",
    "Launch",
    "agent_availability",
    "agent_registry",
    "available_agents",
    "custom_agents_path",
    "load_custom_agents",
    "resolve_agent",
    "resolve_launch",
]
