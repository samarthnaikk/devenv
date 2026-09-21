"""Native ACP client for external AI agents (e.g. ``opencode acp``).

Devenv connects to the agent over the Agent Client Protocol (JSON-RPC over
stdio) and renders the agent's own stream of updates. Devenv does not compile
prompts, force output schemas, or execute the agent's tools; the agent owns its
native loop and only asks the client for permissions and file access.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    connect_to_agent,
)
from acp.core import ClientSideConnection
from acp.schema import (
    ClientCapabilities,
    CreateTerminalResponse,
    DeclineElicitationResponse,
    ElicitationMode,
    EnvVariable,
    FileSystemCapabilities,
    Implementation,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    SessionMode,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from .agents import AgentSpec, Launch, agent_availability

logger = logging.getLogger(__name__)

try:  # pragma: no cover - metadata is optional at runtime
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        DEVENV_VERSION = _pkg_version("devenv-ai")
    except PackageNotFoundError:
        DEVENV_VERSION = "0.0.0"
except Exception:  # pragma: no cover
    DEVENV_VERSION = "0.0.0"

_STDERR_BUFFER_LINES = 40


class ACPAgentError(RuntimeError):
    """Raised when an ACP agent cannot be started or a turn fails."""


PermissionHandler = Callable[
    [ToolCallUpdate, list[PermissionOption]],
    Awaitable[RequestPermissionResponse],
]


@dataclass(frozen=True)
class AgentSessionInfo:
    agent_name: str
    agent_title: str
    agent_version: str
    session_id: str
    modes: list[SessionMode] = field(default_factory=list)
    auth_methods: list[Any] = field(default_factory=list)


class ACPAgentSession:
    """Owns the ``opencode acp`` subprocess and its ACP connection."""

    def __init__(
        self,
        spec: AgentSpec,
        workspace_path: str,
        *,
        permission_handler: PermissionHandler | None = None,
    ) -> None:
        self.spec = spec
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.permission_handler = permission_handler
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        self.info: AgentSessionInfo | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._connection: ClientSideConnection | None = None
        self._launch: Launch | None = None
        self._client = _DevenvACPClient(self)
        self._stderr_lines: list[str] = []
        self._stderr_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> AgentSessionInfo:
        if self._connection is not None:
            assert self.info is not None
            return self.info
        self._closed = False

        availability = agent_availability(self.spec)
        if not availability.available or availability.launch is None:
            raise ACPAgentError(f"Cannot start `{self.spec.name}`: {availability.detail}")
        self._launch = availability.launch

        env = os.environ.copy()
        for key, value in self.spec.env.items():
            env.setdefault(key, value)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._launch.launch_command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_path,
                env=env,
            )
        except OSError as exc:
            raise ACPAgentError(f"Failed to start `{self._launch.command}`: {exc}") from exc

        if self._process.stdin is None or self._process.stdout is None:
            await self.close()
            raise ACPAgentError(f"`{self._launch.command}` did not expose stdio pipes.")

        if self._process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr())

        self._connection = connect_to_agent(
            self._client,
            self._process.stdin,
            self._process.stdout,
        )

        try:
            initialize = await self._connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(
                        read_text_file=True,
                        write_text_file=True,
                    )
                ),
                client_info=Implementation(
                    name="devenv",
                    title="Devenv",
                    version=DEVENV_VERSION,
                ),
            )
            auth_methods = list(getattr(initialize, "auth_methods", []) or [])
            await self._authenticate_if_possible(auth_methods)
            session = await self._connection.new_session(
                cwd=self.workspace_path,
                mcp_servers=[],
            )
        except Exception as exc:
            detail = self._stderr_tail()
            await self.close()
            message = f"ACP handshake failed: {exc}"
            if detail:
                message = f"{message}\n{detail}"
            if self.spec.auth_hint:
                message = f"{message}\n{self.spec.auth_hint}"
            raise ACPAgentError(message) from exc

        agent_info = initialize.agent_info
        self.info = AgentSessionInfo(
            agent_name=(getattr(agent_info, "name", None) or self.spec.name),
            agent_title=(getattr(agent_info, "title", None) or self.spec.title),
            agent_version=(getattr(agent_info, "version", None) or ""),
            session_id=session.session_id,
            modes=list(getattr(session, "modes", []) or []),
            auth_methods=auth_methods,
        )
        logger.info(
            "Connected ACP agent %s (%s) session=%s",
            self.info.agent_title,
            self.info.agent_version or "unknown",
            self.info.session_id,
        )
        return self.info

    async def send_prompt(self, text: str) -> str:
        if self._connection is None or self.info is None:
            raise ACPAgentError("Agent session is not started.")
        from acp.schema import TextContentBlock

        try:
            response = await self._connection.prompt(
                session_id=self.info.session_id,
                prompt=[TextContentBlock(type="text", text=text)],
            )
        except Exception as exc:
            raise ACPAgentError(f"Prompt failed: {exc}") from exc
        return str(getattr(response, "stop_reason", "") or "")

    async def _authenticate_if_possible(self, auth_methods: list[Any]) -> None:
        """Best-effort auth using non-blocking (env-var) methods.

        Terminal/browser methods are left to the agent so we never block the
        handshake on a flow this client cannot drive.
        """

        if not auth_methods or not _auto_auth_enabled():
            return
        method = _select_env_auth_method(auth_methods)
        if method is None or self._connection is None:
            return
        method_id = str(getattr(method, "id", "") or "")
        if not method_id:
            return
        try:
            await self._connection.authenticate(method_id=method_id)
            logger.info("Authenticated ACP agent with method %s", method_id)
        except Exception as exc:  # pragma: no cover - auth is best-effort
            logger.warning("ACP authenticate with %s failed: %s", method_id, exc)

    async def cancel(self) -> None:
        if self._connection is None or self.info is None:
            return
        try:
            await self._connection.cancel(session_id=self.info.session_id)
        except Exception as exc:  # pragma: no cover - cancellation is best-effort
            logger.warning("ACP cancel failed: %s", exc)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return

            if self._connection is not None:
                with contextlib.suppress(Exception):
                    await self._connection.close()
                self._connection = None

            if self._stderr_task is not None:
                self._stderr_task.cancel()
                self._stderr_task = None

            if self._process is not None:
                if self._process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=3)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        with contextlib.suppress(ProcessLookupError):
                            self._process.kill()
                self._process = None

            self._closed = True

    # ------------------------------------------------------------------ internals
    async def _emit(self, update: Any) -> None:
        await self.events.put(update)

    async def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            async for raw in self._process.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > _STDERR_BUFFER_LINES:
                    del self._stderr_lines[0]
                logger.debug("acp[%s] %s", self.spec.name, line)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # pragma: no cover - stderr draining is best-effort
            return

    def _stderr_tail(self) -> str:
        if not self._stderr_lines:
            return ""
        return "\n".join(self._stderr_lines[-5:])


class _DevenvACPClient(Client):
    """ACP client callbacks forwarded into the UI via the owning session."""

    def __init__(self, session: ACPAgentSession) -> None:
        self._session = session

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        await self._session._emit(update)

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        handler = self._session.permission_handler
        if handler is None:
            from acp.schema import DeniedOutcome

            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return await handler(tool_call, options)

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise RequestError.resource_not_found(path) from exc
        if line is not None or limit is not None:
            lines = text.splitlines(keepends=True)
            start = max((line or 1) - 1, 0)
            end = start + limit if limit is not None else None
            text = "".join(lines[start:end])
        return ReadTextFileResponse(content=text)

    async def write_text_file(
        self,
        session_id: str,
        path: str,
        content: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise RequestError.internal_error({"path": path, "reason": str(exc)}) from exc
        return WriteTextFileResponse()

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(
        self, message: str, mode: ElicitationMode, **kwargs: Any
    ) -> Any:
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


def _select_env_auth_method(auth_methods: list[Any]) -> Any | None:
    for method in auth_methods:
        if _auth_method_type(method) == "env_var":
            return method
    return None


def _auth_method_type(method: Any) -> str:
    if isinstance(method, dict):
        return str(method.get("type") or "")
    return str(getattr(method, "type", "") or "")


def _auto_auth_enabled() -> bool:
    raw = os.getenv("DEVENV_ACP_AUTO_AUTH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


__all__ = [
    "ACPAgentError",
    "ACPAgentSession",
    "AgentSessionInfo",
    "PermissionHandler",
]
