from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
ClientSession = None
StdioServerParameters = None
stdio_client = None


@dataclass(frozen=True)
class MCPToolCallResult:
    success: bool
    output: str
    data: dict[str, Any]
    is_error: bool


class MCPToolClient:
    def __init__(
        self,
        *,
        workspace_path: str,
        db_path: str,
        vector_dir: str,
        log_level: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        command = sys.executable
        self._mcp = _load_mcp_dependencies()
        self.server_params = self._mcp["StdioServerParameters"](
            command=command,
            args=[
                "-m",
                "core.runtime.mcp_server",
                "--workspace",
                str(Path(workspace_path).expanduser().resolve()),
                "--db-path",
                db_path,
                "--vector-dir",
                vector_dir,
                "--log-level",
                log_level or "INFO",
            ],
            cwd=str(root),
            env=dict(os.environ),
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="devenv-mcp-client", daemon=True)
        self._thread.start()
        self._started = False
        self._closed = False
        self._session = None
        self._remote_tools: dict[str, dict[str, Any]] = {}
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._runner_future = None
        self._shutdown_event: asyncio.Event | None = None

    def list_tools(self) -> dict[str, dict[str, Any]]:
        self.start()
        return dict(self._remote_tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        self.start()
        return self._submit(self._call_tool(name, arguments))

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("MCP tool client has already been closed.")
        if self._started:
            return
        self._runner_future = asyncio.run_coroutine_threadsafe(self._session_runner(), self._loop)
        self._ready.wait(timeout=30)
        if self._start_error is not None:
            raise RuntimeError("Failed to start MCP tool client.") from self._start_error
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        if self._started:
            self._loop.call_soon_threadsafe(self._signal_shutdown)
            if self._runner_future is not None:
                self._runner_future.result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()
        self._closed = True

    async def _session_runner(self) -> None:
        try:
            async with self._mcp["stdio_client"](self.server_params) as (read_stream, write_stream):
                async with self._mcp["ClientSession"](read_stream, write_stream) as session:
                    self._session = session
                    await self._session.initialize()
                    tool_result = await self._session.list_tools()
                    self._remote_tools = {
                        tool.name: {
                            "description": tool.description,
                            "inputSchema": tool.inputSchema,
                        }
                        for tool in tool_result.tools
                    }
                    self._shutdown_event = asyncio.Event()
                    logger.info("Connected to MCP server: tool_count=%s", len(self._remote_tools))
                    self._ready.set()
                    await self._shutdown_event.wait()
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
            raise
        finally:
            self._session = None
            self._remote_tools = {}
            self._shutdown_event = None
            logger.info("Closed MCP server connection")

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        if self._session is None:
            raise RuntimeError("MCP session is not initialized.")
        result = await self._session.call_tool(name=name, arguments=arguments, read_timeout_seconds=timedelta(seconds=90))
        payload = _decode_tool_payload(result.content)
        return MCPToolCallResult(
            success=bool(payload.get("success", not result.isError)),
            output=str(payload.get("output", "")),
            data=dict(payload.get("data", {})) if isinstance(payload.get("data", {}), dict) else {},
            is_error=bool(result.isError),
        )

    def _submit(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _signal_shutdown(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()


def _decode_tool_payload(content: list[Any]) -> dict[str, Any]:
    if not content:
        return {"success": True, "output": "", "data": {}}

    first = content[0]
    text = getattr(first, "text", None)
    if isinstance(text, str):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"success": True, "output": text, "data": {}}
        if isinstance(payload, dict):
            return payload
    if hasattr(first, "model_dump"):
        dumped = first.model_dump()
        return {"success": True, "output": json.dumps(dumped, sort_keys=True), "data": {}}
    return {"success": True, "output": str(first), "data": {}}


def _load_mcp_dependencies() -> dict[str, Any]:
    try:
        from mcp import ClientSession as imported_client_session, StdioServerParameters as imported_server_parameters
        from mcp.client.stdio import stdio_client as imported_stdio_client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional 'mcp' dependency is not installed. Install project dependencies to enable the MCP runtime."
        ) from exc
    globals()["ClientSession"] = imported_client_session
    globals()["StdioServerParameters"] = imported_server_parameters
    globals()["stdio_client"] = imported_stdio_client
    return {
        "ClientSession": imported_client_session,
        "StdioServerParameters": imported_server_parameters,
        "stdio_client": imported_stdio_client,
    }
