from __future__ import annotations

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse


@dataclass(frozen=True)
class MCPHTTPServerConfig:
    base_url: str = "http://127.0.0.1:8765/mcp"
    auth_token: str | None = None
    startup_timeout_seconds: float = 10.0
    log_level: str = "INFO"

    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(frozen=True)
class MCPHTTPServerRuntimeStatus:
    reachable: bool
    base_url: str
    detail: str = ""
    started_by_manager: bool = False
    auth_enabled: bool = False
    pid: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "base_url": self.base_url,
            "detail": self.detail,
            "started_by_manager": self.started_by_manager,
            "auth_enabled": self.auth_enabled,
            "pid": self.pid,
        }


class MCPHTTPServerManager:
    def __init__(
        self,
        *,
        workspace_path: str,
        db_path: str,
        vector_dir: str,
        config: MCPHTTPServerConfig | None = None,
    ) -> None:
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.db_path = db_path
        self.vector_dir = vector_dir
        self.config = config or default_mcp_http_server_config()
        self._process: subprocess.Popen[str] | None = None
        self._started_by_manager = False
        self._registered_atexit = False

    def inspect(self) -> MCPHTTPServerRuntimeStatus:
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._started_by_manager = False
        reachable = _tcp_endpoint_reachable(self.config.normalized_base_url())
        pid = self._process.pid if self._process is not None and self._process.poll() is None else None
        detail = "MCP HTTP endpoint reachable." if reachable else "MCP HTTP endpoint is unavailable."
        return MCPHTTPServerRuntimeStatus(
            reachable=reachable,
            base_url=self.config.normalized_base_url(),
            detail=detail,
            started_by_manager=self._started_by_manager,
            auth_enabled=bool(self.config.auth_token),
            pid=pid,
        )

    def ensure_server(self) -> MCPHTTPServerRuntimeStatus:
        status = self.inspect()
        if status.reachable:
            return status

        command = self._build_command()
        env = dict(os.environ)
        if self.config.auth_token:
            env.setdefault("DEVENV_MCP_AUTH_TOKEN", self.config.auth_token)

        self._process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._started_by_manager = True
        self._register_atexit()

        deadline = time.monotonic() + max(self.config.startup_timeout_seconds, 0.1)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            status = self.inspect()
            if status.reachable:
                return status
            time.sleep(0.1)

        raise RuntimeError(
            f"Timed out waiting for the Devenv MCP HTTP server at {self.config.normalized_base_url()}."
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        self._started_by_manager = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _build_command(self) -> list[str]:
        parsed = parse.urlparse(self.config.normalized_base_url())
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8765
        path = parsed.path or "/mcp"
        return [
            sys.executable,
            "-m",
            "core.runtime.mcp_server",
            "--workspace",
            self.workspace_path,
            "--db-path",
            self.db_path,
            "--vector-dir",
            self.vector_dir,
            "--transport",
            "streamable-http",
            "--host",
            host,
            "--port",
            str(port),
            "--path",
            path,
            "--log-level",
            self.config.log_level,
        ]

    def _register_atexit(self) -> None:
        if self._registered_atexit:
            return
        atexit.register(self.close)
        self._registered_atexit = True


def default_mcp_http_server_config() -> MCPHTTPServerConfig:
    base_url = os.getenv("DEVENV_MCP_SERVER_URL") or "http://127.0.0.1:8765/mcp"
    auth_token = os.getenv("DEVENV_MCP_AUTH_TOKEN") or None
    startup_timeout_seconds = _float_env("DEVENV_MCP_SERVER_STARTUP_TIMEOUT", 10.0)
    log_level = os.getenv("DEVENV_MCP_SERVER_LOG_LEVEL") or "INFO"
    return MCPHTTPServerConfig(
        base_url=base_url,
        auth_token=auth_token,
        startup_timeout_seconds=startup_timeout_seconds,
        log_level=log_level,
    )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _tcp_endpoint_reachable(base_url: str) -> bool:
    parsed = parse.urlparse(base_url)
    hostname = parsed.hostname
    if not hostname:
        return False
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((hostname, port), timeout=1.0):
            return True
    except OSError:
        return False
