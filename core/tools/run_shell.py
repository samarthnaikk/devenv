from __future__ import annotations

import logging
import subprocess

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class RunShellTool(BaseTool):
    name = "run_shell"
    description = "Run non-interactive shell commands directly or in the background."

    supported_modes: tuple[str, ...] = ("raw", "background")

    def __init__(self) -> None:
        self._background_processes: dict[int, subprocess.Popen[str]] = {}

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "mode": {
                    "type": "string",
                    "description": "Execution strategy.",
                    "enum": list(self.supported_modes),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in seconds for raw mode.",
                    "default": 30,
                    "minimum": 1,
                },
            },
            "required": ["command", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command")
        mode = kwargs.get("mode")
        timeout = kwargs.get("timeout", 30)

        if not isinstance(command, str) or not command.strip():
            return ToolResult(success=False, output="Missing required argument: command", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if timeout is None:
            timeout = 30
        if not isinstance(timeout, int) or timeout < 1:
            return ToolResult(success=False, output="timeout must be a positive integer", data={})

        try:
            if mode == "background":
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                )
                self._background_processes[process.pid] = process
                logger.info("Spawned background shell command: pid=%s command=%s", process.pid, command)
                return ToolResult(
                    success=True,
                    output=f"run_shell started background command with pid {process.pid}",
                    data={"command": command, "mode": mode, "pid": process.pid},
                )

            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            logger.info("Ran shell command: returncode=%s command=%s", completed.returncode, command)
            return ToolResult(
                success=completed.returncode == 0,
                output=stdout or stderr or f"Command exited with status {completed.returncode}",
                data={
                    "command": command,
                    "mode": mode,
                    "returncode": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("run_shell failed: mode=%s command=%s error=%s", mode, command, exc)
            return ToolResult(success=False, output=str(exc), data={})
