from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ._common import ensure_existing_path
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class AuditChangesTool(BaseTool):
    name = "audit_changes"
    description = "Summarize local git status or diff information for a workspace path."

    supported_modes: tuple[str, ...] = ("diff", "status")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Whether to summarize git diff or status output.",
                    "enum": list(self.supported_modes),
                },
                "path": {
                    "type": "string",
                    "description": "Optional repository path or file path inside the repository.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        path = kwargs.get("path", ".")

        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="path must be a non-empty string when provided", data={})

        try:
            target = ensure_existing_path(path)
            repo_root = self._repo_root(target)
            if mode == "status":
                return self._status(repo_root)
            return self._diff(repo_root)
        except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.error("audit_changes failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _repo_root(self, target: Path) -> Path:
        start = target if target.is_dir() else target.parent
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(f"Not a git repository: {start}")
        return Path(completed.stdout.strip())

    def _status(self, repo_root: Path) -> ToolResult:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        entries = [{"code": line[:2].strip(), "path": line[3:]} for line in lines]
        logger.info("Audited git status: repo=%s entries=%s", repo_root, len(entries))
        return ToolResult(
            success=completed.returncode == 0,
            output=f"audit_changes status found {len(entries)} changed path(s)",
            data={"mode": "status", "repo_root": str(repo_root), "entries": entries, "count": len(entries)},
        )

    def _diff(self, repo_root: Path) -> ToolResult:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--stat"],
            capture_output=True,
            text=True,
            check=False,
        )
        stat = completed.stdout.strip()
        logger.info("Audited git diff: repo=%s has_diff=%s", repo_root, bool(stat))
        return ToolResult(
            success=completed.returncode == 0,
            output="audit_changes diff completed",
            data={"mode": "diff", "repo_root": str(repo_root), "stat": stat},
        )
