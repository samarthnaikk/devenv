from __future__ import annotations

import logging

from ._common import ensure_directory, file_matches_pattern, iter_directory, relative_display
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class LocateFilesTool(BaseTool):
    name = "locate_files"
    description = "Find files in the workspace by exact filename or glob pattern."

    supported_modes: tuple[str, ...] = ("glob", "exact")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Filename or glob pattern to match while scanning the workspace.",
                },
                "mode": {
                    "type": "string",
                    "description": "Pattern matching mode.",
                    "enum": list(self.supported_modes),
                },
                "path": {
                    "type": "string",
                    "description": "Optional directory root to scan. Defaults to the current workspace path.",
                },
            },
            "required": ["pattern", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        pattern = kwargs.get("pattern")
        mode = kwargs.get("mode")
        path = kwargs.get("path", ".")

        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(success=False, output="Missing required argument: pattern", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="path must be a non-empty string when provided", data={})

        try:
            root = ensure_directory(path)
            matches: list[dict[str, object]] = []
            for entry, depth in iter_directory(root, max_depth=32):
                if entry.is_dir():
                    continue
                if file_matches_pattern(entry, pattern, mode=mode):
                    matches.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "relative_path": relative_display(entry, root),
                            "depth": depth,
                        }
                    )

            logger.info("Located files: root=%s mode=%s pattern=%s match_count=%s", root, mode, pattern, len(matches))
            return ToolResult(
                success=True,
                output=f"locate_files found {len(matches)} match(es) for '{pattern}' using {mode} mode",
                data={
                    "path": str(root),
                    "pattern": pattern,
                    "mode": mode,
                    "matches": matches,
                    "count": len(matches),
                },
            )
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
            logger.error("locate_files failed: path=%s pattern=%s error=%s", path, pattern, exc)
            return ToolResult(success=False, output=str(exc), data={})
