from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List the files and folders in a directory to inspect project structure safely."

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the target directory to inspect.",
                }
            },
            "required": ["path"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})

        try:
            directory = Path(path).expanduser().resolve()
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            if not directory.is_dir():
                raise NotADirectoryError(f"Expected a directory, got a file: {directory}")

            entries = []
            for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                entries.append({"name": child.name, "path": str(child), "is_dir": child.is_dir()})

            logger.info("Listing directory: path=%s entry_count=%s", directory, len(entries))
            return ToolResult(
                success=True,
                output=f"list_directory completed for {directory}",
                data={"path": str(directory), "entries": entries},
            )
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            logger.error("list_directory failed: path=%s error=%s", path, exc)
            return ToolResult(success=False, output=str(exc), data={})
