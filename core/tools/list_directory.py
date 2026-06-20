from __future__ import annotations

import logging

from .base import BaseTool, ToolResult
from ._common import NOISE_DIRECTORIES, ensure_directory, iter_directory, relative_display

logger = logging.getLogger(__name__)


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List and summarize directory structure in flat, recursive, or topology modes."

    supported_modes: tuple[str, ...] = ("flat", "recursive", "topology")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the target directory to inspect.",
                },
                "mode": {
                    "type": "string",
                    "description": "Directory inspection mode.",
                    "enum": list(self.supported_modes),
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Optional recursion depth for recursive and topology modes.",
                    "default": 3,
                    "minimum": 1,
                },
            },
            "required": ["path", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        mode = kwargs.get("mode")
        max_depth = kwargs.get("max_depth", 3)

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if max_depth is None:
            max_depth = 3
        if not isinstance(max_depth, int) or max_depth < 1:
            return ToolResult(success=False, output="max_depth must be an integer greater than 0", data={})

        try:
            directory = ensure_directory(path)
            if mode == "flat":
                entries = self._flat_entries(directory)
                payload = {"path": str(directory), "mode": mode, "entries": entries}
            elif mode == "recursive":
                entries = self._recursive_entries(directory, max_depth=max_depth)
                payload = {"path": str(directory), "mode": mode, "max_depth": max_depth, "entries": entries}
            else:
                topology = self._topology(directory, max_depth=max_depth)
                payload = {"path": str(directory), "mode": mode, "max_depth": max_depth, "topology": topology}

            logger.info("Listing directory: path=%s mode=%s", directory, mode)
            return ToolResult(
                success=True,
                output=f"list_directory completed for {directory} in {mode} mode",
                data=payload,
            )
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            logger.error("list_directory failed: path=%s error=%s", path, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _flat_entries(self, directory) -> list[dict[str, object]]:
        entries = []
        for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "relative_path": child.name,
                    "is_dir": child.is_dir(),
                }
            )
        return entries

    def _recursive_entries(self, directory, *, max_depth: int) -> list[dict[str, object]]:
        entries = []
        for child, depth in iter_directory(directory, max_depth=max_depth):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "relative_path": relative_display(child, directory),
                    "is_dir": child.is_dir(),
                    "depth": depth,
                }
            )
        return entries

    def _topology(self, directory, *, max_depth: int) -> list[dict[str, object]]:
        topology = []
        for child, depth in iter_directory(
            directory,
            max_depth=max_depth,
            include_files=False,
            noise_directories=NOISE_DIRECTORIES,
        ):
            topology.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "relative_path": relative_display(child, directory),
                    "depth": depth,
                    "is_dir": True,
                }
            )
        return topology
