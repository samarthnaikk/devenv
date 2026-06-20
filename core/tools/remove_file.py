from __future__ import annotations

import logging

from ._common import ensure_file
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class RemoveFileTool(BaseTool):
    name = "remove_file"
    description = "Truncate or permanently delete files inside the workspace."

    supported_modes: tuple[str, ...] = ("soft", "permanent")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Target file path.",
                },
                "mode": {
                    "type": "string",
                    "description": "Deletion strategy.",
                    "enum": list(self.supported_modes),
                },
            },
            "required": ["path", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        mode = kwargs.get("mode")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            file_path = ensure_file(path)
            size_before = file_path.stat().st_size
            if mode == "soft":
                file_path.write_text("", encoding="utf-8")
                exists = True
            else:
                file_path.unlink()
                exists = False

            logger.info("Removed file: path=%s mode=%s", file_path, mode)
            return ToolResult(
                success=True,
                output=f"remove_file completed for {file_path.name} using {mode} mode",
                data={
                    "path": str(file_path),
                    "mode": mode,
                    "size_before": size_before,
                    "exists_after": exists,
                },
            )
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
            logger.error("remove_file failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})
