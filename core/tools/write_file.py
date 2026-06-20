from __future__ import annotations

import logging
from pathlib import Path

from ._common import resolve_path
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create, overwrite, or append text content to files inside the workspace."

    supported_modes: tuple[str, ...] = ("fresh", "overwrite", "append")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Target file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                },
                "mode": {
                    "type": "string",
                    "description": "Write strategy.",
                    "enum": list(self.supported_modes),
                },
            },
            "required": ["path", "content", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")
        mode = kwargs.get("mode")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(content, str):
            return ToolResult(success=False, output="Missing required argument: content", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            file_path = resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if mode == "fresh":
                if file_path.exists():
                    raise FileExistsError(f"File already exists: {file_path}")
                file_path.write_text(content, encoding="utf-8")
            elif mode == "overwrite":
                file_path.write_text(content, encoding="utf-8")
            else:
                with file_path.open("a", encoding="utf-8") as handle:
                    handle.write(content)

            logger.info("Wrote file: path=%s mode=%s bytes=%s", file_path, mode, len(content.encode('utf-8')))
            return ToolResult(
                success=True,
                output=f"write_file completed for {file_path.name} using {mode} mode",
                data={
                    "path": str(file_path),
                    "mode": mode,
                    "bytes_written": len(content.encode("utf-8")),
                    "size_bytes": file_path.stat().st_size,
                },
            )
        except (FileExistsError, PermissionError, OSError) as exc:
            logger.error("write_file failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})
