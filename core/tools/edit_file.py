from __future__ import annotations

import logging

from ._common import ensure_file
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class EditFileTool(BaseTool):
    name = "edit_file"
    description = "Patch targeted file regions and undo the last patch for a path."

    supported_modes: tuple[str, ...] = ("patch", "undo")

    def __init__(self) -> None:
        self._history: dict[str, list[str]] = {}

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
                    "description": "Edit mode.",
                    "enum": list(self.supported_modes),
                },
                "search_block": {
                    "type": "string",
                    "description": "Required in patch mode. Exact text block to replace.",
                },
                "replace_block": {
                    "type": "string",
                    "description": "Required in patch mode. Replacement text block.",
                },
            },
            "required": ["path", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        mode = kwargs.get("mode")
        search_block = kwargs.get("search_block")
        replace_block = kwargs.get("replace_block")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            file_path = ensure_file(path)
            if mode == "patch":
                if not isinstance(search_block, str) or not isinstance(replace_block, str):
                    raise ValueError("patch mode requires string search_block and replace_block arguments")
                if not search_block:
                    raise ValueError("patch mode requires a non-empty search_block")
                before = file_path.read_text(encoding="utf-8")
                if search_block not in before:
                    raise ValueError(f"search_block not found in file: {file_path}")
                self._history.setdefault(str(file_path), []).append(before)
                after = before.replace(search_block, replace_block, 1)
                file_path.write_text(after, encoding="utf-8")
                output = f"edit_file patched {file_path.name} successfully"
            else:
                history = self._history.get(str(file_path), [])
                if not history:
                    raise ValueError(f"No undo history available for file: {file_path}")
                previous = history.pop()
                file_path.write_text(previous, encoding="utf-8")
                output = f"edit_file restored the previous contents of {file_path.name}"

            logger.info("Edited file: path=%s mode=%s", file_path, mode)
            return ToolResult(
                success=True,
                output=output,
                data={
                    "path": str(file_path),
                    "mode": mode,
                    "size_bytes": file_path.stat().st_size,
                    "undo_depth": len(self._history.get(str(file_path), [])),
                },
            )
        except (FileNotFoundError, IsADirectoryError, OSError, PermissionError, ValueError) as exc:
            logger.error("edit_file failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})
