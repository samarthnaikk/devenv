from __future__ import annotations

import logging

from ._common import ensure_file
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


DEFAULT_SLICE_SIZE = 20


class PeekLinesTool(BaseTool):
    name = "peek_lines"
    description = "Read a narrow line window from a text file using range, head, or tail modes."

    supported_modes: tuple[str, ...] = ("range", "head", "tail")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the target text file.",
                },
                "mode": {
                    "type": "string",
                    "description": "Line window mode.",
                    "enum": list(self.supported_modes),
                },
                "start": {
                    "type": "integer",
                    "description": "Starting line number for range mode.",
                    "minimum": 1,
                },
                "end": {
                    "type": "integer",
                    "description": "Ending line number for range mode, or row count for head and tail modes.",
                    "minimum": 1,
                },
            },
            "required": ["path", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        mode = kwargs.get("mode")
        start = kwargs.get("start")
        end = kwargs.get("end")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            file_path = ensure_file(path)
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            slice_info = self._select_lines(lines, mode=mode, start=start, end=end)

            logger.info(
                "Peeked lines: path=%s mode=%s start=%s end=%s returned=%s",
                file_path,
                mode,
                slice_info["line_start"],
                slice_info["line_end"],
                len(slice_info["lines"]),
            )
            return ToolResult(
                success=True,
                output=(
                    f"peek_lines returned lines {slice_info['line_start']}-{slice_info['line_end']} "
                    f"from {file_path.name} using {mode} mode"
                ),
                data={
                    "path": str(file_path),
                    "mode": mode,
                    "line_start": slice_info["line_start"],
                    "line_end": slice_info["line_end"],
                    "line_count": len(lines),
                    "content": "\n".join(slice_info["lines"]),
                    "lines": slice_info["lines"],
                },
            )
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError, ValueError) as exc:
            logger.error("peek_lines failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _select_lines(
        self,
        lines: list[str],
        *,
        mode: str,
        start: object,
        end: object,
    ) -> dict[str, object]:
        if mode == "range":
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("range mode requires integer start and end arguments")
            if start < 1 or end < start:
                raise ValueError("range mode requires 1 <= start <= end")
            selected = lines[start - 1 : end]
            return {"line_start": start, "line_end": min(end, len(lines)), "lines": selected}

        count = end if isinstance(end, int) else DEFAULT_SLICE_SIZE
        if count < 1:
            raise ValueError("head and tail modes require end to be a positive integer when provided")

        if mode == "head":
            selected = lines[:count]
            line_end = min(len(lines), count)
            return {"line_start": 1 if selected else 0, "line_end": line_end, "lines": selected}

        selected = lines[-count:]
        line_start = max(len(lines) - len(selected) + 1, 1) if selected else 0
        return {"line_start": line_start, "line_end": len(lines), "lines": selected}
