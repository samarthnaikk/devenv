from __future__ import annotations

from pathlib import Path

from .base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from disk and return its contents."

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path")

        try:
            contents = read_file(path)
            return ToolResult(success=True, output=contents)
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError) as exc:
            return ToolResult(success=False, output=str(exc))


def read_file(path: str) -> str:
    """
    Read a text file from disk and return its contents.

    Args:
        path: File system path to the file to read.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path points to a directory.
        UnicodeDecodeError: If the file is not valid UTF-8 text.
        OSError: For other I/O failures.
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {file_path}")

    return file_path.read_text(encoding="utf-8")
