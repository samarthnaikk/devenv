from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from mimetypes import guess_type
from pathlib import Path
from typing import Literal

from ._common import ensure_file
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


ReadFileFeature = Literal["content", "metadata", "extension", "all"]


@dataclass(frozen=True)
class FileMetadata:
    path: str
    name: str
    size_bytes: int
    line_count: int | None
    char_count: int | None


@dataclass(frozen=True)
class FileExtensionInfo:
    suffix: str
    extension: str
    mime_type: str | None
    kind: str


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file and optionally return content, metadata, and file type in one call."

    supported_features: tuple[str, ...] = ("content", "metadata", "extension", "all")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the target text file to inspect.",
                },
                "features": {
                    "type": "string",
                    "description": "Optional feature selection for the response payload.",
                    "enum": list(self.supported_features),
                },
            },
            "required": ["path"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        features = kwargs.get("features")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})

        try:
            selected_features = self._normalize_features(features)
            file_path = _resolve_file_path(path)
            logger.info("Reading file: path=%s features=%s", file_path, selected_features)

            result: dict[str, object] = {}
            if "content" in selected_features:
                result["content"] = read_file(path)
            if "metadata" in selected_features:
                result["metadata"] = asdict(build_metadata(file_path))
            if "extension" in selected_features:
                result["extension"] = asdict(build_extension_info(file_path))

            summary = ", ".join(selected_features)
            return ToolResult(
                success=True,
                output=f"read_file completed with features: {summary}",
                data=result,
            )
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError, ValueError) as exc:
            logger.error("read_file failed: path=%s error=%s", path, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _normalize_features(self, features: object) -> list[str]:
        if features is None:
            return ["content"]

        if isinstance(features, str):
            requested = [features]
        elif isinstance(features, Iterable):
            requested = [item for item in features if isinstance(item, str)]
        else:
            requested = []

        if not requested:
            return ["content"]

        if "all" in requested:
            return ["content", "metadata", "extension"]

        invalid = [feature for feature in requested if feature not in self.supported_features]
        if invalid:
            raise ValueError(f"Unsupported feature(s): {', '.join(invalid)}")

        ordered: list[str] = []
        for feature in ("content", "metadata", "extension"):
            if feature in requested and feature not in ordered:
                ordered.append(feature)

        return ordered


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


def _resolve_file_path(path: str) -> Path:
    return ensure_file(path)


def build_metadata(file_path: Path) -> FileMetadata:
    size_bytes = file_path.stat().st_size
    try:
        contents = file_path.read_text(encoding="utf-8")
        line_count = 0 if not contents else contents.count("\n") + (0 if contents.endswith("\n") else 1)
        char_count = len(contents)
    except UnicodeDecodeError:
        line_count = None
        char_count = None

    return FileMetadata(
        path=str(file_path),
        name=file_path.name,
        size_bytes=size_bytes,
        line_count=line_count,
        char_count=char_count,
    )


def build_extension_info(file_path: Path) -> FileExtensionInfo:
    suffix = file_path.suffix.lower()
    extension = suffix.lstrip(".")
    mime_type, _ = guess_type(file_path.name)

    kind_map = {
        "txt": "text",
        "md": "text",
        "csv": "data",
        "json": "data",
        "yaml": "data",
        "yml": "data",
        "toml": "data",
        "py": "code",
        "js": "code",
        "ts": "code",
        "html": "markup",
        "xml": "markup",
        "pdf": "document",
    }

    kind = kind_map.get(extension, "unknown")

    return FileExtensionInfo(
        suffix=suffix,
        extension=extension,
        mime_type=mime_type,
        kind=kind,
    )
