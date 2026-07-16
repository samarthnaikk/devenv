from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceEntry:
    path: str
    name: str
    is_dir: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "is_dir": self.is_dir,
        }


class WorkspaceBrowser:
    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser().resolve()

    def list_entries(self, relative_path: str = "", limit: int = 200) -> list[WorkspaceEntry]:
        target = self._resolve_relative(relative_path)
        if not target.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {target}")

        entries: list[WorkspaceEntry] = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[:limit]:
            entries.append(
                WorkspaceEntry(
                    path=self._to_relative(child),
                    name=child.name,
                    is_dir=child.is_dir(),
                )
            )
        return entries

    def read_text_file(self, relative_path: str) -> str:
        target = self._resolve_relative(relative_path)
        if target.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {target}")
        return target.read_text(encoding="utf-8")

    def read_file_preview(self, relative_path: str) -> dict[str, str]:
        target = self._resolve_relative(relative_path)
        if target.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {target}")

        mime_type, _encoding = mimetypes.guess_type(str(target))
        if mime_type and mime_type.startswith("image/"):
            payload = base64.b64encode(target.read_bytes()).decode("ascii")
            return {
                "kind": "image",
                "content_type": mime_type,
                "content": f"data:{mime_type};base64,{payload}",
            }
        if mime_type == "application/pdf" or target.suffix.lower() == ".pdf":
            payload = base64.b64encode(target.read_bytes()).decode("ascii")
            return {
                "kind": "pdf",
                "content_type": "application/pdf",
                "content": f"data:application/pdf;base64,{payload}",
            }

        try:
            return {
                "kind": "text",
                "content_type": mime_type or "text/plain",
                "content": target.read_text(encoding="utf-8"),
            }
        except UnicodeDecodeError:
            return {
                "kind": "binary",
                "content_type": mime_type or "application/octet-stream",
                "content": "",
            }

    def _resolve_relative(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"Path escapes workspace: {relative_path}")
        return candidate

    def _to_relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))
