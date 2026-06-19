from __future__ import annotations

from pathlib import Path
from typing import Any


class PathSandbox:
    def __init__(self, root_path: str):
        self.allowed_root = Path(root_path).expanduser().resolve()

    def resolve_within_root(self, target_path: str) -> Path:
        candidate = Path(target_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.allowed_root / candidate).resolve()

    def is_safe(self, target_path: str) -> bool:
        try:
            resolved_target = self.resolve_within_root(target_path)
        except Exception:
            return False
        return self.allowed_root in resolved_target.parents or resolved_target == self.allowed_root

    def violation_message(self, target_path: str) -> str:
        return (
            f"Sandbox violation: path '{target_path}' is outside the allowed workspace "
            f"'{self.allowed_root}'."
        )

    def find_unsafe_argument(self, arguments: dict[str, Any]) -> tuple[str, str] | None:
        for key, value in arguments.items():
            if key in {"path", "file_path", "target_path"} and isinstance(value, str):
                if not self.is_safe(value):
                    return key, value
        return None

    def normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        for key, value in arguments.items():
            if key in {"path", "file_path", "target_path"} and isinstance(value, str) and self.is_safe(value):
                normalized[key] = str(self.resolve_within_root(value))
        return normalized
