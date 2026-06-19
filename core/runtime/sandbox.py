from __future__ import annotations

from pathlib import Path


class PathSandbox:
    def __init__(self, root_path: str):
        self.allowed_root = Path(root_path).expanduser().resolve()

    def is_safe(self, target_path: str) -> bool:
        try:
            resolved_target = Path(target_path).expanduser().resolve()
        except Exception:
            return False
        return self.allowed_root in resolved_target.parents or resolved_target == self.allowed_root
