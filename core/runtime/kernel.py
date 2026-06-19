from __future__ import annotations

from pathlib import Path
from typing import Any

from core.ai import AICore
from core.memory import MemoryEngine
from core.tools.base import BaseTool

from .models import RuntimeTurnResult
from .sandbox import PathSandbox


class DevenvKernel:
    def __init__(
        self,
        workspace_path: str,
        db_path: str = "memory.db",
        vector_dir: str = "vectors",
        *,
        memory: MemoryEngine | Any | None = None,
        ai: AICore | Any | None = None,
    ):
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.sandbox = PathSandbox(root_path=self.workspace_path)
        self.memory = memory or MemoryEngine(db_path=db_path, vector_dir=vector_dir)
        self.ai = ai or AICore()
        self.tools: dict[str, BaseTool] = {}
        self.ephemeral_history: list[dict[str, Any]] = []

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def execute_turn(self, user_prompt: str, max_consecutive_tools: int = 5) -> RuntimeTurnResult:
        return RuntimeTurnResult(final_response=None)
