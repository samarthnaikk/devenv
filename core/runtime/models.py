from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecutionStep:
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    output: str
    success: bool
    is_sandboxed_violation: bool


@dataclass(frozen=True)
class RuntimeTurnResult:
    final_response: str | None
    steps: list[ToolExecutionStep] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    workspace_path: str
    db_path: str = "memory.db"
    vector_dir: str = "vectors"
    max_consecutive_tools: int = 5
