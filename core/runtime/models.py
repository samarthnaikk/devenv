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

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "output": self.output,
            "success": self.success,
            "is_sandboxed_violation": self.is_sandboxed_violation,
        }


@dataclass(frozen=True)
class RuntimeTurnResult:
    final_response: str | None
    steps: list[ToolExecutionStep] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)
    ai_logs: list[str] = field(default_factory=list)
    system_logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_response": self.final_response,
            "steps": [step.to_dict() for step in self.steps],
            "total_usage": dict(self.total_usage),
            "ai_logs": list(self.ai_logs),
            "system_logs": list(self.system_logs),
        }


@dataclass(frozen=True)
class RunConfig:
    workspace_path: str
    db_path: str = "memory.db"
    vector_dir: str = "vectors"
    max_consecutive_tools: int = 5
