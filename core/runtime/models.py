from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PlanningMode(Enum):
    AUTO = "auto"
    FORCE_PLAN = "force_plan"
    FORCE_DIRECT = "force_direct"


class AgentState(Enum):
    PLANNING = auto()
    EXECUTING = auto()
    VERIFYING = auto()


@dataclass(frozen=True)
class CheckpointTask:
    task_id: int
    description: str
    is_completed: bool = False
    execution_trace_log: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "is_completed": self.is_completed,
            "execution_trace_log": self.execution_trace_log,
        }


@dataclass(frozen=True)
class ExecutionBlueprint:
    raw_plan_markdown: str
    tasks: list[CheckpointTask] = field(default_factory=list)
    active_task_pointer: int = 0
    verification_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_plan_markdown": self.raw_plan_markdown,
            "tasks": [task.to_dict() for task in self.tasks],
            "active_task_pointer": self.active_task_pointer,
            "verification_passed": self.verification_passed,
        }


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
    state: str = AgentState.PLANNING.name
    blueprint: ExecutionBlueprint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_response": self.final_response,
            "steps": [step.to_dict() for step in self.steps],
            "total_usage": dict(self.total_usage),
            "ai_logs": list(self.ai_logs),
            "system_logs": list(self.system_logs),
            "state": self.state,
            "blueprint": self.blueprint.to_dict() if self.blueprint else None,
        }


@dataclass(frozen=True)
class RunConfig:
    workspace_path: str
    db_path: str = "memory.db"
    vector_dir: str = "vectors"
    max_consecutive_tools: int = 5
