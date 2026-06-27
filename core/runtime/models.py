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


class ProcessStage(Enum):
    CHECKPOINT_CREATION = "checkpoint_creation"
    CONTEXT_MEMORY = "context_memory"
    BRAIN = "brain"
    METADATA = "metadata"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class VerificationResult:
    checkpoint_id: int
    mode: str
    success: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "mode": self.mode,
            "success": self.success,
            "details": self.details,
        }


@dataclass(frozen=True)
class StageTrace:
    stage: str
    checkpoint_id: int | None = None
    success: bool = True
    summary: str = ""
    logs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "checkpoint_id": self.checkpoint_id,
            "success": self.success,
            "summary": self.summary,
            "logs": list(self.logs),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class CheckpointTask:
    task_id: int
    description: str
    objective: str | None = None
    target_path_hint: str | None = None
    expected_artifact: str = "chat"
    verification_mode: str = "chat"
    repair_origin_checkpoint_id: int | None = None
    status_reason: str | None = None
    output_destination: str | None = None
    child_checkpoint_ids: tuple[int, ...] = ()
    is_completed: bool = False
    execution_trace_log: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "objective": self.objective or self.description,
            "target_path_hint": self.target_path_hint,
            "expected_artifact": self.expected_artifact,
            "verification_mode": self.verification_mode,
            "repair_origin_checkpoint_id": self.repair_origin_checkpoint_id,
            "status_reason": self.status_reason,
            "output_destination": self.output_destination,
            "child_checkpoint_ids": list(self.child_checkpoint_ids),
            "is_completed": self.is_completed,
            "execution_trace_log": self.execution_trace_log,
        }


@dataclass(frozen=True)
class ExecutionBlueprint:
    raw_plan_markdown: str
    original_objective: str | None = None
    tasks: list[CheckpointTask] = field(default_factory=list)
    active_task_pointer: int = 0
    verification_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_plan_markdown": self.raw_plan_markdown,
            "original_objective": self.original_objective,
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
    stage_traces: list[StageTrace] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    state: str = AgentState.PLANNING.name
    blueprint: ExecutionBlueprint | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_response": self.final_response,
            "steps": [step.to_dict() for step in self.steps],
            "total_usage": dict(self.total_usage),
            "ai_logs": list(self.ai_logs),
            "system_logs": list(self.system_logs),
            "stage_traces": [trace.to_dict() for trace in self.stage_traces],
            "verification_results": [result.to_dict() for result in self.verification_results],
            "metadata": dict(self.metadata),
            "state": self.state,
            "blueprint": self.blueprint.to_dict() if self.blueprint else None,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RunConfig:
    workspace_path: str
    db_path: str = "memory.db"
    vector_dir: str = "vectors"
    max_consecutive_tools: int = 5
    external_session_configs: tuple["ExternalSessionProviderConfig", ...] = ()


@dataclass(frozen=True)
class ExternalSessionProviderConfig:
    provider: str
    root_path: str
    enabled: bool = True
    session_glob: str = "sessions/**/*.jsonl"
    index_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "root_path": self.root_path,
            "enabled": self.enabled,
            "session_glob": self.session_glob,
            "index_path": self.index_path,
        }


@dataclass(frozen=True)
class ExternalSourceHealth:
    provider: str
    enabled: bool
    available: bool
    root_path: str
    summary: str
    session_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "enabled": self.enabled,
            "available": self.available,
            "root_path": self.root_path,
            "summary": self.summary,
            "session_count": self.session_count,
        }


@dataclass(frozen=True)
class ExternalSessionSummary:
    provider: str
    session_id: str
    title: str
    updated_at: str
    workspace_path: str | None = None
    source_path: str | None = None
    message_count: int = 0
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "session_id": self.session_id,
            "title": self.title,
            "updated_at": self.updated_at,
            "workspace_path": self.workspace_path,
            "source_path": self.source_path,
            "message_count": self.message_count,
            "preview": self.preview,
        }


@dataclass(frozen=True)
class ExternalSessionMessage:
    role: str
    content: str
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ExternalSessionDetail:
    summary: ExternalSessionSummary
    messages: tuple[ExternalSessionMessage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreparedPromptRequest:
    task: str
    provider: str | None = None
    session_ids: tuple[str, ...] = ()
    include_workspace_scan: bool = True
    include_prior_context: bool = True
    output_format: str = "compact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "provider": self.provider,
            "session_ids": list(self.session_ids),
            "include_workspace_scan": self.include_workspace_scan,
            "include_prior_context": self.include_prior_context,
            "output_format": self.output_format,
        }


@dataclass(frozen=True)
class PreparedPromptResult:
    prompt: str
    provider: str | None = None
    session_ids: tuple[str, ...] = ()
    workspace_facts: tuple[str, ...] = ()
    prior_context: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "provider": self.provider,
            "session_ids": list(self.session_ids),
            "workspace_facts": list(self.workspace_facts),
            "prior_context": list(self.prior_context),
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
        }
