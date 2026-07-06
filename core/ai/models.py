from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AIExecutedToolStep:
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    output: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass(frozen=True)
class AIBackendTurnResult:
    content: str | None
    finish_reason: str
    tool_calls: tuple[ToolCallRequest, ...] = ()
    executed_steps: tuple[AIExecutedToolStep, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    backend: str = "opencode"
    metadata: dict[str, Any] = field(default_factory=dict)
    abortable: bool = True


@dataclass(frozen=True)
class AIResponse(AIBackendTurnResult):
    pass


@dataclass(frozen=True)
class AIBackendStatus:
    name: str
    available: bool
    enabled: bool = True
    model: str = ""
    detail: str = ""
    supports_tool_calls: bool = True
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "enabled": self.enabled,
            "model": self.model,
            "detail": self.detail,
            "supports_tool_calls": self.supports_tool_calls,
            "metadata": dict(self.metadata or {}),
        }
