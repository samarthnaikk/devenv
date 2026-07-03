from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AIResponse:
    content: str | None
    tool_calls: tuple[ToolCallRequest, ...]
    finish_reason: str
    usage: dict[str, int]


@dataclass(frozen=True)
class AIBackendStatus:
    name: str
    available: bool
    enabled: bool = True
    model: str = ""
    detail: str = ""
    supports_tool_calls: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "enabled": self.enabled,
            "model": self.model,
            "detail": self.detail,
            "supports_tool_calls": self.supports_tool_calls,
        }
