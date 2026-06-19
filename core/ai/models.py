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
