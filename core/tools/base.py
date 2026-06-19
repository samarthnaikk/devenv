from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str
    data: dict[str, Any]


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError
