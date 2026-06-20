from __future__ import annotations

from collections.abc import Iterable

from core.memory.interface import MemoryEngineInterface
from core.tools import (
    AuditChangesTool,
    EditFileTool,
    InspectSymbolsTool,
    InspectTraceTool,
    ListDirectoryTool,
    LocateFilesTool,
    ManageMemoryTool,
    PeekLinesTool,
    ReadFileTool,
    RemoveFileTool,
    RunDiagnosticsTool,
    RunShellTool,
    SearchTextTool,
    TrackSymbolTool,
    WriteFileTool,
)
from core.tools.base import BaseTool


def build_runtime_tools(memory: MemoryEngineInterface) -> list[BaseTool]:
    return [
        ListDirectoryTool(),
        LocateFilesTool(),
        ReadFileTool(),
        PeekLinesTool(),
        InspectSymbolsTool(),
        SearchTextTool(),
        TrackSymbolTool(),
        WriteFileTool(),
        EditFileTool(),
        RemoveFileTool(),
        RunShellTool(),
        RunDiagnosticsTool(),
        AuditChangesTool(),
        ManageMemoryTool(memory),
        InspectTraceTool(memory),
    ]


def tool_name_set(tools: Iterable[BaseTool]) -> set[str]:
    return {tool.name for tool in tools}
