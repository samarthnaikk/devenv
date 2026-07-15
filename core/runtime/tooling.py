from __future__ import annotations

from collections.abc import Iterable

from core.memory.interface import MemoryEngineInterface
from core.tools import (
    AuditChangesTool,
    EditFileTool,
    GeneratePromptTool,
    InspectSymbolsTool,
    InspectTraceTool,
    KnowledgeSearchTool,
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
    WebSearchTool,
    WriteFileTool,
)
from core.tools.base import BaseTool


def build_runtime_tools(memory: MemoryEngineInterface, *, context_builder=None) -> list[BaseTool]:
    web_search_tool = WebSearchTool()
    return [
        ListDirectoryTool(),
        LocateFilesTool(),
        ReadFileTool(),
        PeekLinesTool(),
        InspectSymbolsTool(),
        SearchTextTool(),
        web_search_tool,
        KnowledgeSearchTool(),
        GeneratePromptTool(context_builder=context_builder, web_search_tool=web_search_tool),
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
