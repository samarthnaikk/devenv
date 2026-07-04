from __future__ import annotations

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web or read a specific URL with structured success and failure payloads."

    supported_modes: tuple[str, ...] = ("search", "read_url")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(self.supported_modes),
                    "description": "Choose 'search' to search the web or 'read_url' to fetch one page.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text. Required when mode is 'search'.",
                },
                "url": {
                    "type": "string",
                    "description": "Absolute URL to read. Required when mode is 'read_url'.",
                },
                "result_count": {
                    "type": "integer",
                    "description": "Optional maximum number of search results to return.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider hint for future search backends.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={"status": "invalid_input"})
        if mode == "search":
            query = kwargs.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(success=False, output="Missing required argument: query", data={"status": "invalid_input"})
        if mode == "read_url":
            url = kwargs.get("url")
            if not isinstance(url, str) or not url.strip():
                return ToolResult(success=False, output="Missing required argument: url", data={"status": "invalid_input"})
        return ToolResult(
            success=False,
            output="web_search is registered but provider support is not implemented yet",
            data={
                "status": "unsupported",
                "mode": mode,
                "results": [],
                "content": "",
            },
        )
