from __future__ import annotations

from typing import Any

from core.runtime.models import PreparedPromptRequest

from .base import BaseTool, ToolResult
from .web_search import WebSearchTool


class GeneratePromptTool(BaseTool):
    name = "generate_prompt"
    description = "Generate a strict coding prompt using workspace context, optional memory, and optional web research."

    def __init__(self, *, context_builder=None, web_search_tool: WebSearchTool | None = None) -> None:
        self.context_builder = context_builder
        self.web_search_tool = web_search_tool or WebSearchTool()

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The coding task to prepare a prompt for."},
                "allow_memory": {"type": "string", "enum": ["true", "false"], "description": "Whether prior memory may be used."},
                "allow_web_search": {"type": "string", "enum": ["true", "false"], "description": "Whether web research may be added."},
                "provider": {"type": "string", "description": "Optional session provider to bias context selection."},
                "session_ids": {
                    "type": "string",
                    "description": "Optional comma-separated session IDs to force into the prompt context.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["compact", "detailed", "strict"],
                    "description": "How detailed the generated prompt should be.",
                },
            },
            "required": ["task"],
        }

    def execute(self, **kwargs) -> ToolResult:
        task = kwargs.get("task")
        if not isinstance(task, str) or not task.strip():
            return ToolResult(success=False, output="Missing required argument: task", data={"status": "invalid_input"})
        if self.context_builder is None:
            return ToolResult(
                success=False,
                output="generate_prompt requires a context builder service",
                data={"status": "unsupported"},
            )
        allow_memory = _parse_bool(kwargs.get("allow_memory"), default=True)
        allow_web_search = _parse_bool(kwargs.get("allow_web_search"), default=False)
        provider = kwargs.get("provider")
        output_format = str(kwargs.get("output_format") or "strict").strip().lower()
        normalized_format = "detailed" if output_format == "detailed" else "compact"
        session_ids = _parse_session_ids(kwargs.get("session_ids"))

        prepared = self.context_builder.prepare_prompt(
            PreparedPromptRequest(
                task=task.strip(),
                provider=provider.strip() if isinstance(provider, str) and provider.strip() else None,
                session_ids=session_ids,
                include_workspace_scan=True,
                include_prior_context=allow_memory,
                output_format=normalized_format,
            )
        )

        prompt_sections = [prepared.prompt.strip()]
        if allow_web_search:
            research = self.web_search_tool.execute(mode="search", query=task.strip(), result_count=3)
            if research.success:
                hints = []
                for result in research.data.get("results", [])[:3]:
                    title = str(result.get("title") or "").strip()
                    url = str(result.get("url") or "").strip()
                    if title and url:
                        hints.append(f"- {title} ({url})")
                if hints:
                    prompt_sections.append("## Web Research Hints\n" + "\n".join(hints))

        if output_format == "strict":
            prompt_sections.append(
                "\n".join(
                    [
                        "## Output Contract",
                        "- Return only the implementation plan or code requested by the user.",
                        "- Do not omit constraints discovered from the provided context.",
                        "- Call out assumptions before making risky changes.",
                    ]
                )
            )

        final_prompt = "\n\n".join(section for section in prompt_sections if section).strip()
        return ToolResult(
            success=True,
            output="generate_prompt prepared a reusable coding prompt",
            data={
                "status": "ok",
                "prompt": final_prompt,
                "provider": prepared.provider,
                "session_ids": list(prepared.session_ids),
                "used_memory": allow_memory,
                "used_web_search": allow_web_search,
            },
        )


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _parse_session_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    return tuple(piece.strip() for piece in value.split(",") if piece.strip())
