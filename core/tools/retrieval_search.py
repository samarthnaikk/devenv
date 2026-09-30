from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 2400
DEFAULT_RESULT_COUNT = 5
VALID_SCOPES: tuple[str, ...] = ("all", "sessions", "memory")
SCOPE_SESSIONS = "sessions"
SCOPE_MEMORY = "memory"
SCOPE_ALL = "all"


class RetrievalSearchTool(BaseTool):
    """Expose the Devenv retrieval engine as a callable tool.

    This is the "retrieval engine as one tool among several" surface: the model
    decides *when* context is needed and *what* to look for, instead of receiving
    a pre-assembled memory block. It is deliberately additive — it reads the
    engine's existing public entry points and never changes ranking.

    Two scopes are available:
      - ``sessions``: the cross-provider external session archives (Codex + OpenCode)
        via ``ContextBuilderService.build_runtime_memory_context``.
      - ``memory``: the local cognitive memory (working / episodic / associative)
        via ``MemoryEngine.retrieve_context``.
    """

    name = "retrieval_search"
    description = (
        "Search Devenv's own memory for relevant prior context. "
        "Covers indexed Codex and OpenCode session archives (scope=sessions) and the local "
        "cognitive memory - working, episodic and associative (scope=memory). "
        "Use this before planning or answering when the result should be grounded in what was "
        "already discussed or decided, in this project or in a previous one."
    )

    def __init__(
        self,
        *,
        memory: Any | None = None,
        context_builder: Any | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._memory = memory
        self._context_builder = context_builder
        self._max_chars = max(200, int(max_chars))
        self._cache: dict[tuple[str, str, int], ToolResult] = {}

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up in prior memory.",
                },
                "scope": {
                    "type": "string",
                    "enum": list(VALID_SCOPES),
                    "description": (
                        "sessions = indexed Codex/OpenCode archives, "
                        "memory = local cognitive memory, all = both."
                    ),
                },
                "result_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum evidence lines per scope.",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return _invalid("Missing required argument: query")

        scope = str(kwargs.get("scope") or SCOPE_ALL).strip().lower()
        if scope not in VALID_SCOPES:
            return _invalid(f"Unsupported scope '{scope}'. Use one of: {', '.join(VALID_SCOPES)}.")

        try:
            result_count = max(1, min(int(kwargs.get("result_count", DEFAULT_RESULT_COUNT)), 10))
        except (TypeError, ValueError):
            return _invalid("result_count must be an integer between 1 and 10.")

        if self._memory is None and self._context_builder is None:
            return ToolResult(
                success=False,
                output="retrieval_search has no memory engine or session index attached.",
                data=_empty_data("unavailable", str(query).strip(), scope),
            )

        query_text = query.strip()
        cache_key = (query_text.lower(), scope, result_count)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        lines: list[str] = []
        session_ids: list[str] = []
        providers: list[str] = []
        notes: list[str] = []
        errors: list[str] = []

        if scope in {SCOPE_ALL, SCOPE_SESSIONS}:
            if self._context_builder is None:
                notes.append("External session index is not attached; skipped scope 'sessions'.")
            else:
                self._collect_sessions(
                    query_text,
                    result_count=result_count,
                    lines=lines,
                    session_ids=session_ids,
                    providers=providers,
                    notes=notes,
                    errors=errors,
                )

        if scope in {SCOPE_ALL, SCOPE_MEMORY}:
            if self._memory is None:
                notes.append("Local memory engine is not attached; skipped scope 'memory'.")
            else:
                self._collect_memory_lines(
                    query_text,
                    result_count=result_count,
                    lines=lines,
                    errors=errors,
                )

        budgeted, truncated = _apply_char_ceiling(lines, self._max_chars)
        status = "ok" if budgeted or session_ids else "no_results"
        result = ToolResult(
            success=bool(budgeted or session_ids),
            output=_format_output(
                query=query_text,
                scope=scope,
                lines=budgeted,
                session_ids=session_ids,
                truncated=truncated,
                status=status,
            ),
            data={
                "status": status,
                "source": "retrieval",
                "query": query_text,
                "scope": scope,
                "result_count": result_count,
                "sessions": [{"session_id": session_id} for session_id in session_ids],
                "session_ids": list(session_ids),
                "providers": sorted({provider for provider in providers if provider}),
                "lines": list(budgeted),
                "truncated": truncated,
                "notes": list(notes),
                "errors": list(errors),
            },
        )
        self._cache[cache_key] = result
        return result

    # -- scope collectors -------------------------------------------------

    def _collect_sessions(
        self,
        query_text: str,
        *,
        result_count: int,
        lines: list[str],
        session_ids: list[str],
        providers: list[str],
        notes: list[str],
        errors: list[str],
    ) -> None:
        try:
            markdown, selected_ids, metadata = self._context_builder.build_runtime_memory_context(
                query_text,
                max_lines=result_count,
            )
        except Exception as exc:  # pragma: no cover - the engine already guards most paths
            logger.warning("retrieval_search session lookup failed: %s", exc)
            errors.append(f"sessions: {exc}")
            return

        lines.extend(_markdown_to_lines(markdown))
        for session_id in selected_ids or ():
            cleaned = str(session_id).strip()
            if cleaned and cleaned not in session_ids:
                session_ids.append(cleaned)
        for provider in metadata.get("context_match_providers", ()) if isinstance(metadata, dict) else ():
            cleaned = str(provider).strip()
            if cleaned and cleaned not in providers:
                providers.append(cleaned)
        if isinstance(metadata, dict):
            reason = str(metadata.get("context_match_reason") or "").strip()
            if reason:
                notes.append(reason)

    def _collect_memory_lines(
        self,
        query_text: str,
        *,
        result_count: int,
        lines: list[str],
        errors: list[str],
    ) -> None:
        try:
            result = self._memory.retrieve_context(query_text, top_k=result_count)
        except Exception as exc:  # pragma: no cover - the engine already guards most paths
            logger.warning("retrieval_search memory lookup failed: %s", exc)
            errors.append(f"memory: {exc}")
            return
        lines.extend(_markdown_to_lines(getattr(result, "markdown_context", "") or ""))


def _invalid(detail: str) -> ToolResult:
    return ToolResult(
        success=False,
        output=detail,
        data=_empty_data("invalid_input", "", ""),
    )


def _empty_data(status: str, query: str, scope: str) -> dict[str, Any]:
    return {
        "status": status,
        "source": "retrieval",
        "query": query,
        "scope": scope,
        "result_count": 0,
        "sessions": [],
        "session_ids": [],
        "providers": [],
        "lines": [],
        "truncated": False,
        "notes": [],
        "errors": [],
    }


def _markdown_to_lines(markdown: str) -> list[str]:
    """Flatten the engine's markdown block into individual evidence lines."""

    collected: list[str] = []
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line:
            collected.append(line)
    return collected


def _apply_char_ceiling(lines: list[str], max_chars: int) -> tuple[list[str], bool]:
    """Hard-cap the evidence text so one tool call cannot flood the context window."""

    budgeted: list[str] = []
    used = 0
    for line in lines:
        remaining = max_chars - used
        if remaining <= 0:
            return budgeted, True
        if len(line) > remaining:
            budgeted.append(line[:remaining].rstrip())
            return budgeted, True
        budgeted.append(line)
        used += len(line) + 1
    return budgeted, False


def _format_output(
    *,
    query: str,
    scope: str,
    lines: list[str],
    session_ids: list[str],
    truncated: bool,
    status: str,
) -> str:
    if status == "no_results":
        return f"retrieval_search found no prior context for '{query}' (scope={scope})."

    header = (
        f"retrieval_search matched {len(session_ids)} session(s) with "
        f"{len(lines)} evidence line(s) for '{query}' (scope={scope})"
    )
    if truncated:
        header += "; evidence truncated to fit the context budget"
    parts = [header]
    if session_ids:
        parts.append("session_ids: " + ", ".join(session_ids))
    parts.extend(lines)
    return "\n".join(parts)


__all__ = ["RetrievalSearchTool", "DEFAULT_MAX_CHARS"]