from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class InspectAuditTool(BaseTool):
    name = "inspect_audit"
    description = "Query or verify the durable runtime audit trail for this workspace."

    supported_modes: tuple[str, ...] = ("list", "turn", "verify")

    def __init__(self, store: Any) -> None:
        self.store = store

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Audit inspection mode.",
                    "enum": list(self.supported_modes),
                },
                "turn_id": {"type": "string", "description": "Required for turn mode."},
                "event_type": {"type": "string", "description": "Optional event-type filter."},
                "limit": {"type": "integer", "description": "Max events to return (default 50)."},
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if self.store is None or not hasattr(self.store, "list_runtime_events"):
            return ToolResult(success=False, output="Audit store is unavailable.", data={})

        try:
            if mode == "turn":
                turn_id = kwargs.get("turn_id")
                if not isinstance(turn_id, str) or not turn_id.strip():
                    raise ValueError("turn mode requires a turn_id argument")
                events = self.store.list_runtime_events(turn_id=turn_id, limit=int(kwargs.get("limit") or 200))
                return ToolResult(
                    success=True,
                    output=f"inspect_audit returned {len(events)} event(s) for turn {turn_id}",
                    data={"mode": mode, "turn_id": turn_id, "events": events},
                )

            if mode == "verify":
                events = list(reversed(self.store.list_runtime_events(limit=100000)))
                from core.runtime.audit import AuditRecorder

                ok, detail = AuditRecorder.verify_chain(events)
                return ToolResult(
                    success=ok,
                    output=f"Audit chain verification: {detail}",
                    data={"mode": mode, "verified": ok, "detail": detail, "events_checked": len(events)},
                )

            event_type = kwargs.get("event_type")
            events = self.store.list_runtime_events(
                event_type=str(event_type) if event_type else None,
                limit=int(kwargs.get("limit") or 50),
            )
            return ToolResult(
                success=True,
                output=f"inspect_audit returned {len(events)} event(s)",
                data={"mode": mode, "events": events},
            )
        except (ValueError, TypeError) as exc:
            logger.error("inspect_audit failed: mode=%s error=%s", mode, exc)
            return ToolResult(success=False, output=str(exc), data={})


__all__ = ["InspectAuditTool"]
