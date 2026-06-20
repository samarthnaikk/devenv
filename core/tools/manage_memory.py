from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ManageMemoryTool(BaseTool):
    name = "manage_memory"
    description = "Prune or update memory nodes through the injected memory engine."

    supported_modes: tuple[str, ...] = ("prune", "update")

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Memory node identifier to mutate.",
                },
                "mode": {
                    "type": "string",
                    "description": "Memory control action.",
                    "enum": list(self.supported_modes),
                },
                "text": {
                    "type": "string",
                    "description": "Updated summary text for update mode.",
                },
            },
            "required": ["node_id", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        node_id = kwargs.get("node_id")
        mode = kwargs.get("mode")
        text = kwargs.get("text")

        if not isinstance(node_id, str) or not node_id.strip():
            return ToolResult(success=False, output="Missing required argument: node_id", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            if mode == "prune":
                deleted = bool(self.memory.forget_node(node_id, strategy="prune"))
                logger.info("Pruned memory node: node_id=%s deleted=%s", node_id, deleted)
                return ToolResult(
                    success=deleted,
                    output=f"manage_memory prune completed for {node_id}",
                    data={"node_id": node_id, "mode": mode, "deleted": deleted},
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError("update mode requires a non-empty text argument")
            existing = getattr(self.memory, "store", None).get_node(node_id) if hasattr(self.memory, "store") else None
            payload = {
                "node_id": node_id,
                "parent_id": existing.parent_id if existing else None,
                "label": existing.label if existing else node_id.replace("_", " ").title(),
                "category": existing.category if existing else "manual",
                "summary": text.strip(),
                "edges": getattr(getattr(self.memory, "store", None), "list_edges_for_node", lambda _node_id: [])(node_id),
            }
            updated_id = self.memory.update_associative_tree(payload)
            logger.info("Updated memory node: node_id=%s", updated_id)
            return ToolResult(
                success=True,
                output=f"manage_memory updated node {updated_id}",
                data={"node_id": updated_id, "mode": mode, "summary": text.strip()},
            )
        except (AttributeError, OSError, ValueError, RuntimeError) as exc:
            logger.error("manage_memory failed: node_id=%s mode=%s error=%s", node_id, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})
