from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ManageMemoryTool(BaseTool):
    name = "manage_memory"
    description = "Prune or update memory nodes through the injected memory engine."

    supported_modes: tuple[str, ...] = ("create", "prune", "update")

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
                "label": {
                    "type": "string",
                    "description": "Optional human-readable label for create mode.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category for create mode.",
                },
                "mode": {
                    "type": "string",
                    "description": "Memory control action.",
                    "enum": list(self.supported_modes),
                },
                "action": {
                    "type": "string",
                    "description": "Alias for mode.",
                    "enum": list(self.supported_modes),
                },
                "text": {
                    "type": "string",
                    "description": "Summary text for create or update mode.",
                },
            },
            "required": ["node_id"],
        }

    def execute(self, **kwargs) -> ToolResult:
        node_id = kwargs.get("node_id")
        mode = kwargs.get("mode")
        action = kwargs.get("action")
        label = kwargs.get("label")
        category = kwargs.get("category")
        text = kwargs.get("text")

        if not isinstance(node_id, str) or not node_id.strip():
            return ToolResult(success=False, output="Missing required argument: node_id", data={})
        if not isinstance(mode, str) or not mode.strip():
            mode = action
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            if mode == "create":
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("create mode requires a non-empty text argument")
                existing = getattr(self.memory, "store", None).get_node(node_id) if hasattr(self.memory, "store") else None
                if existing is not None:
                    return ToolResult(
                        success=False,
                        output=f"Memory node already exists: {node_id}",
                        data={"node_id": node_id, "mode": mode, "created": False},
                    )
                payload = {
                    "node_id": node_id,
                    "parent_id": None,
                    "label": label.strip() if isinstance(label, str) and label.strip() else node_id.replace("_", " ").title(),
                    "category": category.strip() if isinstance(category, str) and category.strip() else "manual",
                    "summary": text.strip(),
                    "edges": (),
                }
                created_id = self.memory.update_associative_tree(payload)
                logger.info("Created memory node: node_id=%s", created_id)
                return ToolResult(
                    success=True,
                    output=f"manage_memory created node {created_id}",
                    data={
                        "node_id": created_id,
                        "mode": mode,
                        "created": True,
                        "summary": text.strip(),
                        "label": payload["label"],
                        "category": payload["category"],
                        "sync_state": _memory_sync_state(self.memory),
                    },
                )

            if mode == "prune":
                deleted = bool(self.memory.forget_node(node_id, strategy="prune"))
                logger.info("Pruned memory node: node_id=%s deleted=%s", node_id, deleted)
                output = f"manage_memory prune completed for {node_id}" if deleted else f"Memory node not found: {node_id}"
                return ToolResult(
                    success=deleted,
                    output=output,
                    data={
                        "node_id": node_id,
                        "mode": mode,
                        "deleted": deleted,
                        "sync_state": _memory_sync_state(self.memory),
                    },
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
                data={
                    "node_id": updated_id,
                    "mode": mode,
                    "summary": text.strip(),
                    "sync_state": _memory_sync_state(self.memory),
                },
            )
        except (AttributeError, OSError, ValueError, RuntimeError) as exc:
            logger.error("manage_memory failed: node_id=%s mode=%s error=%s", node_id, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})


def _memory_sync_state(memory: Any) -> dict[str, Any]:
    store = getattr(memory, "store", None)
    if store is None or not hasattr(store, "get_state"):
        return {}
    return {
        "last_vector_sync_node_id": store.get_state("last_vector_sync_node_id"),
        "last_vector_delete_node_id": store.get_state("last_vector_delete_node_id"),
        "last_vector_sync_at": store.get_state("last_vector_sync_at"),
    }
