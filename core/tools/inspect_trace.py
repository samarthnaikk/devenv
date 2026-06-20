from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class InspectTraceTool(BaseTool):
    name = "inspect_trace"
    description = "Inspect the last retrieval trace or the stored history for a specific memory node."

    supported_modes: tuple[str, ...] = ("last_retrieval", "node_history")

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Trace inspection mode.",
                    "enum": list(self.supported_modes),
                },
                "node_id": {
                    "type": "string",
                    "description": "Required for node_history mode.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        node_id = kwargs.get("node_id")

        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            if mode == "last_retrieval":
                trace = self.memory.get_context_trace()
                payload = asdict(trace)
                logger.info("Inspected last retrieval trace")
                return ToolResult(
                    success=True,
                    output="inspect_trace returned the last retrieval trace",
                    data={"mode": mode, "trace": payload},
                )

            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError("node_history mode requires a node_id argument")

            store = getattr(self.memory, "store", None)
            if store is None:
                raise ValueError("Injected memory engine does not expose a storage interface")
            node = store.get_node(node_id)
            if node is None:
                raise ValueError(f"Memory node not found: {node_id}")
            edges = [asdict(edge) for edge in store.list_edges_for_node(node_id)]
            vector_present = getattr(getattr(self.memory, "vector_index", None), "records", {}).get(node_id) is not None
            logger.info("Inspected node history: node_id=%s", node_id)
            return ToolResult(
                success=True,
                output=f"inspect_trace returned node history for {node_id}",
                data={
                    "mode": mode,
                    "node": asdict(node),
                    "edges": edges,
                    "vector_present": vector_present,
                },
            )
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.error("inspect_trace failed: mode=%s node_id=%s error=%s", mode, node_id, exc)
            return ToolResult(success=False, output=str(exc), data={})
