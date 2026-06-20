from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.ai.models import ToolCallRequest

from .local_model import LocalSmallModel
from .models import StageTrace

READ_ONLY_CONTEXT_TOOLS = ("list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols", "search_text", "track_symbol")


@dataclass(frozen=True)
class ContextPacket:
    checkpoint_id: int
    memory_context: str
    distilled_context: str
    workspace_facts: tuple[str, ...] = ()
    used_local_model_fallback: bool = False
    local_model_name: str = "deterministic-fallback"
    source_lines: tuple[str, ...] = ()

    def as_prompt_block(self) -> str:
        if not self.distilled_context.strip() and not self.workspace_facts:
            return ""
        sections = ["## Context Packet"]
        if self.distilled_context.strip():
            sections.append(self.distilled_context.strip())
        if self.workspace_facts:
            sections.append("## Workspace Facts")
            sections.extend(f"- {fact}" for fact in self.workspace_facts)
        return "\n".join(section for section in sections if section).strip()


def build_context_packet(
    *,
    checkpoint_id: int,
    checkpoint_objective: str,
    memory_context: str,
    local_model: LocalSmallModel,
    tool_names: list[str],
    execute_tool_call,
    char_limit: int,
) -> tuple[ContextPacket, StageTrace]:
    logs = [f"Checkpoint objective: {checkpoint_objective}"]
    candidate_lines = _extract_memory_candidates(memory_context)
    workspace_facts: list[str] = []

    if not candidate_lines and "list_directory" in tool_names:
        step = execute_tool_call(
            ToolCallRequest(
                call_id=f"context_list_{checkpoint_id}",
                tool_name="list_directory",
                arguments={"path": ".", "mode": "flat", "max_depth": 2},
            )
        )
        if step.success:
            logs.append("Read-only workspace scan added to context stage")
            workspace_facts.extend(_workspace_facts_from_output(step.output))
        else:
            logs.append("Workspace scan was attempted but did not succeed")

    selection = local_model.distill(checkpoint_objective, list(candidate_lines) + workspace_facts, max_lines=6)
    distilled = selection.summary[:char_limit].rstrip()
    if not distilled:
        distilled = memory_context.strip()[:char_limit].rstrip()

    packet = ContextPacket(
        checkpoint_id=checkpoint_id,
        memory_context=memory_context,
        distilled_context=distilled,
        workspace_facts=tuple(workspace_facts[:6]),
        used_local_model_fallback=selection.used_fallback,
        local_model_name=selection.model_name,
        source_lines=selection.selected_lines,
    )
    trace = StageTrace(
        stage="context_memory",
        checkpoint_id=checkpoint_id,
        success=True,
        summary="Built distilled context packet",
        logs=logs,
        payload={
            "local_model_name": selection.model_name,
            "used_fallback": selection.used_fallback,
            "memory_chars": len(memory_context),
            "distilled_chars": len(packet.distilled_context),
            "workspace_fact_count": len(packet.workspace_facts),
        },
    )
    return packet, trace


def _extract_memory_candidates(memory_context: str) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in memory_context.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
    return tuple(lines)


def _workspace_facts_from_output(output: str) -> list[str]:
    payload = _extract_tool_payload_json(output)
    facts: list[str] = []
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("topology") or []
        if isinstance(entries, list):
            for entry in entries[:8]:
                if isinstance(entry, dict):
                    relative_path = entry.get("relative_path")
                    if isinstance(relative_path, str) and relative_path.strip():
                        facts.append(f"Workspace path: {relative_path.strip()}")
    return facts


def _extract_tool_payload_json(output: str) -> dict[str, Any] | list[Any] | None:
    if not output:
        return None
    brace_index = output.find("{")
    bracket_index = output.find("[")
    start_candidates = [index for index in (brace_index, bracket_index) if index >= 0]
    if not start_candidates:
        return None
    try:
        return json.loads(output[min(start_candidates) :])
    except json.JSONDecodeError:
        return None
