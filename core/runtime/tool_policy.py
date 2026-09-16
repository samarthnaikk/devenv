from __future__ import annotations

from dataclasses import dataclass

from .models import ExecutionMode, ToolPolicyEvent


@dataclass(frozen=True)
class ToolPolicySpec:
    name: str
    category: str
    mutable: bool = False
    destructive: bool = False
    plan_allowed: bool = False
    verification_allowed: bool = False
    retry_safe: bool = False


TOOL_POLICY_REGISTRY: dict[str, ToolPolicySpec] = {
    "list_directory": ToolPolicySpec("list_directory", "inspect", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "locate_files": ToolPolicySpec("locate_files", "inspect", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "read_file": ToolPolicySpec("read_file", "inspect", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "peek_lines": ToolPolicySpec("peek_lines", "inspect", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "inspect_symbols": ToolPolicySpec("inspect_symbols", "inspect", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "search_text": ToolPolicySpec("search_text", "search", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "track_symbol": ToolPolicySpec("track_symbol", "search", plan_allowed=True, verification_allowed=True, retry_safe=True),
    "web_search": ToolPolicySpec("web_search", "web", plan_allowed=True, verification_allowed=False, retry_safe=True),
    "knowledge_search": ToolPolicySpec("knowledge_search", "knowledge", plan_allowed=True, verification_allowed=False, retry_safe=True),
    "generate_pdf": ToolPolicySpec("generate_pdf", "artifact", mutable=True, verification_allowed=False, retry_safe=False),
    "generate_prompt": ToolPolicySpec("generate_prompt", "artifact", plan_allowed=True, verification_allowed=False, retry_safe=True),
    "write_file": ToolPolicySpec("write_file", "edit", mutable=True, verification_allowed=False, retry_safe=False),
    "edit_file": ToolPolicySpec("edit_file", "edit", mutable=True, verification_allowed=False, retry_safe=False),
    "remove_file": ToolPolicySpec("remove_file", "delete", mutable=True, destructive=True, verification_allowed=False, retry_safe=False),
    "run_shell": ToolPolicySpec("run_shell", "shell", mutable=True, verification_allowed=False, retry_safe=False),
    "run_diagnostics": ToolPolicySpec("run_diagnostics", "diagnostic", plan_allowed=False, verification_allowed=True, retry_safe=True),
    "audit_changes": ToolPolicySpec("audit_changes", "diagnostic", plan_allowed=False, verification_allowed=True, retry_safe=True),
    "manage_memory": ToolPolicySpec("manage_memory", "memory", mutable=True, verification_allowed=False, retry_safe=False),
    "inspect_trace": ToolPolicySpec("inspect_trace", "memory", plan_allowed=True, verification_allowed=True, retry_safe=True),
}


def classify_tool(tool_name: str) -> ToolPolicySpec:
    return TOOL_POLICY_REGISTRY.get(tool_name, ToolPolicySpec(name=tool_name, category="other"))


def allowed_tool_names_for_mode(mode: ExecutionMode, available_tool_names: set[str]) -> set[str]:
    allowed: set[str] = set()
    for tool_name in available_tool_names:
        spec = classify_tool(tool_name)
        if mode is ExecutionMode.PLAN_ONLY:
            if spec.plan_allowed:
                allowed.add(tool_name)
            continue
        if mode is ExecutionMode.VERIFICATION:
            if spec.verification_allowed:
                allowed.add(tool_name)
            continue
        if mode in {ExecutionMode.CHECKPOINT_EXECUTE, ExecutionMode.REPAIR, ExecutionMode.DIRECT_ANSWER}:
            allowed.add(tool_name)
            continue
        if mode is ExecutionMode.BLOCKED_FOR_CLARIFICATION:
            if spec.plan_allowed:
                allowed.add(tool_name)
    return allowed


def build_tool_policy_event(tool_name: str, mode: ExecutionMode, decision: str, reason: str) -> ToolPolicyEvent:
    spec = classify_tool(tool_name)
    return ToolPolicyEvent(
        tool_name=tool_name,
        category=spec.category,
        decision=decision,
        reason=reason,
        mode=mode.value,
        retry_safe=spec.retry_safe,
    )
