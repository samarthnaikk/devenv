from __future__ import annotations

import unittest
from unittest import mock

from core.runtime.models import ExecutionMode
from core.runtime.tool_policy import (
    PLAN_READ_ONLY_TOOLS,
    TOOL_POLICY_REGISTRY,
    allowed_tool_names_for_mode,
    build_tool_policy_event,
    classify_tool,
    plan_read_only_tools,
)
from core.runtime.tooling import build_runtime_tools


class ToolPolicyTest(unittest.TestCase):
    def test_plan_mode_only_allows_read_or_research_tools(self) -> None:
        allowed = allowed_tool_names_for_mode(
            ExecutionMode.PLAN_ONLY,
            {"read_file", "list_directory", "edit_file", "run_shell", "inspect_trace"},
        )

        self.assertEqual(allowed, {"read_file", "list_directory", "inspect_trace"})

    def test_verification_mode_limits_to_diagnostic_and_read_tools(self) -> None:
        allowed = allowed_tool_names_for_mode(
            ExecutionMode.VERIFICATION,
            {"read_file", "run_diagnostics", "audit_changes", "edit_file", "web_search"},
        )

        self.assertEqual(allowed, {"read_file", "run_diagnostics", "audit_changes"})

    def test_classify_tool_carries_mutation_category_metadata(self) -> None:
        spec = classify_tool("remove_file")

        self.assertEqual(spec.category, "delete")
        self.assertTrue(spec.mutable)
        self.assertTrue(spec.destructive)

    def test_registry_covers_every_registered_tool(self) -> None:
        tools = build_runtime_tools(mock.MagicMock(store=None))
        registered = {tool.name for tool in tools}
        self.assertEqual(registered - set(TOOL_POLICY_REGISTRY), set())

    def test_plan_read_only_scope_is_derived_from_registry(self) -> None:
        expected = {name for name, spec in TOOL_POLICY_REGISTRY.items() if spec.read_only}
        self.assertEqual(set(PLAN_READ_ONLY_TOOLS), expected)
        self.assertIn("read_file", PLAN_READ_ONLY_TOOLS)
        self.assertNotIn("write_file", PLAN_READ_ONLY_TOOLS)

    def test_plan_read_only_tools_intersects_availability(self) -> None:
        scoped = plan_read_only_tools({"read_file", "web_search", "not_a_tool"})
        self.assertEqual(scoped, {"read_file"})

    def test_web_and_kernel_plan_scope_agree(self) -> None:
        from core.runtime.web import READ_ONLY_PLAN_TOOLS

        self.assertEqual(tuple(READ_ONLY_PLAN_TOOLS), PLAN_READ_ONLY_TOOLS)

    def test_non_read_only_tools_are_not_plan_allowed(self) -> None:
        for name in ("web_search", "knowledge_search", "generate_prompt", "generate_pdf"):
            self.assertFalse(TOOL_POLICY_REGISTRY[name].plan_allowed, name)

    def test_build_tool_policy_event_includes_retry_safety(self) -> None:
        event = build_tool_policy_event(
            "run_diagnostics",
            ExecutionMode.VERIFICATION,
            "allow",
            "Diagnostics are required after mutation.",
        )

        self.assertEqual(event.category, "diagnostic")
        self.assertEqual(event.mode, "verification")
        self.assertTrue(event.retry_safe)


if __name__ == "__main__":
    unittest.main()
