from __future__ import annotations

import unittest

from core.runtime.models import ExecutionMode
from core.runtime.tool_policy import (
    allowed_tool_names_for_mode,
    build_tool_policy_event,
    classify_tool,
)


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
