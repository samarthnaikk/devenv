from __future__ import annotations

import tempfile
import unittest

from core.runtime.kernel import DevenvKernel
from core.runtime.tooling import build_runtime_tools


class _FakeMemory:
    pass


class KernelToolPolicyTest(unittest.TestCase):
    def test_selected_tool_scope_is_filtered_by_phase_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=_FakeMemory())
            for tool in build_runtime_tools(kernel.memory):
                kernel.register_tool(tool)

            direct_scope = kernel._tool_scope_for_prompt(
                "search the docs",
                selected_tools=["web_search", "read_file", "edit_file"],
                execution_phase=False,
            )
            execution_scope = kernel._tool_scope_for_prompt(
                "fix the backend",
                selected_tools=["web_search", "read_file", "edit_file"],
                execution_phase=True,
            )

        self.assertEqual(direct_scope, ["edit_file", "read_file", "web_search"])
        self.assertEqual(execution_scope, ["edit_file", "read_file", "web_search"])

    def test_planning_allowed_tool_names_excludes_mutation_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=_FakeMemory())
            for tool in build_runtime_tools(kernel.memory):
                kernel.register_tool(tool)

            allowed = kernel._planning_allowed_tool_names()

        self.assertIn("read_file", allowed)
        self.assertNotIn("edit_file", allowed)
        self.assertNotIn("write_file", allowed)


if __name__ == "__main__":
    unittest.main()
