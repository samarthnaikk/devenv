from __future__ import annotations

import tempfile
import unittest
from typing import Any

from core.runtime import DevenvKernel
from core.tools.read_file import ReadFileTool

class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[str] = []


class FakeAI:
        self.registered_tools: list[str] = []

    def register_tool(self, tool) -> None:
        self.registered_tools.append(tool.name)


class DevenvKernelTest(unittest.TestCase):
    def test_register_tool_syncs_runtime_and_ai(self) -> None:
        memory = FakeMemory()
        ai = FakeAI([])
        with tempfile.TemporaryDirectory() as tempdir:
            kernel = DevenvKernel(tempdir, memory=memory, ai=ai)
            kernel.register_tool(ReadFileTool())

        self.assertIn("read_file", kernel.tools)
        self.assertEqual(ai.registered_tools, ["read_file"])


if __name__ == "__main__":
    unittest.main()
