from __future__ import annotations

import importlib.util
import inspect
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

from core.runtime.mcp_client import MCPToolClient
from core.runtime.mcp_server import _annotation_for_property, _build_tool_wrapper
from core.tools.generate_pdf import GeneratePDFTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"
SAMPLE_ROOT = Path(__file__).resolve().parents[2] / "sample-test"


class MCPServerSchemaTest(unittest.TestCase):
    def test_annotation_for_property_supports_boolean_and_array_fields(self) -> None:
        keep_tex_annotation = _annotation_for_property(
            "keep_tex",
            {"type": "boolean", "description": "Keep the tex source."},
            required=False,
        )
        sections_annotation = _annotation_for_property(
            "sections",
            {"type": "array", "items": {"type": "object"}},
            required=True,
        )

        keep_tex_base = get_args(keep_tex_annotation)[0]
        self.assertEqual(str(keep_tex_base), "bool | None")
        self.assertEqual(str(sections_annotation), "list[dict[str, typing.Any]]")

    def test_generate_pdf_wrapper_exposes_native_schema_types(self) -> None:
        wrapper = _build_tool_wrapper(GeneratePDFTool())
        signature = inspect.signature(wrapper)

        sections_base = get_args(signature.parameters["sections"].annotation)[0]
        keep_tex_base = get_args(signature.parameters["keep_tex"].annotation)[0]
        self.assertEqual(str(sections_base), "list[dict[str, typing.Any]]")
        self.assertEqual(str(keep_tex_base), "bool | None")


@unittest.skipIf(importlib.util.find_spec("mcp") is None, "Optional mcp dependency is not installed")
class MCPRuntimeTest(unittest.TestCase):
    def test_list_tools_over_stdio_exposes_all_runtime_schemas(self) -> None:
        client = MCPToolClient(
            workspace_path=str(FIXTURE_ROOT),
            db_path="memory.db",
            vector_dir="vectors",
        )
        try:
            tools = client.list_tools()
        finally:
            client.close()

        self.assertEqual(len(tools), 15)
        self.assertIn("read_file", tools)
        self.assertIn("list_directory", tools)
        self.assertEqual(tools["read_file"]["inputSchema"]["required"], ["path"])
        self.assertEqual(
            tools["list_directory"]["inputSchema"]["properties"]["mode"]["enum"],
            ["flat", "recursive", "topology"],
        )

    def test_close_shuts_down_stdio_context(self) -> None:
        state: dict[str, bool] = {"stdio_closed": False, "session_closed": False}

        @asynccontextmanager
        async def fake_stdio_client(_params):
            try:
                yield object(), object()
            finally:
                state["stdio_closed"] = True

        class FakeSession:
            def __init__(self, *_args, **_kwargs) -> None:
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                state["session_closed"] = True

            async def initialize(self) -> None:
                return None

            async def list_tools(self):
                return SimpleNamespace(tools=[])

        with patch("core.runtime.mcp_client.stdio_client", fake_stdio_client), patch(
            "core.runtime.mcp_client.ClientSession",
            FakeSession,
        ):
            client = MCPToolClient(
                workspace_path=str(FIXTURE_ROOT),
                db_path="memory.db",
                vector_dir="vectors",
            )
            client.start()
            client.close()

        self.assertTrue(state["session_closed"])
        self.assertTrue(state["stdio_closed"])

    def test_call_tool_preserves_quoted_source_payloads(self) -> None:
        with tempfile.TemporaryDirectory(dir=SAMPLE_ROOT) as tempdir:
            workspace = Path(tempdir)
            target = workspace / "quoted.py"
            target.write_text('def demo():\n    return "hello \\"mcp\\""\n', encoding="utf-8")
            client = MCPToolClient(
                workspace_path=str(workspace),
                db_path="memory.db",
                vector_dir="vectors",
            )
            try:
                result = client.call_tool("read_file", {"path": str(target)})
            finally:
                client.close()

        self.assertTrue(result.success)
        self.assertIn('\\"mcp\\"', result.data["content"])


if __name__ == "__main__":
    unittest.main()
