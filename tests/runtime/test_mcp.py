from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

from core.runtime.mcp_client import MCPToolClient
from core.runtime.mcp_server import (
    _annotation_for_property,
    _build_auth_settings,
    _build_static_token_verifier,
    _build_tool_wrapper,
    create_mcp_server,
)
from core.runtime.sandbox import PathSandbox
from core.runtime.tooling import build_runtime_tools
from core.tools import ListDirectoryTool, ReadFileTool, WriteFileTool
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


class MCPSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=SAMPLE_ROOT)
        self.workspace = Path(self._tmp.name)
        (self.workspace / "inside.txt").write_text("inside", encoding="utf-8")
        self.sandbox = PathSandbox(str(self.workspace))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_relative_escape_is_denied(self) -> None:
        wrapper = _build_tool_wrapper(ReadFileTool(), self.sandbox)
        result = json.loads(wrapper(path="../outside.txt"))
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "sandbox_violation")

    def test_absolute_escape_is_denied(self) -> None:
        wrapper = _build_tool_wrapper(WriteFileTool(), self.sandbox)
        result = json.loads(wrapper(path="/tmp/devenv_mcp_escape.txt", content="x", mode="fresh"))
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "sandbox_violation")

    def test_symlink_escape_is_denied(self) -> None:
        link = self.workspace / "escape_link"
        link.symlink_to("/etc/hosts")
        wrapper = _build_tool_wrapper(ReadFileTool(), self.sandbox)
        result = json.loads(wrapper(path="escape_link"))
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "sandbox_violation")

    def test_list_directory_escape_is_denied(self) -> None:
        wrapper = _build_tool_wrapper(ListDirectoryTool(), self.sandbox)
        result = json.loads(wrapper(path="/tmp"))
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["status"], "sandbox_violation")

    def test_inside_path_is_allowed_and_normalized(self) -> None:
        wrapper = _build_tool_wrapper(ReadFileTool(), self.sandbox)
        result = json.loads(wrapper(path="inside.txt"))
        self.assertTrue(result["success"])


@unittest.skipIf(importlib.util.find_spec("mcp") is None, "Optional mcp dependency is not installed")
class MCPHttpConfigTest(unittest.TestCase):
    def test_auth_settings_use_host_and_port(self) -> None:
        settings = _build_auth_settings("127.0.0.1", 8765)
        self.assertEqual(str(settings.issuer_url).rstrip("/"), "http://127.0.0.1:8765")
        self.assertEqual(str(settings.resource_server_url).rstrip("/"), "http://127.0.0.1:8765")

    def test_static_token_verifier_accepts_and_rejects(self) -> None:
        import asyncio

        verifier = _build_static_token_verifier("s3cret")
        self.assertIsNotNone(asyncio.run(verifier.verify_token("s3cret")))
        self.assertIsNone(asyncio.run(verifier.verify_token("wrong")))

    def test_create_mcp_server_forwards_http_settings(self) -> None:
        captured: dict = {}

        class FakeMCP:
            def __init__(self, name, **kwargs) -> None:
                captured.update(kwargs)

            def add_tool(self, *args, **kwargs) -> None:
                return None

        with patch("core.runtime.mcp_server._load_fastmcp", return_value=FakeMCP), patch(
            "core.runtime.mcp_server.MemoryEngine"
        ), patch(
            "core.runtime.mcp_server.resolve_memory_paths", return_value=("memory.db", "vectors")
        ):
            create_mcp_server(
                workspace_path="/tmp/ws",
                host="127.0.0.1",
                port=8765,
                streamable_http_path="/custom",
                auth_token="tok",
            )

        self.assertEqual(captured["host"], "127.0.0.1")
        self.assertEqual(captured["port"], 8765)
        self.assertEqual(captured["streamable_http_path"], "/custom")
        self.assertIn("token_verifier", captured)
        self.assertIn("auth", captured)

    def test_create_mcp_server_wires_context_builder_into_tools(self) -> None:
        captured: dict = {}

        class FakeMCP:
            def __init__(self, name, **kwargs) -> None:
                return None

            def add_tool(self, *args, **kwargs) -> None:
                return None

        def fake_build(memory, *, context_builder=None):
            captured["context_builder"] = context_builder
            return []

        with patch("core.runtime.mcp_server._load_fastmcp", return_value=FakeMCP), patch(
            "core.runtime.mcp_server.MemoryEngine"
        ), patch(
            "core.runtime.mcp_server.resolve_memory_paths", return_value=("memory.db", "vectors")
        ), patch(
            "core.runtime.mcp_server.build_runtime_tools", side_effect=fake_build
        ), patch(
            "core.runtime.mcp_server.ContextBuilderService"
        ) as builder_cls:
            create_mcp_server(workspace_path="/tmp/ws")

        self.assertIs(captured["context_builder"], builder_cls.return_value)

    def test_create_mcp_server_omits_auth_without_token(self) -> None:
        captured: dict = {}

        class FakeMCP:
            def __init__(self, name, **kwargs) -> None:
                captured.update(kwargs)

            def add_tool(self, *args, **kwargs) -> None:
                return None

        with patch("core.runtime.mcp_server._load_fastmcp", return_value=FakeMCP), patch(
            "core.runtime.mcp_server.MemoryEngine"
        ), patch(
            "core.runtime.mcp_server.resolve_memory_paths", return_value=("memory.db", "vectors")
        ):
            create_mcp_server(workspace_path="/tmp/ws")

        self.assertNotIn("token_verifier", captured)
        self.assertNotIn("auth", captured)


@unittest.skipIf(importlib.util.find_spec("mcp") is None, "Optional mcp dependency is not installed")
class MCPRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        # The spawned MCP server writes logs/audit under its workspace. Point
        # them at a temp dir so we never pollute the shared fixture tree.
        self._env = patch.dict(
            "os.environ",
            {"DEVENV_LOG_TO_FILE": "0", "DEVENV_AUDIT_TO_FILE": "0", "DEVENV_AUDIT": "0"},
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        stray = FIXTURE_ROOT / ".devenv"
        if stray.exists():
            import shutil

            shutil.rmtree(stray, ignore_errors=True)

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

        expected = {tool.name for tool in build_runtime_tools(SimpleNamespace())}
        self.assertEqual(set(tools), expected)
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
