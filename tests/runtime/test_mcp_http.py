from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock, patch

from core.runtime.mcp_http import MCPHTTPServerConfig, MCPHTTPServerManager, default_mcp_http_server_config


class MCPHTTPServerManagerTest(unittest.TestCase):
    def test_default_config_reads_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DEVENV_MCP_SERVER_URL": "http://127.0.0.1:9001/mcp",
                "DEVENV_MCP_AUTH_TOKEN": "token-1",
                "DEVENV_MCP_SERVER_STARTUP_TIMEOUT": "5",
                "DEVENV_MCP_SERVER_LOG_LEVEL": "DEBUG",
            },
            clear=False,
        ):
            config = default_mcp_http_server_config()

        self.assertEqual(config.base_url, "http://127.0.0.1:9001/mcp")
        self.assertEqual(config.auth_token, "token-1")
        self.assertEqual(config.startup_timeout_seconds, 5.0)
        self.assertEqual(config.log_level, "DEBUG")

    @patch("core.runtime.mcp_http._tcp_endpoint_reachable", side_effect=[False, True])
    @patch("core.runtime.mcp_http.subprocess.Popen")
    def test_ensure_server_starts_subprocess_when_endpoint_is_down(self, popen_mock: Mock, reachable_mock: Mock) -> None:
        process = Mock()
        process.poll.return_value = None
        process.pid = 4242
        popen_mock.return_value = process

        with tempfile.TemporaryDirectory() as tempdir:
            manager = MCPHTTPServerManager(
                workspace_path=tempdir,
                db_path="memory.db",
                vector_dir="vectors",
            )
            status = manager.ensure_server()

        self.assertTrue(status.reachable)
        self.assertTrue(status.started_by_manager)
        self.assertEqual(status.pid, 4242)
        self.assertTrue(popen_mock.called)
        self.assertGreaterEqual(reachable_mock.call_count, 2)

    @patch("core.runtime.mcp_http._tcp_endpoint_reachable", return_value=True)
    def test_inspect_reports_auth_and_reachability(self, reachable_mock: Mock) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manager = MCPHTTPServerManager(
                workspace_path=tempdir,
                db_path="memory.db",
                vector_dir="vectors",
                config=MCPHTTPServerConfig(
                    base_url="http://127.0.0.1:8765/mcp",
                    auth_token="secret",
                    startup_timeout_seconds=10.0,
                    log_level="INFO",
                ),
            )
            status = manager.inspect()

        self.assertTrue(status.reachable)
        self.assertTrue(status.auth_enabled)
        self.assertEqual(status.base_url, "http://127.0.0.1:8765/mcp")
        reachable_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
