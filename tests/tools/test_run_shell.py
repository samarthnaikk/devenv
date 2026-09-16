from __future__ import annotations

import subprocess
import time
import unittest
from unittest import mock

from core.tools.run_shell import RunShellTool


class RunShellToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = RunShellTool()

    def test_raw_mode_captures_stdout(self) -> None:
        result = self.tool.execute(command="printf 'hello-shell'", mode="raw")

        self.assertTrue(result.success)
        self.assertEqual(result.data["stdout"], "hello-shell")

    def test_background_mode_returns_pid(self) -> None:
        result = self.tool.execute(command="true", mode="background")

        self.assertTrue(result.success)
        self.assertGreater(result.data["pid"], 0)
        time.sleep(0.05)
        self.tool._background_processes[result.data["pid"]].wait(timeout=1)

    @mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep 2", timeout=1, output="", stderr=""))
    def test_raw_mode_returns_structured_timeout_details(self, _mock_run) -> None:
        result = self.tool.execute(command="sleep 2", mode="raw", timeout=1)

        self.assertFalse(result.success)
        self.assertTrue(result.data["timed_out"])
        self.assertEqual(result.data["timeout"], 1)
        self.assertEqual(result.data["command"], "sleep 2")
