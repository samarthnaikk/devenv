from __future__ import annotations

import unittest
from pathlib import Path

from core.tools.read_file import ReadFileTool


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "sample-test" / "tool-fixtures"


class ReadFileToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReadFileTool()

    def test_default_feature_returns_content(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT / "README.md"))

        self.assertTrue(result.success)
        self.assertIn("Tool Fixture Workspace", result.data["content"])

    def test_all_feature_returns_content_metadata_and_extension(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT / "app" / "main.py"), features="all")

        self.assertTrue(result.success)
        self.assertIn("content", result.data)
        self.assertIn("metadata", result.data)
        self.assertIn("extension", result.data)
        self.assertEqual(result.data["extension"]["extension"], "py")

    def test_invalid_feature_is_rejected(self) -> None:
        result = self.tool.execute(path=str(FIXTURE_ROOT / "README.md"), features="unknown")

        self.assertFalse(result.success)
        self.assertIn("Unsupported feature", result.output)
