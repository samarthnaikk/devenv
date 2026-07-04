from __future__ import annotations

import unittest

from core.tools.web_search import WebSearchTool


class WebSearchToolTest(unittest.TestCase):
    def test_input_schema_exposes_search_and_read_url_modes(self) -> None:
        schema = WebSearchTool().input_schema()

        self.assertEqual(schema["required"], ["mode"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["search", "read_url"])

    def test_execute_requires_query_for_search_mode(self) -> None:
        result = WebSearchTool().execute(mode="search")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    def test_execute_returns_structured_unsupported_payload_until_provider_added(self) -> None:
        result = WebSearchTool().execute(mode="read_url", url="https://example.com")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "unsupported")
        self.assertEqual(result.data["mode"], "read_url")


if __name__ == "__main__":
    unittest.main()
