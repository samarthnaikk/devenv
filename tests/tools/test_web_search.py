from __future__ import annotations

import unittest
import urllib.error
from unittest.mock import patch

from core.tools.web_search import WebSearchTool


class _FakeHeaders:
    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")
        self.headers = _FakeHeaders()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class WebSearchToolTest(unittest.TestCase):
    def test_input_schema_exposes_search_and_read_url_modes(self) -> None:
        schema = WebSearchTool().input_schema()

        self.assertEqual(schema["required"], ["mode"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["search", "search_images", "read_url"])

    def test_execute_requires_query_for_search_mode(self) -> None:
        result = WebSearchTool().execute(mode="search")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(
            """
            <html><body>
              <a class="result__a" href="https://example.com/one">First Result</a>
              <a class="result__a" href="https://example.com/two">Second Result</a>
            </body></html>
            """
        ),
    )
    def test_search_returns_normalized_results(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search", query="devenv", result_count=2)

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(len(result.data["results"]), 2)
        self.assertEqual(result.data["results"][0]["title"], "First Result")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(
            """
            <html><body>
              <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fexample%2Frepo">Example Repo</a>
            </body></html>
            """
        ),
    )
    def test_search_decodes_duckduckgo_redirect_results(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search", query="example repo", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["url"], "https://github.com/example/repo")

    @patch(
        "urllib.request.urlopen",
        side_effect=[
            _FakeResponse("<html><body>No matches</body></html>"),
            _FakeResponse("<rss><channel></channel></rss>"),
            _FakeResponse(
                """
                <html><body>
                  <li class="b_algo"><h2><a href="https://example.com/bing-lite">Lite Result</a></h2></li>
                </body></html>
                """
            ),
        ],
    )
    def test_search_falls_back_to_lite_results_when_primary_markup_is_empty(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search", query="fallback test", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["title"], "Lite Result")

    @patch(
        "urllib.request.urlopen",
        side_effect=[
            urllib.error.URLError("blocked"),
            _FakeResponse(
                """
                <rss><channel>
                  <item><title>Bing RSS Result</title><link>https://example.com/rss</link></item>
                </channel></rss>
                """
            ),
        ],
    )
    def test_search_falls_back_to_bing_rss_results(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search", query="rss fallback", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["title"], "Bing RSS Result")

    @patch(
        "urllib.request.urlopen",
        side_effect=[
            urllib.error.URLError("blocked"),
            _FakeResponse("<rss><channel></channel></rss>"),
            _FakeResponse(
                """
                <html><body>
                  <li class="b_algo"><h2><a href="https://example.com/bing">Bing Result</a></h2></li>
                </body></html>
                """
            ),
        ],
    )
    def test_search_falls_back_to_bing_markup(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search", query="bing fallback", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["title"], "Bing Result")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse("<html><head><title>Example</title></head><body><h1>Hello</h1><p>World</p></body></html>"),
    )
    def test_read_url_returns_readable_content(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="read_url", url="https://example.com")

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(result.data["title"], "Example")
        self.assertIn("Hello World", result.data["content"])

    def test_read_url_rejects_non_http_urls(self) -> None:
        result = WebSearchTool().execute(mode="read_url", url="file:///tmp/test.txt")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_url")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(
            """
            <html><body>
              murl&quot;:&quot;https://images.example.com/hero.png&quot;, &quot;t&quot;:&quot;Hero Shot&quot;
            </body></html>
            """
        ),
    )
    def test_search_images_returns_normalized_results(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search_images", query="chat ui", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["url"], "https://images.example.com/hero.png")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(
            """
            <html><body>
              murl&quot;:&quot;https://video.example.com/thumb.png&quot;, &quot;t&quot;:&quot;YouTube Creator Cup&quot;
              murl&quot;:&quot;https://images.example.com/architecture-diagram.png&quot;, &quot;t&quot;:&quot;Software architecture diagram&quot;
            </body></html>
            """
        ),
    )
    def test_search_images_prefers_query_relevant_results(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search_images", query="software architecture illustration", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["url"], "https://images.example.com/architecture-diagram.png")

    @patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(
            """
            <html><body>
              murl&quot;:&quot;https://video.example.com/thumb.png&quot;, &quot;t&quot;:&quot;YouTube Creator Cup&quot;
              murl&quot;:&quot;https://news.example.com/crowd.jpg&quot;, &quot;t&quot;:&quot;Breaking local news&quot;
            </body></html>
            """
        ),
    )
    def test_search_images_skips_multiple_irrelevant_candidates(self, _mock_urlopen) -> None:
        result = WebSearchTool().execute(mode="search_images", query="software architecture illustration", result_count=2)

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "no_results")


if __name__ == "__main__":
    unittest.main()
