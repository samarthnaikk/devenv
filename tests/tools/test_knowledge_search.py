from __future__ import annotations

import unittest
import urllib.parse
from unittest.mock import patch

from core.tools.knowledge_search import KnowledgeSearchTool


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


def _fake_urlopen(request, timeout=10):  # noqa: ARG001
    url = getattr(request, "full_url", str(request))
    if "github.com/search?" in url:
        return _FakeResponse(
            """
            <html><body>
              {"csrf_tokens":{"/example/repo-one/star":{"post":"abc"},"/example/repo-two/unstar":{"post":"xyz"}}}
            </body></html>
            """
        )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("q", [""])[0]
    if "documentation" in query or " docs" in query:
        return _FakeResponse('<a class="result__a" href="https://docs.example.com/guide">Docs Guide</a>')
    if "site:stackoverflow.com" in query:
        return _FakeResponse('<a class="result__a" href="https://stackoverflow.com/questions/123">StackOverflow Thread</a>')
    if "site:reddit.com" in query:
        return _FakeResponse('<a class="result__a" href="https://reddit.com/r/example/post">Reddit Post</a>')
    if "site:youtube.com" in query:
        return _FakeResponse('<a class="result__a" href="https://youtube.com/watch?v=abc">YouTube Video</a>')
    if "site:quora.com" in query:
        return _FakeResponse('<a class="result__a" href="https://quora.com/example">Quora Answer</a>')
    return _FakeResponse('<a class="result__a" href="https://example.com/overview">General Overview</a>')


class KnowledgeSearchToolTest(unittest.TestCase):
    def test_execute_requires_query(self) -> None:
        result = KnowledgeSearchTool().execute()

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    @patch("urllib.request.urlopen", side_effect=_fake_urlopen)
    def test_execute_groups_resources_by_source(self, _mock_urlopen) -> None:
        result = KnowledgeSearchTool().execute(query="calendar app feature", result_count=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "ok")
        resources = {group["source"]: group["results"] for group in result.data["resources"]}
        self.assertEqual(resources["github"][0]["url"], "https://github.com/example/repo-one")
        self.assertEqual(resources["documentation"][0]["title"], "Docs Guide")
        self.assertEqual(resources["stackoverflow"][0]["title"], "StackOverflow Thread")

    @patch("urllib.request.urlopen", side_effect=_fake_urlopen)
    def test_execute_respects_requested_sources(self, _mock_urlopen) -> None:
        result = KnowledgeSearchTool().execute(query="calendar app feature", sources=["github", "youtube"], result_count=1)

        self.assertTrue(result.success)
        resources = result.data["resources"]
        self.assertEqual([group["source"] for group in resources], ["github", "youtube"])

    def test_query_normalization_removes_source_words(self) -> None:
        from core.tools.knowledge_search import _normalize_knowledge_query

        result = _normalize_knowledge_query("find similar github repos reddit threads and youtube videos for calendar app feature")

        self.assertEqual(result, "calendar app feature")

    def test_query_normalization_expands_chatapp(self) -> None:
        from core.tools.knowledge_search import _normalize_knowledge_query

        result = _normalize_knowledge_query("add chatapp references to this codebase")

        self.assertEqual(result, "chat app")

    def test_general_only_reference_requests_expand_to_richer_sources(self) -> None:
        from core.tools.knowledge_search import _expand_requested_sources

        result = _expand_requested_sources("find github repo references for a chat app", ["general"])

        self.assertEqual(result, ["github", "youtube", "documentation", "general"])

    @patch("urllib.request.urlopen", side_effect=_fake_urlopen)
    def test_duplicate_query_reuses_cache(self, mock_urlopen) -> None:
        tool = KnowledgeSearchTool()

        first = tool.execute(query="calendar app feature", sources=["github"], result_count=1)
        second = tool.execute(query="calendar app feature", sources=["github"], result_count=1)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(mock_urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
