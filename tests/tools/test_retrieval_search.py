from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.tools.retrieval_search import RetrievalSearchTool


class _FakeMemory:
    def __init__(self, markdown: str = "", *, raises: Exception | None = None) -> None:
        self.markdown = markdown
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def retrieve_context(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(markdown_context=self.markdown)


class _FakeContextBuilder:
    def __init__(
        self,
        markdown: str = "",
        session_ids: tuple[str, ...] = (),
        metadata: dict | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.markdown = markdown
        self.session_ids = session_ids
        self.metadata = metadata or {}
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def build_runtime_memory_context(self, query: str, *, provider_name=None, max_lines: int = 6):
        self.calls.append((query, max_lines))
        if self.raises is not None:
            raise self.raises
        return self.markdown, list(self.session_ids), self.metadata


class RetrievalSearchToolTest(unittest.TestCase):
    def test_execute_requires_query(self) -> None:
        result = RetrievalSearchTool().execute()

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    def test_execute_rejects_blank_query(self) -> None:
        result = RetrievalSearchTool().execute(query="   ")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    def test_execute_rejects_unknown_scope(self) -> None:
        result = RetrievalSearchTool(memory=_FakeMemory()).execute(query="x", scope="everything")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    def test_execute_rejects_non_integer_result_count(self) -> None:
        result = RetrievalSearchTool(memory=_FakeMemory()).execute(query="x", result_count="lots")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "invalid_input")

    def test_result_count_is_clamped_to_bounds(self) -> None:
        memory = _FakeMemory(markdown="- one\n")
        tool = RetrievalSearchTool(memory=memory)

        tool.execute(query="high", result_count=99)
        tool.execute(query="low", result_count=0)

        self.assertEqual(memory.calls, [("high", 10), ("low", 1)])

    def test_memory_scope_uses_retrieve_context_and_returns_lines(self) -> None:
        memory = _FakeMemory(markdown="# Heading\n- [episode] hello world\n- [working] second line\n")
        result = RetrievalSearchTool(memory=memory).execute(query="hello", scope="memory", result_count=2)

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(memory.calls, [("hello", 2)])
        self.assertEqual(result.data["lines"], ["[episode] hello world", "[working] second line"])

    def test_sessions_scope_surfaces_session_ids_and_providers(self) -> None:
        context_builder = _FakeContextBuilder(
            markdown="- prior decision about retrieval\n",
            session_ids=("sess_a", "sess_b"),
            metadata={"context_match_providers": ["codex", "opencode"], "context_match_reason": "lexical overlap"},
        )
        result = RetrievalSearchTool(context_builder=context_builder).execute(
            query="retrieval", scope="sessions", result_count=3
        )

        self.assertTrue(result.success)
        self.assertEqual(context_builder.calls, [("retrieval", 3)])
        self.assertEqual(result.data["session_ids"], ["sess_a", "sess_b"])
        self.assertEqual(result.data["providers"], ["codex", "opencode"])
        self.assertIn("lexical overlap", result.data["notes"])
        self.assertEqual(result.data["lines"], ["prior decision about retrieval"])

    def test_all_scope_merges_sessions_and_memory(self) -> None:
        memory = _FakeMemory(markdown="- memory line\n")
        context_builder = _FakeContextBuilder(markdown="- session line\n", session_ids=("sess_a",))
        result = RetrievalSearchTool(memory=memory, context_builder=context_builder).execute(
            query="merge", scope="all"
        )

        self.assertEqual(result.data["lines"], ["session line", "memory line"])
        self.assertEqual(result.data["session_ids"], ["sess_a"])

    def test_char_ceiling_truncates_and_flags(self) -> None:
        memory = _FakeMemory(markdown="- " + ("a" * 5000) + "\n")
        tool = RetrievalSearchTool(memory=memory, max_chars=200)

        result = tool.execute(query="long", scope="memory")

        self.assertTrue(result.data["truncated"])
        self.assertLessEqual(sum(len(line) for line in result.data["lines"]), 200)

    def test_duplicate_query_reuses_cache(self) -> None:
        memory = _FakeMemory(markdown="- cached line\n")
        tool = RetrievalSearchTool(memory=memory)

        first = tool.execute(query="same", scope="memory")
        second = tool.execute(query="same", scope="memory")

        self.assertIs(first, second)
        self.assertEqual(len(memory.calls), 1)

    def test_graceful_degradation_when_sessions_unattached(self) -> None:
        memory = _FakeMemory(markdown="- only memory\n")
        result = RetrievalSearchTool(memory=memory).execute(query="x", scope="all")

        self.assertTrue(result.success)
        self.assertTrue(any("sessions" in note for note in result.data["notes"]))
        self.assertEqual(result.data["lines"], ["only memory"])

    def test_graceful_degradation_when_memory_unattached(self) -> None:
        context_builder = _FakeContextBuilder(markdown="- only sessions\n", session_ids=("sess_a",))
        result = RetrievalSearchTool(context_builder=context_builder).execute(query="x", scope="all")

        self.assertTrue(result.success)
        self.assertTrue(any("memory" in note for note in result.data["notes"]))
        self.assertEqual(result.data["lines"], ["only sessions"])

    def test_unavailable_without_any_engine(self) -> None:
        result = RetrievalSearchTool().execute(query="x")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "unavailable")

    def test_engine_errors_do_not_propagate(self) -> None:
        memory = _FakeMemory(raises=RuntimeError("boom"))
        context_builder = _FakeContextBuilder(raises=RuntimeError("kaboom"))

        result = RetrievalSearchTool(memory=memory, context_builder=context_builder).execute(query="x", scope="all")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "no_results")
        self.assertEqual(len(result.data["errors"]), 2)

    def test_no_results_status_when_engines_return_nothing(self) -> None:
        result = RetrievalSearchTool(memory=_FakeMemory()).execute(query="x", scope="memory")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "no_results")


if __name__ == "__main__":
    unittest.main()
