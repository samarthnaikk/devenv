from __future__ import annotations

import random
import string
import unittest
from types import SimpleNamespace

import core.runtime.context_builder as context_builder
from core.memory.models import ExternalSessionChunkEmbedding
from core.runtime.context_builder import (
    ContextBuilderService,
    ExternalSessionChunk,
    _chunk_token_hits,
    _exact_prompt_hits,
    _is_ascii_token,
    _token_matches,
    _tokenize,
)
from core.runtime.models import ExternalSessionSummary


def _reference_hits(tokens: set[str], haystack: str) -> tuple[int, int]:
    token_hits = sum(1 for token in tokens if _token_matches(token, haystack))
    exact_hits = _exact_prompt_hits(tokens, [haystack])
    return token_hits, exact_hits


def _optimized_hits(tokens: set[str], chunk: ExternalSessionChunk) -> tuple[int, int]:
    simple_tokens = tuple(token for token in tokens if _is_ascii_token(token))
    compound_tokens = tuple(token for token in tokens if not _is_ascii_token(token))
    return _chunk_token_hits(simple_tokens, compound_tokens, chunk.search_words, chunk.search_text)


def _make_chunk(text: str) -> ExternalSessionChunk:
    return ExternalSessionChunk(
        provider="codex",
        session_id="s1",
        title="Session title",
        workspace_path="/tmp/workspace",
        role="assistant",
        source="codex",
        text=text,
    )


class ChunkTokenHitParityTest(unittest.TestCase):
    """Guards the latency refactor: optimized token-set scoring must equal the regex path."""

    def test_curated_edge_cases(self) -> None:
        cases = [
            ("the reviewer reviews bug fixes review issue", {"reviews", "review", "reviewer", "bug", "bugs", "fix"}),
            ("payload_utils.py handles face-match and test/publish", {"payload_utils", "payload", "face-match", "face", "test/publish", "publish"}),
            ("café foo_bar baz", {"café", "foo_bar", "foo", "bar", "baz"}),
            ("FOO Bar fOo", {"foo", "bar"}),
            ("foo-bar-baz", {"foo-bar", "bar-baz", "foo-bar-baz"}),
            ("authentication_bypass open_email_relay", {"authentication", "bypass", "authentication_bypass", "open", "relay"}),
            ("", {"foo"}),
            ("a", {"a", "a-b"}),
        ]
        for haystack, tokens in cases:
            chunk = _make_chunk(haystack)
            self.assertEqual(
                _reference_hits(set(tokens), chunk.search_text),
                _optimized_hits(set(tokens), chunk),
                msg=f"mismatch for haystack={haystack!r} tokens={tokens!r}",
            )

    def test_randomized_parity(self) -> None:
        rng = random.Random(20260926)
        alphabet = string.ascii_letters + string.digits + "_ -/.\té中/"
        for _ in range(3000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
            token_source = " ".join(
                "".join(rng.choice(string.ascii_letters + string.digits + "_-/") for _ in range(rng.randint(1, 8)))
                for _ in range(rng.randint(1, 8))
            )
            tokens = _tokenize(token_source)
            if not tokens:
                continue
            chunk = _make_chunk(text)
            self.assertEqual(
                _reference_hits(tokens, chunk.search_text),
                _optimized_hits(tokens, chunk),
                msg=f"mismatch for haystack={text!r} tokens={tokens!r}",
            )

    def test_search_text_and_words_are_cached_and_stable(self) -> None:
        chunk = _make_chunk("Review the Payload_Utils face-match flow")
        self.assertIs(chunk.search_text, chunk.search_text)
        self.assertIs(chunk.search_words, chunk.search_words)
        self.assertIn("payload_utils", chunk.search_words)
        self.assertIn("face", chunk.search_words)
        self.assertIn("match", chunk.search_words)
        self.assertEqual(chunk.search_text, chunk.search_text.lower())


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        rng = random.Random(sum(ord(char) for char in text))
        return [rng.uniform(-1.0, 1.0) for _ in range(8)]


@unittest.skipIf(context_builder._np is None, "numpy not installed")
class SemanticHitsParityTest(unittest.TestCase):
    """The NumPy cosine path must select the same sessions/chunks as the pure-Python path."""

    def _make_service(self, vectors, chunk_records) -> ContextBuilderService:
        service = object.__new__(ContextBuilderService)
        service._session_embedder = _FakeEmbedder()  # type: ignore[assignment]
        service._session_vector_cache = {}
        service._session_chunk_record_cache = {}
        service._chunk_embedding_matrix_cache = {}
        service._session_embedding_vectors = lambda name: vectors  # type: ignore[assignment]
        service._session_chunk_records = lambda name: chunk_records  # type: ignore[assignment]
        return service

    @staticmethod
    def _records(session_id: str, count: int, seed: int) -> list[ExternalSessionChunkEmbedding]:
        rng = random.Random(seed)
        return [
            ExternalSessionChunkEmbedding(
                unified_session_id=f"u::{session_id}",
                provider="codex",
                session_id=session_id,
                chunk_index=index,
                content_hash=f"h{index}",
                embedding=tuple(rng.uniform(-1.0, 1.0) for _ in range(8)),
                role="assistant",
                source="codex",
                text=f"chunk {index}",
                indexed_at=0.0,
            )
            for index in range(count)
        ]

    def test_numpy_and_python_paths_agree(self) -> None:
        rng = random.Random(7)
        vectors = {"s3": tuple(rng.uniform(-1.0, 1.0) for _ in range(8))}
        chunk_records = {
            "s1": self._records("s1", 5, seed=1),
            "s2": self._records("s2", 3, seed=2),
        }
        summaries = [
            ExternalSessionSummary("codex", session_id, "title", "2026-01-01T00:00:00Z")
            for session_id in ("s1", "s2", "s3")
        ]
        provider = SimpleNamespace(name="codex")
        variants = ("alpha beta gamma", "delta epsilon")

        service = self._make_service(vectors, chunk_records)
        numpy_hits = service._semantic_session_hits(provider, summaries, variants)

        service._chunk_embedding_matrix_cache = {}
        original = context_builder._np
        context_builder._np = None
        try:
            python_hits = service._semantic_session_hits(provider, summaries, variants)
        finally:
            context_builder._np = original

        self.assertEqual(set(numpy_hits), set(python_hits))
        for session_id, numpy_hit in numpy_hits.items():
            python_hit = python_hits[session_id]
            self.assertAlmostEqual(numpy_hit["score"], python_hit["score"], places=9)
            self.assertEqual(
                [chunk.chunk_index for _score, chunk in numpy_hit["chunks"]],
                [chunk.chunk_index for _score, chunk in python_hit["chunks"]],
            )


if __name__ == "__main__":
    unittest.main()

