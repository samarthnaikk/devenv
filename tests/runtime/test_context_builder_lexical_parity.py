from __future__ import annotations

import random
import string
import unittest

from core.runtime.context_builder import (
    ExternalSessionChunk,
    _chunk_token_hits,
    _exact_prompt_hits,
    _is_ascii_token,
    _token_matches,
    _tokenize,
)


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


if __name__ == "__main__":
    unittest.main()
