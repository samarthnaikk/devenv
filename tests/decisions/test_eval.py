from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.decisions.config import DecisionConfig
from core.decisions.eval import (
    compare_to_baseline,
    default_corpus_dir,
    evaluate,
    load_corpus,
)
from core.runtime import decisions_cli


class EvalCorpusTest(unittest.TestCase):
    def test_shipped_corpora_are_well_formed(self) -> None:
        for domain in ("intent", "memory"):
            items = load_corpus(default_corpus_dir() / f"{domain}.jsonl")
            self.assertGreater(len(items), 0, domain)

    def test_evaluate_intent_heuristic(self) -> None:
        corpus = load_corpus(default_corpus_dir() / "intent.jsonl")
        outcome = evaluate("intent", corpus, config=DecisionConfig.from_env(env={}))
        self.assertEqual(outcome.metrics.total, len(corpus))
        self.assertGreaterEqual(outcome.metrics.accuracy, 0.0)
        self.assertLessEqual(outcome.metrics.accuracy, 1.0)

    def test_evaluate_memory_heuristic(self) -> None:
        corpus = load_corpus(default_corpus_dir() / "memory.jsonl")
        outcome = evaluate("memory", corpus, config=DecisionConfig.from_env(env={}))
        self.assertEqual(outcome.metrics.total, len(corpus))

    def test_compare_to_baseline_detects_regression(self) -> None:
        corpus = load_corpus(default_corpus_dir() / "intent.jsonl")
        outcome = evaluate("intent", corpus, config=DecisionConfig.from_env(env={}))
        baseline = {"metrics": {"accuracy": outcome.metrics.accuracy + 0.2}}
        ok, _detail = compare_to_baseline(outcome, baseline)
        self.assertFalse(ok)


class DecisionsCliTest(unittest.TestCase):
    def test_status_prints_json(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = decisions_cli.main(["status"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("provider", payload)

    def test_eval_prints_metrics(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = decisions_cli.main(["eval", "intent", "--provider", "heuristic"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["domain"], "intent")

    def test_missing_corpus_returns_error(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = decisions_cli.main(["eval", "intent", "--corpus", "/nonexistent/corpus.jsonl"])
        self.assertEqual(code, 2)

    def test_replay_passes_on_matching_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            baseline_dir = Path(tempdir)
            corpus_dir = default_corpus_dir()
            for domain in ("intent", "memory"):
                outcome = evaluate(domain, load_corpus(corpus_dir / f"{domain}.jsonl"), config=DecisionConfig.from_env(env={}))
                (baseline_dir / f"{domain}.json").write_text(
                    json.dumps(outcome.to_dict()), encoding="utf-8"
                )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = decisions_cli.main(
                    ["replay", "--corpus-dir", str(corpus_dir), "--baseline-dir", str(baseline_dir)]
                )
        self.assertEqual(code, 0)

    def test_replay_fails_on_regressed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            baseline_dir = Path(tempdir)
            corpus_dir = default_corpus_dir()
            for domain in ("intent", "memory"):
                (baseline_dir / f"{domain}.json").write_text(
                    json.dumps({"metrics": {"accuracy": 1.1}}), encoding="utf-8"
                )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
                code = decisions_cli.main(
                    ["replay", "--corpus-dir", str(corpus_dir), "--baseline-dir", str(baseline_dir)]
                )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
