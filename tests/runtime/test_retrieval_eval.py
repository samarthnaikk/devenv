from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "retrieval_eval.py"
_spec = importlib.util.spec_from_file_location("retrieval_eval", _SCRIPT)
retrieval_eval = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(retrieval_eval)


class RankingMetricTest(unittest.TestCase):
    def test_precision_at_k(self) -> None:
        self.assertEqual(retrieval_eval.precision_at_k({"a"}, ["a", "b", "c"], 1), 1.0)
        self.assertEqual(retrieval_eval.precision_at_k({"a"}, ["b", "a"], 1), 0.0)
        self.assertEqual(retrieval_eval.precision_at_k({"a"}, ["b", "a"], 2), 0.5)
        self.assertEqual(retrieval_eval.precision_at_k({"a"}, [], 3), 0.0)

    def test_ndcg_at_k(self) -> None:
        self.assertEqual(retrieval_eval.ndcg_at_k({"a"}, ["a"], 1), 1.0)
        self.assertEqual(retrieval_eval.ndcg_at_k({"a"}, [], 4), 0.0)
        # Relevant item at rank 2: dcg = 1/log2(3); idcg (1 relevant, k=4) = 1/log2(2)
        expected = round((1.0 / 1.584962500721156) / (1.0 / 1.0), 3)
        self.assertEqual(retrieval_eval.ndcg_at_k({"a"}, ["b", "a"], 4), expected)

    def test_ndcg_perfect_ordering(self) -> None:
        self.assertEqual(retrieval_eval.ndcg_at_k({"a", "b"}, ["a", "b", "c"], 4), 1.0)


class SameProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspaces = {
            "s1": "/Users/x/devenv",
            "s2": "/Users/x/devenv/",
            "s3": "/Users/x/facepred",
            "s4": None,
        }

    def test_precision_counts_matches(self) -> None:
        value = retrieval_eval.same_project_precision(
            ["s1", "s3", "s2"], self.workspaces, "/Users/x/devenv", 3
        )
        self.assertEqual(value, 0.667)

    def test_unknown_workspace_counts_as_miss(self) -> None:
        value = retrieval_eval.same_project_precision(
            ["s1", "s4"], self.workspaces, "/Users/x/devenv", 2
        )
        self.assertEqual(value, 0.5)

    def test_missing_target_workspace_is_zero(self) -> None:
        self.assertEqual(
            retrieval_eval.same_project_precision(["s1"], self.workspaces, None, 3), 0.0
        )

    def test_ground_truth_project_prefers_known(self) -> None:
        project = retrieval_eval.ground_truth_project({"s4", "s3"}, self.workspaces)
        self.assertEqual(project, "/Users/x/facepred")

    def test_normalize_workspace(self) -> None:
        self.assertEqual(
            retrieval_eval._normalize_workspace("/Users/X/devenv/"),
            retrieval_eval._normalize_workspace("/Users/x/devenv"),
        )


class ScoreQuestionTest(unittest.TestCase):
    def _record(self) -> dict:
        return {
            "id": "Q1",
            "runtime": {"session_ids": ["s1", "s3", "s2"], "context": "the proof answer"},
            "per_provider": {},
            "candidates": {},
            "workspace_by_session": {
                "s1": "/Users/x/devenv",
                "s2": "/Users/x/devenv",
                "s3": "/Users/x/facepred",
            },
            "selector": {"session_ids": ["s3", "s1"]},
        }

    def test_selects_and_scores_project_and_selector(self) -> None:
        truth = {"session_ids": ["s1"], "keywords": ["proof"], "proof": "proof answer"}
        score = retrieval_eval.score_question(
            self._record(), truth, [], set(), workspace_path="/Users/x/devenv"
        )
        self.assertEqual(score["runtime_rank"], 1)
        self.assertEqual(score["same_project_precision_at_3"], 0.667)
        self.assertEqual(score["ground_truth_project"], "/Users/x/devenv")
        self.assertTrue(score["ground_truth_same_project"])
        self.assertEqual(score["precision_at_1"], 1.0)
        self.assertEqual(score["selector_rank"], 2)

    def test_no_selector_record_leaves_rank_none(self) -> None:
        record = self._record()
        record["selector"] = {}
        truth = {"session_ids": ["s1"], "keywords": [], "proof": ""}
        score = retrieval_eval.score_question(
            record, truth, [], set(), workspace_path="/Users/x/devenv"
        )
        self.assertIsNone(score["selector_rank"])
        self.assertEqual(score["precision_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
