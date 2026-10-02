from __future__ import annotations

import unittest

from core.decisions.models import DecisionError
from core.decisions.questions import build_intent_questions, parse_intent_answers


class BuildIntentQuestionsTest(unittest.TestCase):
    def test_shape(self) -> None:
        questions = build_intent_questions()
        self.assertIn("route", questions)
        self.assertIn("mutates_workspace", questions)
        route = questions["route"]
        self.assertEqual(route["type"], "choice")
        self.assertEqual(
            set(route["criteria"]),
            {"recall", "explain", "inspect", "mutate", "meta"},
        )
        self.assertEqual(questions["mutates_workspace"]["type"], "noul")


class ParseIntentAnswersTest(unittest.TestCase):
    def test_recall_maps_to_local_knowledge(self) -> None:
        result = parse_intent_answers(
            {
                "route": {"type": "choice", "choice": "recall", "confidence": 0.91},
                "mutates_workspace": {"type": "noul", "noul": 0.1},
            }
        )
        self.assertEqual(result.route, "recall")
        self.assertEqual(result.confidence, 0.91)
        self.assertTrue(result.use_local_knowledge)
        self.assertEqual(result.reason, "systemone:recall")

    def test_mutate_never_uses_local_knowledge(self) -> None:
        result = parse_intent_answers(
            {
                "route": {"type": "choice", "choice": "mutate", "confidence": 0.99},
                "mutates_workspace": {"type": "noul", "noul": 0.95},
            }
        )
        self.assertFalse(result.use_local_knowledge)

    def test_mutation_overrides_recall(self) -> None:
        result = parse_intent_answers(
            {
                "route": {"type": "choice", "choice": "explain", "confidence": 0.8},
                "mutates_workspace": {"type": "noul", "noul": 0.8},
            }
        )
        self.assertFalse(result.use_local_knowledge)

    def test_missing_mutation_answer_defaults_read_only(self) -> None:
        result = parse_intent_answers({"route": {"type": "choice", "choice": "recall", "confidence": 0.7}})
        self.assertEqual(result.mutates_workspace, 0.0)
        self.assertTrue(result.use_local_knowledge)

    def test_missing_route_raises(self) -> None:
        with self.assertRaises(DecisionError):
            parse_intent_answers({"mutates_workspace": {"type": "noul", "noul": 0.5}})

    def test_empty_choice_raises(self) -> None:
        with self.assertRaises(DecisionError):
            parse_intent_answers({"route": {"type": "choice", "choice": ""}})


if __name__ == "__main__":
    unittest.main()
