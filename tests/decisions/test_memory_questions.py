from __future__ import annotations

import unittest

from core.decisions.questions import build_memory_write_questions, parse_memory_write_answers


class BuildMemoryWriteQuestionsTest(unittest.TestCase):
    def test_one_noul_per_candidate_with_structured_instructions(self) -> None:
        questions = build_memory_write_questions(["Django auth uses session cookies.", "Hello!"])
        self.assertEqual(set(questions), {"write_0", "write_1"})
        first = questions["write_0"]
        self.assertEqual(first["type"], "noul")
        self.assertEqual(first["instructions"]["candidate"], "Django auth uses session cookies.")
        self.assertIn("question", first["instructions"])
        self.assertEqual(set(first["criteria"]), {"true", "false"})

    def test_empty_candidates(self) -> None:
        self.assertEqual(build_memory_write_questions([]), {})


class ParseMemoryWriteAnswersTest(unittest.TestCase):
    def test_high_probability_writes(self) -> None:
        decisions = parse_memory_write_answers({"write_0": {"type": "noul", "noul": 0.9}}, 1)
        self.assertTrue(decisions[0].should_write)
        self.assertEqual(decisions[0].source, "systemone")
        self.assertAlmostEqual(decisions[0].confidence, 0.8)

    def test_low_probability_drops(self) -> None:
        decisions = parse_memory_write_answers({"write_0": {"type": "noul", "noul": 0.1}}, 1)
        self.assertFalse(decisions[0].should_write)
        self.assertEqual(decisions[0].source, "systemone")

    def test_missing_answer_keeps_candidate(self) -> None:
        decisions = parse_memory_write_answers({}, 2)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(decision.should_write for decision in decisions))
        self.assertTrue(all(decision.source == "heuristic" for decision in decisions))

    def test_mixed_batch(self) -> None:
        decisions = parse_memory_write_answers(
            {
                "write_0": {"type": "noul", "noul": 0.95},
                "write_1": {"type": "noul", "noul": 0.05},
            },
            2,
        )
        self.assertTrue(decisions[0].should_write)
        self.assertFalse(decisions[1].should_write)


if __name__ == "__main__":
    unittest.main()