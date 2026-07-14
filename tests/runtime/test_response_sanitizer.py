from __future__ import annotations

import unittest

from core.runtime.response_sanitizer import sanitize_response_text


class ResponseSanitizerPresentationTest(unittest.TestCase):
    def test_sanitize_response_text_inserts_breaks_before_inline_markdown_headings(self) -> None:
        cleaned = sanitize_response_text(
            "Here's the integration plan. ## Claude Code Integration Plan ### Files to Create - core/ai/claude_code_backend.py - core/ai/routing.py"
        )

        self.assertIsNotNone(cleaned)
        self.assertIn("Here's the integration plan.\n\n## Claude Code Integration Plan", cleaned)
        self.assertIn("\n\n### Files to Create\n- core/ai/claude_code_backend.py\n- core/ai/routing.py", cleaned)

    def test_sanitize_response_text_promotes_inline_numbered_items_to_list_lines(self) -> None:
        cleaned = sanitize_response_text(
            "We should package it like this: 1. System Message 2. Retrieved Memory Block 3. Working Memory Snapshot 4. Current User Prompt"
        )

        self.assertIsNotNone(cleaned)
        self.assertIn("We should package it like this:\n1. System Message\n2. Retrieved Memory Block", cleaned)


if __name__ == "__main__":
    unittest.main()
