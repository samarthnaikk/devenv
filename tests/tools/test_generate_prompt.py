from __future__ import annotations

import unittest

from core.runtime.models import PreparedPromptResult
from core.tools.generate_prompt import GeneratePromptTool


class _FakeContextBuilder:
    def prepare_prompt(self, request):
        return PreparedPromptResult(
            prompt=f"Task: {request.task}",
            provider=request.provider,
            session_ids=request.session_ids,
        )


class _FakeWebSearchTool:
    def execute(self, **kwargs):
        return type(
            "Result",
            (),
            {
                "success": True,
                "output": "ok",
                "data": {"results": [{"title": "Guide", "url": "https://example.com/guide"}]},
            },
        )()


class GeneratePromptToolTest(unittest.TestCase):
    def test_tool_requires_context_builder(self) -> None:
        result = GeneratePromptTool(context_builder=None).execute(task="Build a feature")

        self.assertFalse(result.success)
        self.assertEqual(result.data["status"], "unsupported")

    def test_tool_builds_prompt_from_context_and_web_hints(self) -> None:
        tool = GeneratePromptTool(context_builder=_FakeContextBuilder(), web_search_tool=_FakeWebSearchTool())

        result = tool.execute(task="Build a feature", allow_web_search="true", output_format="strict")

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "ok")
        self.assertIn("Task: Build a feature", result.data["prompt"])
        self.assertIn("## Web Research Hints", result.data["prompt"])
        self.assertIn("## Output Contract", result.data["prompt"])


if __name__ == "__main__":
    unittest.main()
