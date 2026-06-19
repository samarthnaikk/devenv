from __future__ import annotations

import unittest

from core.ai import AIResponse, ToolCallRequest


class AIModelsTest(unittest.TestCase):
    def test_models_are_importable_and_frozen(self) -> None:
        tool_call = ToolCallRequest(call_id="call_1", tool_name="read_file", arguments={"path": "README.md"})
        response = AIResponse(
            content=None,
            tool_calls=(tool_call,),
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10},
        )

        self.assertEqual(tool_call.tool_name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments["path"], "README.md")


if __name__ == "__main__":
    unittest.main()
