from __future__ import annotations

from pathlib import Path
from typing import Any

from core.ai import AICore
from core.ai.models import AIResponse, ToolCallRequest
from core.memory import MemoryEngine
from core.tools.base import BaseTool

from .models import RuntimeTurnResult, ToolExecutionStep
from .sandbox import PathSandbox


class DevenvKernel:
    def __init__(
        self,
        workspace_path: str,
        db_path: str = "memory.db",
        vector_dir: str = "vectors",
        *,
        memory: MemoryEngine | Any | None = None,
        ai: AICore | Any | None = None,
    ):
        self.workspace_path = str(Path(workspace_path).expanduser().resolve())
        self.sandbox = PathSandbox(root_path=self.workspace_path)
        self.memory = memory or MemoryEngine(db_path=db_path, vector_dir=vector_dir)
        self.ai = ai or AICore()
        self.tools: dict[str, BaseTool] = {}
        self.ephemeral_history: list[dict[str, Any]] = []

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        self.ai.register_tool(tool)

    def execute_turn(self, user_prompt: str, max_consecutive_tools: int = 5) -> RuntimeTurnResult:
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        self.memory.record_working_memory(
            messages=conversation[-10:],
            active_state={"workspace_path": self.workspace_path},
        )
        memory_result = self.memory.retrieve_context(user_prompt)
        ai_response = self.ai.chat(messages=list(conversation), memory_context=memory_result.markdown_context)
        if ai_response.tool_calls:
            return RuntimeTurnResult(final_response=None, steps=[self._intercept_tool_call(tool_call) for tool_call in ai_response.tool_calls])
        final_response = ai_response.content
        if final_response is not None:
            conversation.append({"role": "assistant", "content": final_response})
        self.ephemeral_history = conversation
        return RuntimeTurnResult(final_response=final_response, total_usage=dict(ai_response.usage))

    def _intercept_tool_call(self, tool_call: ToolCallRequest) -> ToolExecutionStep:
        unsafe_argument = self.sandbox.find_unsafe_argument(tool_call.arguments)
        if unsafe_argument is not None:
            _key, value = unsafe_argument
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=self.sandbox.violation_message(value),
                success=False,
                is_sandboxed_violation=True,
            )
        return ToolExecutionStep(
            step_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            output="Tool call intercepted before execution.",
            success=False,
            is_sandboxed_violation=False,
        )
