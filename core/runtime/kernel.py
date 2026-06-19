from __future__ import annotations

import json
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
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}

        while True:
            ai_response = self.ai.chat(messages=list(conversation), memory_context=memory_result.markdown_context)
            total_usage = dict(ai_response.usage)
            if ai_response.tool_calls:
                conversation.append(_assistant_tool_call_message(ai_response))
                for tool_call in ai_response.tool_calls:
                    step = self._execute_tool_call(tool_call)
                    steps.append(step)
                    conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                continue

            final_response = ai_response.content
            if final_response is not None:
                conversation.append({"role": "assistant", "content": final_response})
            self._finalize_turn(user_prompt, final_response or "", conversation)
            return RuntimeTurnResult(final_response=final_response, steps=steps, total_usage=total_usage)

    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolExecutionStep:
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

        tool = self.tools.get(tool_call.tool_name)
        if tool is None:
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=f"Tool '{tool_call.tool_name}' is not registered in the runtime.",
                success=False,
                is_sandboxed_violation=False,
            )

        normalized_arguments = self.sandbox.normalize_arguments(tool_call.arguments)
        result = tool.execute(**normalized_arguments)
        return ToolExecutionStep(
            step_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            arguments=normalized_arguments,
            output=result.output,
            success=result.success,
            is_sandboxed_violation=False,
        )

    def _finalize_turn(self, user_prompt: str, final_response: str, conversation: list[dict[str, Any]]) -> None:
        self.ephemeral_history = conversation
        self.memory.record_working_memory(
            messages=self.ephemeral_history[-10:],
            active_state={"workspace_path": self.workspace_path},
        )
        self.memory.add_episodic_log(
            user_prompt,
            final_response,
            metadata={"workspace_path": self.workspace_path},
        )


def _assistant_tool_call_message(ai_response: AIResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": ai_response.content,
        "tool_calls": [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in ai_response.tool_calls
        ],
    }


def _tool_message(call_id: str, tool_name: str, output: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": output,
    }
