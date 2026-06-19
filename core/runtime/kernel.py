from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.ai import AICore
from core.ai.models import AIResponse, ToolCallRequest
from core.env import load_dotenv
from core.memory import MemoryEngine
from core.tools.base import BaseTool

from .models import RuntimeTurnResult, ToolExecutionStep
from .sandbox import PathSandbox

logger = logging.getLogger(__name__)


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
        load_dotenv(self.workspace_path)
        self.sandbox = PathSandbox(root_path=self.workspace_path)
        self.memory = memory or MemoryEngine(db_path=db_path, vector_dir=vector_dir)
        self.ai = ai or AICore()
        self.tools: dict[str, BaseTool] = {}
        self.ephemeral_history: list[dict[str, Any]] = []

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool
        self.ai.register_tool(tool)
        logger.info("Registered tool with runtime and AI: tool=%s", tool.name)

    def execute_turn(self, user_prompt: str, max_consecutive_tools: int = 5) -> RuntimeTurnResult:
        logger.info("Starting runtime turn: workspace=%s prompt=%s", self.workspace_path, user_prompt)
        ai_logs = [f"Queued prompt: {user_prompt}"]
        system_logs = [f"Workspace: {self.workspace_path}"]
        conversation = list(self.ephemeral_history)
        conversation.append({"role": "user", "content": user_prompt})

        self._record_working_memory(conversation)
        memory_context = self._retrieve_memory_context(user_prompt)
        logger.info("Retrieved memory context: chars=%s", len(memory_context))
        system_logs.append(f"Memory context chars: {len(memory_context)}")
        steps: list[ToolExecutionStep] = []
        total_usage: dict[str, int] = {}

        while True:
            ai_response = self.ai.chat(messages=list(conversation), memory_context=memory_context)
            _merge_usage(total_usage, ai_response.usage)
            ai_logs.append(
                f"AI response: finish_reason={ai_response.finish_reason}, tool_calls={len(ai_response.tool_calls)}, total_tokens={ai_response.usage.get('total_tokens', 0)}"
            )
            logger.info(
                "AI response received: finish_reason=%s tool_calls=%s usage=%s",
                ai_response.finish_reason,
                len(ai_response.tool_calls),
                ai_response.usage,
            )
            if ai_response.tool_calls:
                conversation.append(_assistant_tool_call_message(ai_response))
                for tool_call in ai_response.tool_calls:
                    if len(steps) >= max_consecutive_tools:
                        final_response = "Tool execution limit reached before the request could be completed."
                        logger.warning("Tool limit reached: max_consecutive_tools=%s", max_consecutive_tools)
                        system_logs.append(f"Tool limit reached at {max_consecutive_tools} step(s)")
                        conversation.append({"role": "assistant", "content": final_response})
                        self._finalize_turn(user_prompt, final_response, conversation)
                        return RuntimeTurnResult(
                            final_response=final_response,
                            steps=steps,
                            total_usage=total_usage,
                            ai_logs=ai_logs,
                            system_logs=system_logs,
                        )
                    step = self._execute_tool_call(tool_call)
                    steps.append(step)
                    ai_logs.append(f"Tool requested: {tool_call.tool_name}")
                    system_logs.append(f"Tool step {len(steps)}: {tool_call.tool_name} success={step.success}")
                    conversation.append(_tool_message(tool_call.call_id, tool_call.tool_name, step.output))
                continue

            final_response = ai_response.content
            if final_response is not None:
                conversation.append({"role": "assistant", "content": final_response})
                ai_logs.append("Assistant produced final response")
            logger.info("Finishing runtime turn: final_response_present=%s total_steps=%s", final_response is not None, len(steps))
            self._finalize_turn(user_prompt, final_response or "", conversation)
            system_logs.append("Turn completed and stored in memory")
            return RuntimeTurnResult(
                final_response=final_response,
                steps=steps,
                total_usage=total_usage,
                ai_logs=ai_logs,
                system_logs=system_logs,
            )

    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolExecutionStep:
        logger.info("Intercepted tool call: tool=%s arguments=%s", tool_call.tool_name, tool_call.arguments)
        unsafe_argument = self.sandbox.find_unsafe_argument(tool_call.arguments)
        if unsafe_argument is not None:
            _key, value = unsafe_argument
            logger.warning("Sandbox violation detected: tool=%s path=%s", tool_call.tool_name, value)
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
            logger.error("Requested tool is not registered: tool=%s", tool_call.tool_name)
            return ToolExecutionStep(
                step_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=f"Tool '{tool_call.tool_name}' is not registered in the runtime.",
                success=False,
                is_sandboxed_violation=False,
            )

        normalized_arguments = self.sandbox.normalize_arguments(tool_call.arguments)
        logger.info("Executing tool: tool=%s normalized_arguments=%s", tool_call.tool_name, normalized_arguments)
        result = tool.execute(**normalized_arguments)
        logger.info("Tool finished: tool=%s success=%s", tool_call.tool_name, result.success)
        return ToolExecutionStep(
            step_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            arguments=normalized_arguments,
            output=_format_tool_output(result.output, result.data),
            success=result.success,
            is_sandboxed_violation=False,
        )

    def _finalize_turn(self, user_prompt: str, final_response: str, conversation: list[dict[str, Any]]) -> None:
        self.ephemeral_history = conversation
        self._record_working_memory(self.ephemeral_history)
        logger.info("Recording episodic log for completed turn")
        try:
            self.memory.add_episodic_log(
                user_prompt,
                final_response,
                metadata={"workspace_path": self.workspace_path},
            )
        except Exception as exc:
            logger.warning("Failed to record episodic log; continuing without persisted memory: error=%s", exc)

    def _record_working_memory(self, conversation: list[dict[str, Any]]) -> None:
        try:
            self.memory.record_working_memory(
                messages=conversation[-10:],
                active_state={"workspace_path": self.workspace_path},
            )
        except Exception as exc:
            logger.warning("Failed to record working memory; continuing: error=%s", exc)

    def _retrieve_memory_context(self, user_prompt: str) -> str:
        try:
            return self.memory.retrieve_context(user_prompt).markdown_context
        except Exception as exc:
            logger.warning("Memory retrieval failed; continuing without memory context: error=%s", exc)
            return ""


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


def _format_tool_output(output: str, data: dict[str, Any]) -> str:
    if not data:
        return output
    return f"{output}\n{json.dumps(data, sort_keys=True)}"


def _merge_usage(total_usage: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total_usage[key] = total_usage.get(key, 0) + value
