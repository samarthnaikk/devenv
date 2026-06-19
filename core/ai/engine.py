from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any
from urllib import error, request

from core.tools.base import BaseTool

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are Devenv AI, a local-first coding assistant. "
    "Be precise, use tools when they help, and keep responses grounded in the provided context."
)


class AICore:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        tools: Iterable[BaseTool] | None = None,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> None:
        resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_api_key:
            raise ValueError("Missing Groq API key. Set GROQ_API_KEY or pass api_key explicitly.")

        self.api_key = resolved_api_key
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.system_instructions = system_instructions.strip()
        self._tools: dict[str, BaseTool] = {}

        for tool in tools or ():
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def chat(
        self,
        messages: list[dict[str, Any]],
        memory_context: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._compile_system_frame(memory_context)},
                *messages,
            ],
            "tools": self._build_tool_definitions(),
            "tool_choice": "auto",
            "temperature": temperature,
        }
        return self._post_chat_completion(payload)

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool_name in sorted(self._tools):
            tool = self._tools[tool_name]
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema(),
                    },
                }
            )
        return definitions

    def _compile_system_frame(self, memory_context: str | None) -> str:
        sections = [
            "## System Core Instructions",
            self.system_instructions,
            "## Reconciled Tool Declarations",
            json.dumps(self._build_tool_definitions(), indent=2, sort_keys=True),
        ]

        if memory_context and memory_context.strip():
            sections.extend(
                [
                    "## Cognitive Memory Context",
                    memory_context.strip(),
                ]
            )

        return "\n\n".join(sections)

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq chat completion failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Groq chat completion failed: {exc.reason}") from exc

        try:
            payload_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON response from Groq chat completions endpoint.") from exc

        if not isinstance(payload_data, dict):
            raise ValueError("Malformed response shape from Groq chat completions endpoint.")

        return payload_data
