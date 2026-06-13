from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from .models import WorkingMemoryMessage, WorkingMemorySnapshot


class WorkingMemoryManager:
    def __init__(self, max_messages: int = 15) -> None:
        self.max_messages = max_messages
        self._messages: deque[WorkingMemoryMessage] = deque(maxlen=max_messages)
        self._active_state: dict[str, Any] = {}

    def record(self, messages: list[dict[str, Any]], active_state: dict[str, Any]) -> None:
        self._messages.clear()
        for message in messages[-self.max_messages :]:
            self._messages.append(
                WorkingMemoryMessage(
                    role=str(message["role"]),
                    content=str(message["content"]),
                    timestamp=message.get("timestamp"),
                )
            )
        self._active_state = dict(active_state)

    def snapshot(self) -> WorkingMemorySnapshot:
        return WorkingMemorySnapshot(messages=tuple(self._messages), active_state=dict(self._active_state))

    def as_prompt_block(self) -> str:
        snapshot = self.snapshot()
        lines = ["## Working Memory"]
        if snapshot.active_state:
            state_pairs = ", ".join(f"{key}={value}" for key, value in sorted(snapshot.active_state.items()))
            lines.append(f"Active state: {state_pairs}")
        for message in snapshot.messages:
            lines.append(f"- {message.role}: {message.content}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "messages": [asdict(message) for message in snapshot.messages],
            "active_state": snapshot.active_state,
        }
