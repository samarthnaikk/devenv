from __future__ import annotations

import asyncio
import importlib.util
import unittest

from acp.schema import AgentMessageChunk, AvailableCommandsUpdate

from core.ai.acp_agent import AgentSessionInfo
from core.ai.agents import AgentSpec

_TEXTUAL_AVAILABLE = importlib.util.find_spec("textual") is not None
if _TEXTUAL_AVAILABLE:
    from textual.app import App

    from core.runtime.tui_agent import AgentScreen
else:  # pragma: no cover - exercised only without textual
    App = object  # type: ignore[assignment,misc]
    AgentScreen = object  # type: ignore[assignment,misc]


class FakeAgentSession:
    def __init__(self) -> None:
        self.spec = AgentSpec(name="fake", title="Fake Agent", command="fake", args=("acp",))
        self.events: asyncio.Queue = asyncio.Queue()
        self.permission_handler = None
        self.started = False
        self.closed = False
        self.prompts: list[str] = []

    async def start(self) -> AgentSessionInfo:
        self.started = True
        return AgentSessionInfo(
            agent_name="fake",
            agent_title="Fake Agent",
            agent_version="0.0.1",
            session_id="sess-test",
        )

    async def send_prompt(self, text: str) -> str:
        self.prompts.append(text)
        return "end_turn"

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


if _TEXTUAL_AVAILABLE:

    class _Harness(App[None]):
        def __init__(self, screen: AgentScreen) -> None:
            super().__init__()
            self._screen = screen

        def on_mount(self) -> None:
            self.push_screen(self._screen)

else:  # pragma: no cover - exercised only without textual
    _Harness = object  # type: ignore[assignment,misc]


def _message_chunk(text: str) -> AgentMessageChunk:
    return AgentMessageChunk.model_validate(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
    )


@unittest.skipIf(not _TEXTUAL_AVAILABLE, "textual is not installed")
class AgentScreenTest(unittest.IsolatedAsyncioTestCase):
    async def test_screen_connects_and_renders_streamed_updates(self) -> None:
        session = FakeAgentSession()
        screen = AgentScreen(session)  # type: ignore[arg-type]
        app = _Harness(screen)

        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertTrue(session.started)

            session.events.put_nowait(_message_chunk("hello stream"))
            await pilot.pause()

            bubbles = screen.query(".agent-bubble-assistant")
            self.assertGreater(len(bubbles), 0)

    async def test_screen_handles_commands_update_without_breaking(self) -> None:
        session = FakeAgentSession()
        screen = AgentScreen(session)  # type: ignore[arg-type]
        app = _Harness(screen)

        async with app.run_test() as pilot:
            await pilot.pause()
            session.events.put_nowait(
                AvailableCommandsUpdate.model_validate(
                    {
                        "sessionUpdate": "available_commands_update",
                        "availableCommands": [
                            {"name": "init", "description": "Initialize"}
                        ],
                    }
                )
            )
            await pilot.pause()

        self.assertTrue(session.started)


if __name__ == "__main__":
    unittest.main()
