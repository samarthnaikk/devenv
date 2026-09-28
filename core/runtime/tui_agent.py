"""Textual screens for connecting to native AI agents over ACP.

This module is imported lazily by the TUI so the rest of the runtime keeps
working even when the optional agent stack is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
)

from core.ai.acp_agent import ACPAgentError, ACPAgentSession
from core.ai.agents import AgentAvailability

from .tui_theme import BLUE, ERROR as ERROR_COLOR, TEAL, TEXT, TEXT_MUTED, WARN
from .tui_widgets import Choice, DiffView, SelectionScreen, ToolTrace

try:  # pragma: no cover - rich ships with textual
    from rich.markup import escape as _rich_escape
except Exception:  # pragma: no cover

    def _rich_escape(text: str) -> str:
        return text


LOGGER = logging.getLogger(__name__)

_AGENT_CSS = f"""
#agent-header {{
    height: auto;
    background: {TEXT};
    color: #0d0f12;
    padding: 0 1;
    text-style: bold;
}}

#agent-transcript {{
    height: 1fr;
    padding: 0 1;
    background: #0d0f12;
}}

.agent-bubble {{
    height: auto;
    margin: 0 0 1 0;
    padding: 0 1;
}}

.agent-bubble-user {{
    background: #1c2026;
    color: {BLUE};
    border-left: thick {BLUE};
}}

.agent-bubble-assistant {{
    background: #16191e;
    color: {TEXT};
    border-left: thick {TEAL};
}}

.agent-bubble-thought {{
    color: {TEXT_MUTED};
    border-left: thick #2d333b;
}}

.agent-tool {{
    background: #16191e;
    color: {WARN};
    border-left: thick {WARN};
    margin: 0 0 1 0;
    padding: 0 1;
}}

.agent-system {{
    color: {TEXT_MUTED};
    height: auto;
}}

#agent-composer {{
    background: #0a0c0e;
    border: round #2d333b;
    color: {TEXT};
}}

#agent-composer:focus {{
    border: round {TEAL};
}}
"""


class AgentScreen(Screen[None]):
    """Live ACP conversation with a native agent such as OpenCode."""

    CSS = _AGENT_CSS

    BINDINGS = [
        ("escape", "cancel_turn", "Cancel"),
        ("ctrl+q", "close_agent", "Close agent"),
    ]

    def __init__(self, session: ACPAgentSession) -> None:
        super().__init__()
        self.session = session
        self._consumer: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._live_parts: list[str] = []
        self._live_widget: Static | None = None
        self._thought_parts: list[str] = []
        self._thought_widget: Static | None = None
        self._tool_widgets: dict[str, Any] = {}
        self._busy = False
        self._connected = False
        self._placeholder: Static | None = None

    # ------------------------------------------------------------------ compose
    def compose(self) -> ComposeResult:
        yield Static(self._header_text("connecting…"), id="agent-header")
        with VerticalScroll(id="agent-transcript"):
            self._placeholder = Static(
                f"[{TEXT_MUTED}]Connecting to {_rich_escape(self.session.spec.title)}…[/]",
                id="agent-placeholder",
                markup=True,
            )
            yield self._placeholder
        yield Input(placeholder="Ask the agent…", id="agent-composer")
        yield Footer()

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.session.permission_handler = self._request_permission
        self.query_one("#agent-composer", Input).focus()
        self.run_worker(self._start_session(), name="agent-start", exclusive=True)

    def on_unmount(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        self._schedule_close()

    def _schedule_close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop during teardown
            return
        self._close_task = loop.create_task(self.session.close())

    async def _start_session(self) -> None:
        try:
            info = await self.session.start()
        except ACPAgentError as exc:
            if self.is_mounted:
                self._append_system(f"failed to connect: {exc}", error=True)
                self._update_header("connection failed")
            return
        if not self.is_mounted:
            return
        self._connected = True
        version = f" {info.agent_version}" if info.agent_version else ""
        self._update_header(f"{info.agent_title}{version}")
        self._append_system(f"connected · session {info.session_id}")
        self._consumer = asyncio.create_task(self._consume_events())

    async def _consume_events(self) -> None:
        while True:
            try:
                update = await self.session.events.get()
            except asyncio.CancelledError:
                return
            if not self.is_mounted:
                return
            try:
                self._handle_update(update)
            except Exception:  # pragma: no cover - rendering must not kill the stream
                LOGGER.exception("Failed to render ACP update")

    # ------------------------------------------------------------------ input
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "agent-composer":
            return
        value = event.value.strip()
        event.input.value = ""
        if not value or self._busy:
            return
        if value.startswith("/") and value.rstrip() in {"/exit", "/quit"}:
            self.action_close_agent()
            return
        self.run_worker(self._send_prompt(value), name="agent-prompt")

    async def _send_prompt(self, text: str) -> None:
        self._busy = True
        self._set_composer_enabled(False)
        self._append_user(text)
        self._begin_live_message()
        try:
            stop_reason = await self.session.send_prompt(text)
        except ACPAgentError as exc:
            self._append_system(str(exc), error=True)
        else:
            self._append_system(f"turn complete · {stop_reason or 'end_turn'}")
        finally:
            self._busy = False
            self._live_widget = None
            self._live_parts = []
            self._set_composer_enabled(True)
            self._focus_composer()

    # ------------------------------------------------------------------ actions
    def action_cancel_turn(self) -> None:
        if self._busy:
            self._append_system("cancelling…")
            self.run_worker(self.session.cancel(), name="agent-cancel")
            return
        self.action_close_agent()

    def action_close_agent(self) -> None:
        self._schedule_close()
        self.app.pop_screen()

    async def _request_permission(
        self,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> RequestPermissionResponse:
        if not options:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        title = getattr(tool_call, "title", None) or "Tool call"
        kind = getattr(tool_call, "kind", None) or "tool"
        choices = [
            Choice(
                value=option.option_id,
                label=option.name or option.option_id,
                detail=str(getattr(option, "kind", "") or ""),
            )
            for option in options
        ]
        choice = await self.app.push_screen_wait(
            SelectionScreen(
                choices,
                title="Permission required",
                hint=f"{title} ({kind})",
            )
        )
        if choice is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=choice, outcome="selected")
        )

    # ------------------------------------------------------------------ rendering
    def _handle_update(self, update: Any) -> None:
        if isinstance(update, AgentMessageChunk):
            self._thought_widget = None
            self._append_live(self._content_text(update.content))
        elif isinstance(update, AgentThoughtChunk):
            self._append_thought(self._content_text(update.content))
        elif isinstance(update, ToolCallStart):
            self._handle_tool_start(update)
        elif isinstance(update, ToolCallProgress):
            self._handle_tool_progress(update)
        elif isinstance(update, AgentPlanUpdate):
            self._handle_plan(update)
        elif isinstance(update, AvailableCommandsUpdate):
            self._update_header(self._header_text(f"{len(update.available_commands)} commands"))
        elif isinstance(update, CurrentModeUpdate):
            self._append_system(f"mode · {update.current_mode_id}")
        elif isinstance(update, UsageUpdate):
            self._append_system(f"context · {update.used}/{update.size} tokens")

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, TextContentBlock):
            return content.text
        if isinstance(content, dict) and content.get("type") == "text":
            return str(content.get("text") or "")
        return ""

    def _handle_tool_start(self, update: ToolCallStart) -> None:
        title = update.title or update.tool_call_id
        kind = update.kind or ""
        label = f"tool · {title}" + (f" ({kind})" if kind else "")
        widget = ToolTrace(label, collapsed=True)
        self._tool_widgets[update.tool_call_id] = widget
        self._mount(widget)

    def _handle_tool_progress(self, update: ToolCallProgress) -> None:
        widget = self._tool_widgets.get(update.tool_call_id)
        if widget is None:
            return
        title = update.title or update.tool_call_id
        status = update.status or "in_progress"
        widget.set_status(f"tool · {title} · {status}", self._tool_content_text(update))
        self._mount_tool_diffs(update)
        self._scroll_end()

    @staticmethod
    def _tool_content_text(update: Any) -> str:
        parts: list[str] = []
        for item in getattr(update, "content", None) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def _mount_tool_diffs(self, update: Any) -> None:
        for item in getattr(update, "content", None) or []:
            path = getattr(item, "path", None)
            old_text = getattr(item, "old_text", None)
            new_text = getattr(item, "new_text", None)
            if path is not None and (old_text is not None or new_text is not None):
                self._mount(DiffView(str(path), old_text, new_text))

    def _handle_plan(self, update: AgentPlanUpdate) -> None:
        lines = [f"[b {BLUE}]plan[/]"]
        icons = {"completed": "◆", "in_progress": "▶", "pending": "◇"}
        for entry in update.entries:
            icon = icons.get(str(entry.status), "◇")
            lines.append(
                f"  [{TEXT_MUTED}]{icon}[/] [{TEXT}]{_rich_escape(entry.content)}[/]"
            )
        self._mount(Static("\n".join(lines), classes="agent-system", markup=True))

    def _append_user(self, text: str) -> None:
        self._mount(
            Static(
                f"[b {BLUE}]you[/]\n{_rich_escape(text)}",
                classes="agent-bubble agent-bubble-user",
                markup=True,
            )
        )

    def _begin_live_message(self) -> None:
        self._live_parts = []
        self._live_widget = Static("", classes="agent-bubble agent-bubble-assistant", markup=True)
        self._thought_parts = []
        self._thought_widget = None
        self._mount(self._live_widget)

    def _append_live(self, text: str) -> None:
        if not text or not self.is_mounted:
            return
        if self._live_widget is None:
            self._begin_live_message()
        self._live_parts.append(text)
        assert self._live_widget is not None
        self._live_widget.update(_rich_escape("".join(self._live_parts)))
        self._scroll_end()

    def _append_thought(self, text: str) -> None:
        if not text or not self.is_mounted:
            return
        if self._thought_widget is None:
            self._thought_parts = []
            self._thought_widget = Static(
                f"[{TEXT_MUTED}]thinking[/]",
                classes="agent-bubble agent-bubble-thought",
                markup=True,
            )
            self._mount(self._thought_widget)
        self._thought_parts.append(text)
        assert self._thought_widget is not None
        self._thought_widget.update(
            f"[{TEXT_MUTED}]thinking[/]\n{_rich_escape(''.join(self._thought_parts))}"
        )
        self._scroll_end()

    def _append_system(self, text: str, *, error: bool = False) -> None:
        color = ERROR_COLOR if error else TEXT_MUTED
        self._mount(
            Static(
                f"[{color}]{_rich_escape(text)}[/]",
                classes="agent-system",
                markup=True,
            )
        )

    def _mount(self, widget: Any) -> None:
        if not self.is_mounted:
            return
        if self._placeholder is not None and self._placeholder.is_mounted:
            self._placeholder.remove()
            self._placeholder = None
        container = self.query_one("#agent-transcript", VerticalScroll)
        container.mount(widget)
        self._scroll_end()

    def _scroll_end(self) -> None:
        try:
            self.query_one("#agent-transcript", VerticalScroll).scroll_end(animate=False)
        except Exception:  # pragma: no cover - widget may be gone during teardown
            pass

    def _set_composer_enabled(self, enabled: bool) -> None:
        try:
            self.query_one("#agent-composer", Input).disabled = not enabled
        except Exception:  # pragma: no cover
            pass

    def _focus_composer(self) -> None:
        try:
            self.query_one("#agent-composer", Input).focus()
        except Exception:  # pragma: no cover
            pass

    def _header_text(self, status: str) -> str:
        title = _rich_escape(self.session.spec.title)
        return f"{title}  ·  {_rich_escape(status)}"

    def _update_header(self, status: str) -> None:
        try:
            self.query_one("#agent-header", Static).update(self._header_text(status))
        except Exception:  # pragma: no cover
            pass


__all__ = ["AgentScreen"]
