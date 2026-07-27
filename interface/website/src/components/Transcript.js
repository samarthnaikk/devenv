import React from "react";
import { useApp } from "../context/AppContext.js";
import { UserMessage } from "./UserMessage.js";
import { ThinkingMessage } from "./ThinkingMessage.js?v=trace2";
import { AssistantMessage } from "./AssistantMessage.js";
import { ErrorMessage } from "./ErrorMessage.js";
import { PlanFlowchart } from "./PlanFlowchart.js?v=flow4";
import { showToast } from "./Header.js";
import { BeamFrame, MotionBadge, MotionDeck, MotionNumber, MotionReveal, MotionShimmerText, MotionStack, MotionTilt } from "./MotionPrimitives.js";

const PLAYBOOKS = [
  {
    label: "Codebase",
    icon: "folder_code",
    title: "Ask about this codebase",
    copy: "Get grounded answers from the current repository.",
    suggestion: "Explain this codebase and the part I should start with.",
    selectedTools: ["list_directory", "search_text", "inspect_symbols"],
    planMode: true,
  },
  {
    label: "Debug",
    icon: "search",
    title: "Find a bug or issue",
    copy: "Trace files and symbols to isolate a problem quickly.",
    suggestion: "Find the bug causing the current issue in this app.",
    selectedTools: ["locate_files", "read_file", "search_text", "track_symbol"],
    planMode: false,
  },
  {
    label: "Research",
    icon: "language",
    title: "Look something up",
    copy: "Use live sources when the answer depends on current information.",
    suggestion: "Look up the latest information relevant to this task.",
    selectedTools: ["web_search", "knowledge_search"],
    planMode: false,
  },
];

export function Transcript() {
  const { state, dispatch } = useApp();
  const scrollRef = React.useRef(null);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [state.transcript]);

  const handleCopyMessage = React.useCallback(async (message) => {
    const text = String(message?.content || "").trim();
    if (!text) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "true");
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      showToast(dispatch, "Message copied");
    } catch {
      showToast(dispatch, "Copy failed");
    }
  }, [dispatch]);

  const handleReplyMessage = React.useCallback((message) => {
    if (!message) return;
    dispatch({
      type: "SET_REPLY_TARGET",
      payload: {
        id: message.id,
        role: message.role,
        author: message.role === "user" ? "You" : message.role === "error" ? "Error" : "Devenv",
        excerpt: buildReplyExcerpt(message.content),
      },
    });
    showToast(dispatch, "Reply attached");
  }, [dispatch]);

  if (!state.health || state.bootError) return null;

  if (!state.transcript.length) {
    return React.createElement(
      "div",
      { className: "transcript-scroll transcript-scroll-empty p-margin-desktop", ref: scrollRef },
      React.createElement(
        "div",
        { className: "empty-state-shell empty-state-shell-compact flex flex-col items-start justify-center gap-4 px-4" },
        React.createElement(
          BeamFrame,
          { active: false, tone: "ocean", className: "empty-state-hero empty-state-hero-compact rounded-3xl p-6 w-full" },
          React.createElement(
            "div",
            { className: "empty-state-hero-stack flex flex-col items-start gap-4" },
            React.createElement(
              "div",
              { className: "empty-state-badge w-14 h-14 rounded-full flex items-center justify-center text-on-primary" },
              React.createElement("span", { className: "material-symbols-outlined text-[26px]" }, "neurology")
            ),
            React.createElement(
              "h1",
              { className: "font-headline-lg text-headline-lg text-on-surface empty-state-title empty-state-title-compact" },
              React.createElement(MotionShimmerText, { className: "empty-state-title-line" }, "What do you want to do?")
            ),
            React.createElement("div", { className: "empty-state-copy max-w-2xl font-body-lg text-body-lg text-on-surface-variant" }, "Ask a question, debug something, or search for current information.")
          ),
          React.createElement(
            MotionDeck,
            { className: "empty-state-command-deck w-full mt-4" },
            PLAYBOOKS.map((playbook, index) =>
              React.createElement(
                MotionReveal,
                { key: playbook.label, delay: index * 80 },
                React.createElement(
                  MotionTilt,
                  null,
                  React.createElement(
                    "button",
                    {
                      type: "button",
                      className: "empty-state-command-card text-left",
                      onClick: () => {
                        const event = new CustomEvent("opencode-suggestion", { detail: playbook });
                        window.dispatchEvent(event);
                      },
                    },
                    React.createElement("span", { className: "empty-state-command-kicker" }, playbook.label),
                    React.createElement(
                      "div",
                      { className: "empty-state-command-head" },
                      React.createElement("span", { className: "material-symbols-outlined text-[19px] text-primary" }, playbook.icon),
                      React.createElement("strong", null, playbook.title)
                    ),
                    React.createElement("p", { className: "empty-state-command-copy" }, playbook.copy),
                    React.createElement("div", { className: "empty-state-command-launch-row" }, React.createElement("span", { className: "empty-state-command-launch" }, "Use this"))
                  )
                )
              )
            )
          )
        )
      )
    );
  }

  return React.createElement(
    "div",
    { className: "transcript-scroll p-margin-desktop space-y-8", ref: scrollRef },
    React.createElement(
      "div",
      { className: "transcript-runway" },
      React.createElement("div", { className: "transcript-runway-grid", "aria-hidden": "true" }),
      React.createElement("div", { className: "transcript-runway-orbit transcript-runway-orbit-one", "aria-hidden": "true" }),
      React.createElement("div", { className: "transcript-runway-orbit transcript-runway-orbit-two", "aria-hidden": "true" }),
      React.createElement(
        BeamFrame,
        { active: state.isRunning, tone: "mono", className: "transcript-runway-rail rounded-[24px] px-4 py-3" },
        React.createElement(
          "div",
          { className: "transcript-runway-head" },
          React.createElement(
            "div",
            { className: "transcript-runway-copy" },
            React.createElement(MotionBadge, { className: "transcript-runway-badge", active: state.isRunning }, state.isRunning ? "Live trace" : "Transcript"),
            React.createElement(
              "div",
              { className: "transcript-runway-text" },
              React.createElement("strong", null, "Conversation runway"),
              React.createElement(
                "span",
                null,
                state.isRunning
                  ? "Reasoning, tools, and responses are landing in a staged stream."
                  : "The shell keeps every turn legible while the runtime swaps between plans, tools, and direct answers."
              )
            )
          ),
          React.createElement(
            "div",
            { className: "transcript-runway-stats" },
            React.createElement(
              "div",
              { className: "transcript-runway-stat" },
              React.createElement("span", null, "Entries"),
              React.createElement(MotionNumber, { value: state.transcript.length })
            ),
            React.createElement(
              "div",
              { className: "transcript-runway-stat" },
              React.createElement("span", null, "Mode"),
              React.createElement("strong", null, state.planMode ? "Plan" : state.isRunning ? "Live" : "Direct")
            )
          )
        )
      ),
      React.createElement(
        "div",
        { className: "transcript-stream" },
        state.transcript.map((item, index) =>
          React.createElement(
            MotionReveal,
            { key: item.id, delay: Math.min(index * 55, 330), className: "transcript-entry-reveal" },
            React.createElement(
              "div",
              { className: `transcript-entry-shell transcript-entry-${item.role || "assistant"}` },
              renderTranscriptItem(item, handleCopyMessage, handleReplyMessage)
            )
          )
        )
      )
    )
  );
}

function renderTranscriptItem(item, handleCopyMessage, handleReplyMessage) {
  switch (item.role) {
    case "user":
      return React.createElement(UserMessage, {
        message: item,
        onCopy: () => handleCopyMessage(item),
        onReply: () => handleReplyMessage(item),
      });
    case "thinking":
      return React.createElement(ThinkingMessage, { message: item });
    case "plan":
      return React.createElement(PlanFlowchart, { blueprint: item.blueprint, mode: item.mode || "auto" });
    case "error":
      return React.createElement(ErrorMessage, {
        message: item,
        onCopy: () => handleCopyMessage(item),
        onReply: () => handleReplyMessage(item),
      });
    default:
      return React.createElement(AssistantMessage, {
        message: item,
        onCopy: () => handleCopyMessage(item),
        onReply: () => handleReplyMessage(item),
      });
  }
}

function buildReplyExcerpt(content) {
  const normalized = String(content || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "Empty message";
  return normalized.length > 140 ? `${normalized.slice(0, 137)}...` : normalized;
}
