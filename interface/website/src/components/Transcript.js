import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { UserMessage } from "./UserMessage.js";
import { ThinkingMessage } from "./ThinkingMessage.js?v=trace2";
import { AssistantMessage } from "./AssistantMessage.js";
import { ErrorMessage } from "./ErrorMessage.js";
import { PlanFlowchart } from "./PlanFlowchart.js?v=flow4";
import { showToast } from "./Header.js";
import { BeamFrame, MotionDeck, MotionReveal, MotionShimmerText, MotionStack } from "./MotionPrimitives.js";

const SUGGESTIONS = [
  "Do you remember anything about the old retrieval logic for this project?",
  "What prior Codex session context is relevant to infinite memory here?",
  "Is this a new context or does it match an older Devenv session?",
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
      { className: "flex-1 overflow-y-auto p-margin-desktop space-y-8", ref: scrollRef },
      React.createElement(
        "div",
        { className: "empty-state-shell flex flex-col items-center justify-center min-h-[60vh] gap-10 px-12" },
        React.createElement(
          BeamFrame,
          { active: false, tone: "ocean", className: "empty-state-hero rounded-3xl p-8 max-w-3xl w-full" },
          React.createElement(
            "div",
            { className: "flex flex-col items-center gap-4" },
            React.createElement(
              "div",
              { className: "empty-state-badge w-14 h-14 rounded-full flex items-center justify-center text-on-primary" },
              React.createElement("span", { className: "material-symbols-outlined text-[26px]" }, "neurology")
            ),
            React.createElement(
              "h1",
              { className: "font-headline-lg text-headline-lg text-on-surface text-center empty-state-title" },
              React.createElement(MotionShimmerText, { className: "empty-state-title-line" }, "Inspect faster."),
              React.createElement("span", { className: "empty-state-title-line" }, "Plan cleaner."),
              React.createElement("span", { className: "empty-state-title-line" }, "Ship with motion.")
            ),
            React.createElement("div", { className: "max-w-2xl text-center font-body-lg text-body-lg text-on-surface-variant" }, "Ask Devenv to inspect the codebase, route into a plan, or search live sources. The interface stays light, tactile, and traceable while the runtime decides what to use.")
          )
        ),
        React.createElement(
          MotionStack,
          { className: "empty-state-preview-grid w-full max-w-3xl" },
          [
            { label: "Plan", icon: "conversion_path", title: "Multi-node flow", copy: "Blueprints render as connected steps instead of a one-line shrug." },
            { label: "Trace", icon: "network_intelligence", title: "Visible reasoning surface", copy: "Thinking, tools, and retrieval cues stay legible while a turn is running." },
            { label: "Motion", icon: "animation", title: "Light, tactile shell", copy: "Beams, metal shimmer, stacked cards, and staged panel transitions unify the interface." },
          ].map((card, index) =>
            React.createElement(
              MotionReveal,
              { key: card.label, delay: index * 90 },
              React.createElement(
                "div",
                { className: "empty-state-preview-card" },
                React.createElement("span", { className: "empty-state-preview-kicker" }, card.label),
                React.createElement(
                  "div",
                  { className: "empty-state-preview-head" },
                  React.createElement("span", { className: "material-symbols-outlined text-[20px] text-primary" }, card.icon),
                  React.createElement("strong", null, card.title)
                ),
                React.createElement("p", { className: "empty-state-preview-copy" }, card.copy)
              )
            )
          )
        ),
        React.createElement(
          MotionDeck,
          { className: "empty-state-suggestions grid grid-cols-1 gap-3 w-full max-w-2xl" },
          SUGGESTIONS.map((suggestion) =>
            React.createElement(
              MotionReveal,
              { key: suggestion, delay: SUGGESTIONS.indexOf(suggestion) * 70 },
              React.createElement(
                "button",
                {
                  type: "button",
                  className: "empty-state-card text-left p-4 bg-surface-container border border-outline-variant rounded-2xl font-body-md text-body-md text-on-surface",
                  onClick: () => {
                    const event = new CustomEvent("opencode-suggestion", { detail: { suggestion } });
                    window.dispatchEvent(event);
                  },
                },
                suggestion
              )
            )
          )
        )
      )
    );
  }

  return React.createElement(
    "div",
    { className: "flex-1 overflow-y-auto p-margin-desktop space-y-8", ref: scrollRef },
    state.transcript.map((item) => {
      switch (item.role) {
        case "user":
          return React.createElement(UserMessage, {
            key: item.id,
            message: item,
            onCopy: () => handleCopyMessage(item),
            onReply: () => handleReplyMessage(item),
          });
        case "thinking":
          return React.createElement(ThinkingMessage, { key: item.id, message: item });
        case "plan":
          return React.createElement(PlanFlowchart, { key: item.id, blueprint: item.blueprint, mode: item.mode || "auto" });
        case "error":
          return React.createElement(ErrorMessage, {
            key: item.id,
            message: item,
            onCopy: () => handleCopyMessage(item),
            onReply: () => handleReplyMessage(item),
          });
        default:
          return React.createElement(AssistantMessage, {
            key: item.id,
            message: item,
            onCopy: () => handleCopyMessage(item),
            onReply: () => handleReplyMessage(item),
          });
      }
    })
  );
}

function buildReplyExcerpt(content) {
  const normalized = String(content || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "Empty message";
  return normalized.length > 140 ? `${normalized.slice(0, 137)}...` : normalized;
}
