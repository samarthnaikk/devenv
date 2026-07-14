import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { UserMessage } from "./UserMessage.js";
import { ThinkingMessage } from "./ThinkingMessage.js";
import { AssistantMessage } from "./AssistantMessage.js";
import { ErrorMessage } from "./ErrorMessage.js";
import { PlanFlowchart } from "./PlanFlowchart.js";

const SUGGESTIONS = [
  "Do you remember anything about the old retrieval logic for this project?",
  "What prior Codex session context is relevant to infinite memory here?",
  "Is this a new context or does it match an older Devenv session?",
];

export function Transcript() {
  const { state } = useApp();
  const scrollRef = React.useRef(null);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [state.transcript]);

  if (!state.health || state.bootError) return null;

  if (!state.transcript.length) {
    return React.createElement(
      "div",
      { className: "flex-1 overflow-y-auto p-margin-desktop space-y-8", ref: scrollRef },
      React.createElement(
        "div",
        { className: "flex flex-col items-center justify-center min-h-[60vh] gap-10 px-12" },
        React.createElement(
          "div",
          { className: "flex flex-col items-center gap-4" },
          React.createElement(
            "div",
            { className: "w-12 h-12 rounded-full bg-primary flex items-center justify-center text-on-primary" },
            React.createElement("span", { className: "material-symbols-outlined text-[24px]" }, "neurology")
          ),
          React.createElement("h1", { className: "font-headline-lg text-headline-lg text-on-surface text-center" }, "What should we recall?"),
          React.createElement("div", { className: "max-w-lg text-center font-body-lg text-body-lg text-on-surface-variant" }, "Ask Devenv to search memory, inspect prior sessions, or route turns through OpenCode with explicit consent.")
        ),
        React.createElement(
          "div",
          { className: "grid grid-cols-1 gap-3 w-full max-w-2xl" },
          SUGGESTIONS.map((suggestion) =>
            React.createElement(
              "button",
              {
                key: suggestion,
                type: "button",
                className: "text-left p-4 bg-surface-container border border-outline-variant rounded-lg hover:bg-surface-container-high transition-colors font-body-md text-body-md text-on-surface",
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
    );
  }

  return React.createElement(
    "div",
    { className: "flex-1 overflow-y-auto p-margin-desktop space-y-8", ref: scrollRef },
    state.transcript.map((item) => {
      switch (item.role) {
        case "user":
          return React.createElement(UserMessage, { key: item.id, message: item });
        case "thinking":
          return React.createElement(ThinkingMessage, { key: item.id, message: item });
        case "plan":
          return React.createElement(PlanFlowchart, { key: item.id, blueprint: item.blueprint, mode: item.mode || "auto" });
        case "error":
          return React.createElement(ErrorMessage, { key: item.id, message: item });
        default:
          return React.createElement(AssistantMessage, { key: item.id, message: item });
      }
    })
  );
}
