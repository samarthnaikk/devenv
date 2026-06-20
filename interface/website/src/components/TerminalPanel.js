import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";
import { PlanRail } from "./PlanRail.js";

export function TerminalPanel({
  transcript,
  prompt,
  blueprint,
  runtimeState,
  stageTraces,
  verificationResults,
  showThinking,
  onPromptChange,
  onSubmit,
  isRunning,
  isCoolingDown,
  cooldownLabel,
  onToggleCollapse,
  collapseLabel,
  collapseGlyph,
}) {
  const messages = transcript.map((item, index) =>
    React.createElement(
      "article",
      {
        key: `${item.role}-${index}`,
        className: `terminal-bubble ${item.role}`,
      },
      React.createElement(
        "div",
        { className: "bubble-role" },
        item.role === "user"
          ? "You"
          : item.role === "thinking"
            ? React.createElement(
                "span",
                { className: "thinking-label" },
                "Thinking",
                item.pending
                  ? React.createElement("span", { className: "typing-dots", "aria-hidden": "true" },
                      React.createElement("span", null),
                      React.createElement("span", null),
                      React.createElement("span", null)
                    )
                  : null
              )
            : item.role === "error"
              ? "Rate Limit"
              : "Devenv"
      ),
      React.createElement("div", {
        className: "bubble-content markdown-body",
        dangerouslySetInnerHTML: {
          __html: renderMarkdown(
            item.role === "thinking" && !showThinking ? summarizeThinkingContent(item.content, item.pending) : item.content
          ),
        },
      })
    )
  );

  return React.createElement(
    "section",
    { className: "content-panel terminal-panel" },
    React.createElement(
      "div",
      { className: "terminal-header" },
      React.createElement(
        "div",
        { className: "terminal-header-copy" },
        React.createElement("div", { className: "panel-label" }, "Chat"),
        React.createElement("h2", { className: "terminal-title" }, "Ask Devenv")
      ),
      onToggleCollapse
        ? React.createElement(
            "button",
            {
              className: "pane-header-button",
              type: "button",
              onClick: onToggleCollapse,
              title: collapseLabel,
              "aria-label": collapseLabel,
            },
            collapseGlyph || ">"
          )
        : null
    ),
    React.createElement(PlanRail, {
      blueprint,
      runtimeState,
      stageTraces,
      verificationResults,
    }),
    React.createElement("div", { className: "terminal-log" }, messages),
    React.createElement(
      "form",
      {
        className: "terminal-form",
        onSubmit: (event) => {
          event.preventDefault();
          onSubmit();
        },
      },
      React.createElement("textarea", {
        className: "terminal-input",
        rows: 4,
        value: prompt,
        placeholder: isCoolingDown
          ? `Groq cooldown active. Input unlocks in ${cooldownLabel}.`
          : "Ask Devenv to inspect the workspace, summarize architecture, or trace a bug...",
        onChange: (event) => onPromptChange(event.target.value),
        disabled: isCoolingDown,
      }),
      React.createElement(
        "button",
        { className: "terminal-submit", type: "submit", disabled: isRunning || isCoolingDown || !prompt.trim() },
        isCoolingDown ? `Cooldown ${cooldownLabel}` : isRunning ? "Running..." : "Run Prompt"
      )
    )
  );
}

function summarizeThinkingContent(content, pending) {
  const text = String(content || "");
  const lowered = text.toLowerCase();
  if (lowered.includes("retrying in")) {
    const retryLine = text
      .split("\n")
      .find((line) => line.toLowerCase().includes("retrying in"));
    return retryLine ? retryLine.replace(/^ERROR\s+/, "") : "Retrying shortly...";
  }
  if (lowered.includes("tool requested")) {
    return pending ? "Calling tools..." : "Tool call completed.";
  }
  if (lowered.includes("planning response") || lowered.includes("state: planning")) {
    return pending ? "Planning next step..." : "Planning completed.";
  }
  return pending ? "Thinking..." : "Completed.";
}
