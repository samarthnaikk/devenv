import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";

export function TerminalPanel({ transcript, prompt, onPromptChange, onSubmit, isRunning }) {
  const messages = transcript.map((item, index) =>
    React.createElement(
      "article",
      {
        key: `${item.role}-${index}`,
        className: `terminal-bubble ${item.role}`,
      },
      React.createElement("div", { className: "bubble-role" }, item.role === "user" ? "You" : "Devenv"),
      React.createElement("div", {
        className: "bubble-content markdown-body",
        dangerouslySetInnerHTML: { __html: renderMarkdown(item.content) },
      })
    )
  );

  return React.createElement(
    "section",
    { className: "content-panel terminal-panel" },
    React.createElement(
      "div",
      { className: "terminal-header" },
      React.createElement("div", { className: "panel-label" }, "Chat"),
      React.createElement("h2", { className: "terminal-title" }, "Ask Devenv"),
      React.createElement(
        "p",
        { className: "terminal-caption" },
        "Use chat on the right, browse files in the middle, and watch raw runtime output on the left."
      )
    ),
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
        placeholder: "Ask Devenv to inspect the workspace, summarize architecture, or trace a bug...",
        onChange: (event) => onPromptChange(event.target.value),
      }),
      React.createElement(
        "button",
        { className: "terminal-submit", type: "submit", disabled: isRunning || !prompt.trim() },
        isRunning ? "Running..." : "Run Prompt"
      )
    )
  );
}
