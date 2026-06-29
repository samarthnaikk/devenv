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
  workspacePath,
  provider,
  model,
  availableModels,
  contextBudget,
  planningMode,
  onPlanningModeChange,
  localOnlyEnabled,
  onLocalOnlyChange,
  showThinking,
  onShowThinkingChange,
  onModelChange,
  onPromptChange,
  onSubmit,
  isRunning,
  isCoolingDown,
  cooldownLabel,
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
        React.createElement("div", { className: "panel-label" }, "Devenv"),
        React.createElement("h2", { className: "terminal-title" }, "Ask Anything From Prior Sessions"),
        React.createElement(
          "p",
          { className: "terminal-caption" },
          workspacePath || "Loading workspace..."
        )
      ),
      React.createElement(
        "div",
        { className: "terminal-controls" },
        React.createElement(
          "label",
          { className: "terminal-select-group" },
          React.createElement("span", { className: "status-label" }, "Planning"),
          React.createElement(
            "select",
            {
              className: "terminal-select",
              value: planningMode || "auto",
              onChange: (event) => onPlanningModeChange?.(event.target.value),
            },
            React.createElement("option", { value: "auto" }, "Auto"),
            React.createElement("option", { value: "force_plan" }, "Plan"),
            React.createElement("option", { value: "force_direct" }, "Direct")
          )
        ),
        React.createElement(
          "label",
          { className: "terminal-select-group" },
          React.createElement("span", { className: "status-label" }, "Model"),
          React.createElement(
            "select",
            {
              className: "terminal-select",
              value: model || "",
              disabled: localOnlyEnabled,
              onChange: (event) => onModelChange?.(event.target.value),
            },
            (availableModels?.length ? availableModels : [model || ""]).map((modelName) =>
              React.createElement("option", { key: modelName, value: modelName }, modelName || "Unknown")
            )
          )
        ),
        React.createElement(
          "label",
          { className: `terminal-toggle${localOnlyEnabled ? " enabled" : ""}` },
          React.createElement("input", {
            type: "checkbox",
            checked: Boolean(localOnlyEnabled),
            onChange: (event) => onLocalOnlyChange?.(event.target.checked),
          }),
          React.createElement("span", null, "Local only")
        ),
        React.createElement(
          "label",
          { className: `terminal-toggle${showThinking ? " enabled" : ""}` },
          React.createElement("input", {
            type: "checkbox",
            checked: Boolean(showThinking),
            onChange: (event) => onShowThinkingChange?.(event.target.checked),
          }),
          React.createElement("span", null, "Show thinking")
        ),
        React.createElement(
          "div",
          { className: "terminal-meta" },
          `${provider || "Unknown"} · ${contextBudget?.remainingLabel || "Unknown"}`
        )
      )
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
          : "Ask about a past project, old review comments, architecture decisions, or repository history...",
        onChange: (event) => onPromptChange(event.target.value),
        disabled: isCoolingDown,
      }),
      React.createElement(
        "button",
        { className: "terminal-submit", type: "submit", disabled: isRunning || isCoolingDown || !prompt.trim() },
        isCoolingDown ? `Cooldown ${cooldownLabel}` : isRunning ? "Thinking..." : "Send"
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
