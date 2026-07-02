import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";

const SUGGESTIONS = [
  "Map this repository and explain how the backend is wired.",
  "Find the bug in the runtime flow and propose a fix.",
  "Create a plan to make this app look exactly like Codex.",
];

const VERBOSITY_OPTIONS = [
  { value: "force_direct", label: "High" },
  { value: "auto", label: "Medium" },
  { value: "force_plan", label: "Max" },
];

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
  const hasMessages = transcript.length > 0;
  const visibleMessages = transcript.filter((item) => item.role !== "thinking" || showThinking);
  const statusLabel = runtimeState || (blueprint?.tasks?.length ? "Working" : "Ready");
  const branchName = blueprint?.target_branch || "main";

  return React.createElement(
    "section",
    { className: `content-panel terminal-panel${hasMessages ? " has-messages" : ""}` },
    React.createElement(
      "div",
      { className: "codex-window-chrome", "aria-hidden": "true" },
      React.createElement(
        "div",
        { className: "mac-controls" },
        React.createElement("span", { className: "mac-dot red" }),
        React.createElement("span", { className: "mac-dot yellow" }),
        React.createElement("span", { className: "mac-dot green" })
      ),
      React.createElement("div", { className: "thread-title" }, hasMessages ? "Current thread" : "New chat"),
      React.createElement(
        "div",
        { className: "top-actions" },
        React.createElement("button", { type: "button", className: "ghost-action" }, "Open"),
        React.createElement("button", { type: "button", className: "ghost-action" }, "Commit")
      )
    ),
    React.createElement(
      "div",
      { className: "terminal-scroll-region" },
      hasMessages
        ? React.createElement(
            "div",
            { className: "chat-thread" },
            visibleMessages.map((item, index) =>
              React.createElement(
                "article",
                {
                  key: `${item.id || item.role}-${index}`,
                  className: `thread-message ${item.role}`,
                },
                React.createElement("div", { className: "thread-message-role" }, roleLabel(item, showThinking)),
                React.createElement("div", {
                  className: "thread-message-body markdown-body",
                  dangerouslySetInnerHTML: {
                    __html: renderMarkdown(
                      item.role === "thinking" && !showThinking ? summarizeThinkingContent(item.content, item.pending) : item.content
                    ),
                  },
                })
              )
            ),
            blueprint?.tasks?.length
              ? React.createElement(
                  "aside",
                  { className: "thread-plan-summary" },
                  React.createElement("div", { className: "thread-plan-heading" }, "Plan"),
                  React.createElement("div", { className: "thread-plan-state" }, statusLabel),
                  React.createElement(
                    "ul",
                    { className: "thread-plan-list" },
                    blueprint.tasks.map((task) =>
                      React.createElement(
                        "li",
                        { key: `${task.task_id}-${task.description}` },
                        `${task.is_completed ? "Done" : "Next"}: ${task.description}`
                      )
                    )
                  ),
                  stageTraces?.length
                    ? React.createElement(
                        "div",
                        { className: "thread-plan-meta" },
                        stageTraces.map((trace) => `${trace.stage}: ${trace.summary}`).join(" · ")
                      )
                    )
                    : null,
                  verificationResults?.length
                    ? React.createElement(
                        "div",
                        { className: "thread-plan-meta" },
                        verificationResults.every((entry) => entry.success) ? "Verification passed" : "Verification pending"
                      )
                    )
                    : null
                )
              : null
          )
        : React.createElement(
            "div",
            { className: "codex-empty-state" },
            React.createElement(
              "div",
              { className: "hero-stack" },
              React.createElement(
                "div",
                { className: "codex-glyph", "aria-hidden": "true" },
                React.createElement("span", null)
              ),
              React.createElement("h1", { className: "hero-title" }, "What should we build?"),
              React.createElement("div", { className: "hero-subtitle" }, workspacePath || "Loading workspace...")
            ),
            React.createElement(
              "div",
              { className: "suggestion-row" },
              SUGGESTIONS.map((suggestion) =>
                React.createElement(
                  "button",
                  {
                    key: suggestion,
                    type: "button",
                    className: "suggestion-card",
                    onClick: () => onPromptChange(suggestion),
                  },
                  suggestion
                )
              )
            )
          )
    ),
    React.createElement(
      "form",
      {
        className: "terminal-form codex-composer",
        onSubmit: (event) => {
          event.preventDefault();
          onSubmit();
        },
      },
      React.createElement(
        "div",
        { className: "composer-shell" },
        React.createElement("textarea", {
          className: "terminal-input composer-input",
          rows: hasMessages ? 3 : 2,
          value: prompt,
          placeholder: isCoolingDown
            ? `Cooldown active. Input unlocks in ${cooldownLabel}.`
            : "Ask Codex anything, @ to use files, / for commands",
          onChange: (event) => onPromptChange(event.target.value),
          disabled: isCoolingDown,
        }),
        React.createElement(
          "div",
          { className: "composer-toolbar" },
          React.createElement(
            "div",
            { className: "composer-toolbar-left" },
            React.createElement(
              "label",
              { className: "toolbar-select compact" },
              React.createElement("span", null, "Permissions"),
              React.createElement(
                "select",
                {
                  value: localOnlyEnabled ? "local" : "default",
                  onChange: (event) => onLocalOnlyChange?.(event.target.value === "local"),
                },
                React.createElement("option", { value: "default" }, "Default permissions"),
                React.createElement("option", { value: "local" }, "Work locally")
              )
            ),
            React.createElement(
              "label",
              { className: "toolbar-select compact" },
              React.createElement("span", null, "Model"),
              React.createElement(
                "select",
                {
                  value: model || "",
                  onChange: (event) => onModelChange?.(event.target.value),
                  disabled: localOnlyEnabled,
                },
                (availableModels?.length ? availableModels : [model || ""]).map((modelName) =>
                  React.createElement("option", { key: modelName, value: modelName }, simplifyModelLabel(modelName))
                )
              )
            ),
            React.createElement(
              "label",
              { className: "toolbar-select compact" },
              React.createElement("span", null, "Reasoning"),
              React.createElement(
                "select",
                {
                  value: planningMode || "auto",
                  onChange: (event) => onPlanningModeChange?.(event.target.value),
                },
                VERBOSITY_OPTIONS.map((option) =>
                  React.createElement("option", { key: option.value, value: option.value }, option.label)
                )
              )
            )
          ),
          React.createElement(
            "div",
            { className: "composer-toolbar-right" },
            React.createElement(
              "label",
              { className: `terminal-toggle inline-toggle${showThinking ? " enabled" : ""}` },
              React.createElement("input", {
                type: "checkbox",
                checked: Boolean(showThinking),
                onChange: (event) => onShowThinkingChange?.(event.target.checked),
              }),
              React.createElement("span", null, "Show thinking")
            ),
            React.createElement("div", { className: "composer-meta" }, `${provider || "Unknown"} · ${contextBudget?.remainingLabel || "Unknown"} · ${branchName}`),
            React.createElement(
              "button",
              { className: "terminal-submit composer-submit", type: "submit", disabled: isRunning || isCoolingDown || !prompt.trim() },
              isCoolingDown ? cooldownLabel : isRunning ? "Working" : "Send"
            )
          )
        )
      )
    )
  );
}

function simplifyModelLabel(modelName) {
  const value = String(modelName || "").trim();
  if (!value) {
    return "Unknown";
  }
  return value.replace(/^.*\//, "").replace(/-/g, " ");
}

function roleLabel(item, showThinking) {
  if (item.role === "user") {
    return "You";
  }
  if (item.role === "thinking") {
    return showThinking ? "Thinking" : "Working";
  }
  if (item.role === "error") {
    return "System";
  }
  return "Codex";
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
  return pending ? "Working..." : "Completed.";
}
