import React from "https://esm.sh/react@18";

export function HeaderBar({
  workspacePath,
  provider,
  model,
  usage,
  contextBudget,
  planModeEnabled,
  onPlanModeChange,
  showThinking,
  onShowThinkingChange,
}) {
  const chips = [
    { label: "Plan Mode", value: planModeEnabled ? "On" : "Off", isToggle: true },
    { label: "Show Thinking", value: showThinking ? "On" : "Off", isThinkingToggle: true },
    { label: "Provider", value: provider || "Unknown" },
    { label: "Model", value: model || "Unknown" },
    { label: "Tokens", value: String(usage?.total_tokens || 0) },
    { label: "Context Left", value: contextBudget?.remainingLabel || "Unknown" },
    { label: "Resets", value: contextBudget?.resetLabel || "Idle" },
  ];

  return React.createElement(
    "header",
    { className: "header-bar" },
    React.createElement(
      "div",
      { className: "header-copy" },
      React.createElement("p", { className: "eyebrow" }, "Devenv Runtime"),
      React.createElement("h1", null, "Web Terminal"),
      React.createElement("span", { className: "workspace-path" }, workspacePath || "Loading workspace...")
    ),
    React.createElement(
      "div",
      { className: "status-strip" },
      contextBudget?.isLow
        ? React.createElement(
            "div",
            { className: "status-warning" },
            `Warning: less than 10% context remains (${contextBudget.remainingLabel}).`
          )
        : null,
      chips.map((chip) =>
        chip.isToggle || chip.isThinkingToggle
          ? React.createElement(
              "label",
              {
                key: chip.label,
                className: `status-pill status-pill-toggle${
                  (chip.isToggle ? planModeEnabled : showThinking) ? " enabled" : ""
                }`,
              },
              React.createElement("span", { className: "status-label" }, chip.label),
              React.createElement(
                "span",
                { className: "status-toggle-value" },
                React.createElement("input", {
                  type: "checkbox",
                  checked: Boolean(chip.isToggle ? planModeEnabled : showThinking),
                  onChange: (event) =>
                    chip.isToggle
                      ? onPlanModeChange?.(event.target.checked)
                      : onShowThinkingChange?.(event.target.checked),
                }),
                React.createElement("span", { className: "status-value" }, chip.value)
              )
            )
          : React.createElement(
              "div",
              { key: chip.label, className: "status-pill" },
              React.createElement("span", { className: "status-label" }, chip.label),
              React.createElement("span", { className: "status-value" }, chip.value)
            )
      )
    )
  );
}
