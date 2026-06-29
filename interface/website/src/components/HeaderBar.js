import React from "https://esm.sh/react@18";

export function HeaderBar({
  workspacePath,
  provider,
  model,
  availableModels,
  usage,
  contextBudget,
  planningMode,
  onPlanningModeChange,
  localOnlyEnabled,
  onLocalOnlyChange,
  showThinking,
  onShowThinkingChange,
  onModelChange,
}) {
  const chips = [
    { label: "Local", value: localOnlyEnabled ? "On" : "Off", isLocalToggle: true },
    { label: "Show Thinking", value: showThinking ? "On" : "Off", isThinkingToggle: true },
    { label: "Provider", value: provider || "Unknown" },
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
      React.createElement(
        "label",
        { className: "status-pill status-pill-select" },
        React.createElement("span", { className: "status-label" }, "Planning"),
        React.createElement(
          "select",
          {
            className: "status-select",
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
        { className: "status-pill status-pill-select" },
        React.createElement("span", { className: "status-label" }, "Model"),
        React.createElement(
          "select",
          {
            className: "status-select",
            value: model || "",
            disabled: localOnlyEnabled,
            onChange: (event) => onModelChange?.(event.target.value),
          },
          (availableModels?.length ? availableModels : [model || ""]).map((modelName) =>
            React.createElement("option", { key: modelName, value: modelName }, modelName || "Unknown")
          )
        )
      ),
      contextBudget?.isLow
        ? React.createElement(
            "div",
            { className: "status-warning" },
            `Warning: less than 10% context remains (${contextBudget.remainingLabel}).`
          )
        : null,
      chips.map((chip) =>
        chip.isLocalToggle || chip.isThinkingToggle
          ? React.createElement(
              "label",
              {
                key: chip.label,
                className: `status-pill status-pill-toggle${
                  (chip.isLocalToggle ? localOnlyEnabled : showThinking) ? " enabled" : ""
                }`,
              },
              React.createElement("span", { className: "status-label" }, chip.label),
              React.createElement(
                "span",
                { className: "status-toggle-value" },
                React.createElement("input", {
                  type: "checkbox",
                  checked: Boolean(chip.isLocalToggle ? localOnlyEnabled : showThinking),
                  onChange: (event) =>
                    chip.isLocalToggle
                      ? onLocalOnlyChange?.(event.target.checked)
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
