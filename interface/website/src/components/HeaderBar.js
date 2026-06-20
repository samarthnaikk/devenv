import React from "https://esm.sh/react@18";

export function HeaderBar({ workspacePath, provider, model, usage }) {
  const chips = [
    { label: "Provider", value: provider || "Unknown" },
    { label: "Model", value: model || "Unknown" },
    { label: "Tokens", value: String(usage?.total_tokens || 0) },
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
      chips.map((chip) =>
        React.createElement(
          "div",
          { key: chip.label, className: "status-pill" },
          React.createElement("span", { className: "status-label" }, chip.label),
          React.createElement("span", { className: "status-value" }, chip.value)
        )
      )
    )
  );
}
