import React from "https://esm.sh/react@18";

export function StepsPanel({ steps, usage }) {
  const stepRows = steps.map((step) =>
    React.createElement(
      "article",
      { key: step.step_id, className: "step-card" },
      React.createElement(
        "div",
        { className: "step-header" },
        React.createElement("strong", null, step.tool_name),
        React.createElement("span", { className: step.success ? "step-ok" : "step-fail" }, step.success ? "ok" : "fail")
      ),
      React.createElement("pre", { className: "step-block" }, JSON.stringify(step.arguments, null, 2)),
      React.createElement("pre", { className: "step-block" }, step.output)
    )
  );

  return React.createElement(
    "aside",
    { className: "content-panel steps-panel" },
    React.createElement("div", { className: "panel-label" }, "Execution Trace"),
    React.createElement("div", { className: "usage-row" }, `Total tokens: ${usage.total_tokens || 0}`),
    React.createElement("div", { className: "steps-list" }, stepRows.length ? stepRows : "No tool calls yet.")
  );
}
