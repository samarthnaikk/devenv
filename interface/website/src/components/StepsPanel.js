import React from "react";
import { renderMarkdown } from "../lib/markdown.js";
import { BeamFrame, MetalSurface, MotionDeck, MotionReveal } from "./MotionPrimitives.js";

export function StepsPanel({ title, steps, usage, textLogs = [] }) {
  const stepRows = steps.map((step) =>
    React.createElement(
      MotionReveal,
      { key: step.step_id, delay: 40 },
      React.createElement(
        MetalSurface,
        { className: "step-card" },
        React.createElement("span", { className: "step-card-orbit", "aria-hidden": "true" }),
        React.createElement(
          "div",
          { className: "step-header" },
          React.createElement(
            "div",
            { className: "step-header-copy" },
            React.createElement("strong", { className: "step-title" }, step.tool_name),
            React.createElement("span", { className: "step-caption" }, "Arguments and tool output from this execution step")
          ),
          React.createElement("span", { className: step.success ? "step-ok" : "step-fail" }, step.success ? "ok" : "fail")
        ),
        React.createElement("pre", { className: "step-block" }, JSON.stringify(step.arguments, null, 2)),
        React.createElement("div", {
          className: "markdown-body step-output",
          dangerouslySetInnerHTML: { __html: renderMarkdown(step.output) },
        })
      )
    )
  );

  const logRows = textLogs.map((entry, index) =>
    React.createElement(
      MotionReveal,
      { key: `${title || "log"}-${index}`, delay: index * 35 },
      React.createElement(
        MetalSurface,
        { className: "step-card log-card" },
        React.createElement("span", { className: "step-card-orbit", "aria-hidden": "true" }),
        React.createElement("div", { className: "markdown-body step-log-copy" }, entry)
      )
    )
  );

  return React.createElement(
    MotionReveal,
    { className: "content-panel-shell" },
    React.createElement(
      BeamFrame,
      { active: false, tone: "mono", className: "content-panel steps-panel" },
      React.createElement("span", { className: "content-panel-ribbon", "aria-hidden": "true" }),
      React.createElement(
        "div",
        { className: "content-panel-head" },
        React.createElement(
          "div",
          { className: "content-panel-copy" },
          React.createElement("div", { className: "panel-label" }, title || "Execution Trace"),
          React.createElement("h2", { className: "content-panel-title" }, "Execution surface"),
          React.createElement("p", { className: "content-panel-note" }, "Trace tool calls, arguments, and the first visible output without leaving the workspace.")
        ),
        React.createElement(
          "div",
          { className: "content-panel-rail" },
          React.createElement("span", { className: "content-panel-pill" }, `${steps.length || textLogs.length || 0} events`),
          React.createElement("span", { className: "content-panel-pill" }, `Total tokens ${usage.total_tokens || 0}`),
          React.createElement("span", { className: "content-panel-rail-copy" }, stepRows.length ? "Detailed tool execution" : "Log-only trace")
        )
      ),
      React.createElement(
        MotionDeck,
        { className: "steps-list" },
        stepRows.length
          ? stepRows
          : logRows.length
            ? logRows
            : React.createElement("div", { className: "content-panel-empty" }, "No entries yet.")
      )
    )
  );
}
