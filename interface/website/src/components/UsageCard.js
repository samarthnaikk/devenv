import React from "react";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatDuration } from "../utils/format.js";
import { showToast } from "./Header.js";
import { BeamFrame, MotionNumber, MotionReveal, MotionShimmerText } from "./MotionPrimitives.js";

export function UsageCard() {
  const { state, dispatch } = useApp();
  const statusLabel = state.isRunning ? "Running" : "Idle";
  const statusColor = state.isRunning ? "bg-primary" : "bg-outline";
  const elapsed = state.isRunning ? formatDuration(Date.now() - state.runStartedAt) : formatDuration(state.latestElapsedMs || 0);
  const usedTokens = Number(state.sessionUsageTotal || 0);
  const budgetTokens = Number(state.sessionBudgetTokens || 0);
  const budgetRatio = budgetTokens > 0 ? Math.min(1, usedTokens / budgetTokens) : 0;

  const applyBudget = () => {
    const nextValue = Number.parseInt(state.budgetInput, 10);
    dispatch({ type: "SET_BUDGET_TOKENS", payload: Number.isFinite(nextValue) && nextValue > 0 ? nextValue : null });
    showToast(dispatch, Number.isFinite(nextValue) && nextValue > 0 ? "Session budget updated" : "Session budget cleared");
  };

  return React.createElement(
    "section",
    { className: "workspace-card-stack space-y-3" },
    React.createElement(
      MotionReveal,
      null,
      React.createElement(
        "div",
        { className: "workspace-card-title-row" },
        React.createElement(
          "h3",
          { className: "font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2" },
          React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "analytics"),
          "USAGE & RUNTIME"
        ),
        React.createElement(
          MotionShimmerText,
          { className: "workspace-card-title-meta", active: state.isRunning },
          state.isRunning ? "Turn in progress" : "Budget, timing, and session burn"
        )
      )
    ),
    React.createElement(
      "div",
      { className: "workspace-summary-rail workspace-summary-rail-secondary" },
      React.createElement("span", { className: "workspace-summary-pill" }, statusLabel),
      React.createElement("span", { className: "workspace-summary-pill" }, budgetTokens > 0 ? `${usedTokens}/${budgetTokens}` : "No cap"),
      React.createElement("span", { className: "workspace-summary-pill" }, elapsed),
      React.createElement("span", { className: "workspace-summary-copy" }, "Track live burn, elapsed time, and the session budget before the runtime drifts too far.")
    ),
    React.createElement(
      BeamFrame,
      { active: state.isRunning, tone: "ocean", className: "workspace-card-shell rounded-[22px] p-3" },
      React.createElement("span", { className: "workspace-card-ribbon workspace-card-ribbon-secondary", "aria-hidden": "true" }),
      React.createElement(
        "div",
        { className: "workspace-usage-grid grid grid-cols-2 gap-3" },
        React.createElement(
          "div",
          { className: "workspace-usage-stat p-3 bg-surface-container rounded-lg border border-outline-variant" },
          React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Status"),
          React.createElement(
            "div",
            { className: "flex items-center gap-2" },
            React.createElement("div", { className: `w-2 h-2 rounded-full ${statusColor}` }),
            React.createElement("span", { className: "font-body-md text-body-md font-bold uppercase" }, escapeHtml(statusLabel))
          )
        ),
        React.createElement(
          "div",
          { className: "workspace-usage-stat p-3 bg-surface-container rounded-lg border border-outline-variant" },
          React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Elapsed"),
          React.createElement("div", { className: "font-body-md text-body-md font-bold" }, React.createElement(MotionNumber, { value: elapsed }))
        ),
        React.createElement(
          "div",
          { className: "workspace-usage-stat p-3 bg-surface-container rounded-lg border border-outline-variant" },
          React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Last request"),
          React.createElement("div", { className: "font-body-md text-body-md font-bold" }, React.createElement(MotionNumber, { value: formatDuration(state.latestElapsedMs || 0) }))
        ),
        React.createElement(
          "div",
          { className: "workspace-usage-stat p-3 bg-surface-container rounded-lg border border-outline-variant" },
          React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Session total"),
          React.createElement("div", { className: "font-body-md text-body-md font-bold" }, React.createElement(MotionNumber, { value: `${String(state.sessionUsageTotal || 0)} tokens` }))
        )
      ),
      React.createElement(
        "div",
        { className: "workspace-budget-panel mt-3" },
        React.createElement(
          "div",
          { className: "workspace-budget-head" },
          React.createElement("span", { className: "workspace-budget-label" }, "Budget load"),
          React.createElement("span", { className: "workspace-budget-value" }, budgetTokens > 0 ? `${Math.round(budgetRatio * 100)}%` : "Open")
        ),
        React.createElement(
          "div",
          { className: "workspace-budget-bar" },
          React.createElement("span", {
            className: "workspace-budget-fill",
            style: { width: budgetTokens > 0 ? `${Math.max(6, Math.round(budgetRatio * 100))}%` : "18%" },
          })
        ),
        React.createElement("div", { className: "workspace-budget-detail" }, budgetTokens > 0 ? `${usedTokens} of ${budgetTokens} tokens used this session` : "No cap set. The runtime can keep spending until you apply a budget.")
      ),
      React.createElement(
        "div",
        { className: "flex flex-col gap-1.5 pt-3" },
        React.createElement("label", { className: "font-label-caps text-label-caps text-on-surface-variant" }, "TOKEN BUDGET"),
        React.createElement(
          "div",
          { className: "flex gap-2" },
          React.createElement("input", {
            className: "flex-1 bg-surface-container-highest border border-outline-variant rounded-lg font-code-sm text-code-sm px-3 py-2 outline-none focus:border-primary",
            type: "text",
            value: state.budgetInput,
            onChange: (e) => dispatch({ type: "SET_BUDGET_INPUT", payload: e.target.value }),
          }),
          React.createElement(
            "button",
            {
              type: "button",
              className: "px-4 py-2 bg-surface-variant text-on-surface rounded-lg font-label-caps text-label-caps hover:bg-outline-variant transition-colors",
              onClick: applyBudget,
            },
            "Apply"
          )
        )
      )
    )
  );
}
