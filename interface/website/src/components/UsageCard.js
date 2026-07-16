import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatDuration } from "../utils/format.js";
import { showToast } from "./Header.js";

export function UsageCard() {
  const { state, dispatch } = useApp();
  const statusLabel = state.isRunning ? "Running" : "Idle";
  const statusColor = state.isRunning ? "bg-primary" : "bg-outline";
  const elapsed = state.isRunning ? formatDuration(Date.now() - state.runStartedAt) : formatDuration(state.latestElapsedMs || 0);

  const applyBudget = () => {
    const nextValue = Number.parseInt(state.budgetInput, 10);
    dispatch({ type: "SET_BUDGET_TOKENS", payload: Number.isFinite(nextValue) && nextValue > 0 ? nextValue : null });
    showToast(dispatch, Number.isFinite(nextValue) && nextValue > 0 ? "Session budget updated" : "Session budget cleared");
  };

  return React.createElement(
    "section",
    { className: "space-y-3" },
    React.createElement(
      "h3",
      { className: "font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2" },
      React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "analytics"),
      "USAGE & RUNTIME"
    ),
    React.createElement(
      "div",
      { className: "grid grid-cols-2 gap-3" },
      React.createElement(
        "div",
        { className: "p-3 bg-surface-container rounded-lg border border-outline-variant" },
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
        { className: "p-3 bg-surface-container rounded-lg border border-outline-variant" },
        React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Elapsed"),
        React.createElement("div", { className: "font-body-md text-body-md font-bold" }, escapeHtml(elapsed))
      ),
      React.createElement(
        "div",
        { className: "p-3 bg-surface-container rounded-lg border border-outline-variant" },
        React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Last request"),
        React.createElement("div", { className: "font-body-md text-body-md font-bold" }, escapeHtml(formatDuration(state.latestElapsedMs || 0)))
      ),
      React.createElement(
        "div",
        { className: "p-3 bg-surface-container rounded-lg border border-outline-variant" },
        React.createElement("div", { className: "font-label-caps text-label-caps text-outline mb-1 uppercase" }, "Session total"),
        React.createElement("div", { className: "font-body-md text-body-md font-bold" }, `${String(state.sessionUsageTotal || 0)} tokens`)
      )
    ),
    React.createElement(
      "div",
      { className: "flex flex-col gap-1.5 pt-2" },
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
  );
}
