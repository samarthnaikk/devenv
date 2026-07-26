import React from "react";
import { useApp } from "../context/AppContext.js";
import { Transcript } from "./Transcript.js";
import { Composer } from "./Composer.js";
import { BeamFrame, MotionNumber, MotionReveal, MotionShimmerText, MotionStage, MotionSwap } from "./MotionPrimitives.js";

export function ChatColumn() {
  const { state } = useApp();
  const routeModes = ["direct", "plan", "memory", "web", "knowledge"];
  const pendingMode = state.isRunning ? state.pendingRunMode : state.planMode ? "plan" : "direct";
  const activeModeIndex = Math.max(0, routeModes.indexOf(pendingMode));
  const statCards = [
    {
      label: "Context",
      value: state.retrievalStatus?.label || "New context",
      detail: state.retrievalStatus?.detail || "No prior session reused yet.",
    },
    {
      label: "Backend",
      value: state.activeBackend || state.preferredBackend || "opencode",
      detail: state.isRunning ? "Actively answering" : "Ready for next turn",
    },
    {
      label: "Last turn",
      value: `${Number(state.latestTurnTokens || 0)}`,
      detail: state.latestElapsedMs ? `${Math.max(0.1, state.latestElapsedMs / 1000).toFixed(1)}s end-to-end` : "No completed turn yet",
      numeric: true,
    },
  ];

  return React.createElement(
    "section",
    { className: "chat-column flex-1 min-w-0 flex flex-col h-full bg-background relative" },
    React.createElement("div", { className: "chat-column-grid", "aria-hidden": "true" }),
    React.createElement("div", { className: "chat-column-glow chat-column-glow-one", "aria-hidden": "true" }),
    React.createElement("div", { className: "chat-column-glow chat-column-glow-two", "aria-hidden": "true" }),
    React.createElement("div", { className: "chat-column-ribbon chat-column-ribbon-one", "aria-hidden": "true" }),
    React.createElement("div", { className: "chat-column-ribbon chat-column-ribbon-two", "aria-hidden": "true" }),
    React.createElement(
      MotionStage,
      { axis: "x", className: "chat-status-shell px-margin-desktop pt-3" },
      React.createElement(
        BeamFrame,
        { active: state.isRunning, tone: "mono", className: "chat-status-rail rounded-[26px] px-4 py-3" },
        React.createElement(
          "div",
          {
            className: "chat-status-mode-tabs",
            style: { "--active-index": String(activeModeIndex), "--tab-count": String(routeModes.length) },
          },
          React.createElement("span", { className: "chat-status-mode-indicator", "aria-hidden": "true" }),
          routeModes.map((mode) =>
            React.createElement(
              "span",
              {
                key: mode,
                className: `chat-status-tab${pendingMode === mode ? " is-active" : ""}`,
              },
              mode
            )
          )
        ),
        React.createElement(
          MotionSwap,
          { className: "chat-status-routecopy" },
          React.createElement("span", { className: "chat-status-route-label" }, describePendingMode(pendingMode))
        ),
        React.createElement(
          "div",
          { className: "chat-status-grid" },
          statCards.map((card, index) =>
            React.createElement(
              MotionReveal,
              { key: card.label, delay: index * 70 },
              React.createElement(
                "div",
                { className: "chat-status-card" },
                React.createElement("div", { className: "chat-status-card-label" }, card.label),
                React.createElement(
                  "div",
                  { className: "chat-status-card-value" },
                  card.numeric
                    ? React.createElement(MotionNumber, { value: card.value })
                    : React.createElement(MotionShimmerText, { active: state.isRunning && card.label === "Backend" }, card.value)
                ),
                React.createElement("div", { className: "chat-status-card-detail" }, card.detail)
              )
            )
          )
        )
      )
    ),
    React.createElement(Transcript, null),
    React.createElement(Composer, null)
  );
}

function describePendingMode(mode) {
  if (mode === "plan") return "Blueprint route active. The next turn is staged as a flow before execution.";
  if (mode === "memory") return "Memory route active. Prior sessions and workspace facts take priority.";
  if (mode === "web") return "Web route active. Current answers are pushed through live-source retrieval.";
  if (mode === "knowledge") return "Knowledge route active. Repos, docs, and references are being gathered.";
  return "Direct route active. The runtime can choose memory, tools, plan, or live search.";
}
