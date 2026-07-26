import React from "react";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel } from "../utils/format.js";
import { ThinkingOrb } from "./ThinkingOrb.js";
import { BeamFrame, MotionDeck, MotionNumber, MotionShimmerText } from "./MotionPrimitives.js";

export function Footer() {
  const { state } = useApp();
  const limit = state.rateLimitInfo?.limit || 12000;
  const recentUsage = state.usageWindow.reduce((sum, entry) => sum + entry.totalTokens, 0);
  const remaining = Math.max(limit - recentUsage, 0);
  const remainingLabel = `${remaining}/${limit}`;
  const preferredBackend = state.preferredBackend || state.activeBackend;
  const preferredStatus = state.backends?.[preferredBackend] || null;
  const backendReadyLabel = preferredStatus && preferredStatus.available === false
    ? `${formatBackendLabel(preferredBackend)} offline`
    : `${formatBackendLabel(state.activeBackend)} ready`;
  const modelLabel = state.healthMeta.selectedModelsByBackend?.[preferredBackend] || state.healthMeta.model || "";
  const routeLabel = state.planMode ? "Plan" : !state.selectedTools.length ? "Auto" : `${state.selectedTools.length} route${state.selectedTools.length === 1 ? "" : "s"}`;

  return React.createElement(
    "footer",
    { className: "app-footer px-4 pb-4 pt-2 shrink-0" },
    React.createElement(
      BeamFrame,
      { active: state.isRunning, tone: "mono", className: "app-footer-shell flex justify-between items-center rounded-[22px] px-4 py-3" },
      React.createElement("span", { className: "app-footer-ribbon", "aria-hidden": "true" }),
      React.createElement(
        "div",
        { className: "flex items-center gap-3" },
        state.isRunning
          ? React.createElement(ThinkingOrb, { state: state.pendingRunMode === "web" || state.pendingRunMode === "knowledge" ? "searching" : "working", size: 20, label: "Runtime process" })
          : React.createElement("div", { className: "w-2 h-2 rounded-full bg-primary glowing-pip" }),
        React.createElement(
          "div",
          { className: "flex flex-col" },
          React.createElement("span", { className: "font-label-caps text-[10px] text-on-surface" }, state.isRunning ? "Running" : backendReadyLabel),
          React.createElement(
            MotionShimmerText,
            { className: "font-code-sm text-[9px] text-on-surface-variant", active: state.isRunning },
            modelLabel
          )
        )
      ),
      React.createElement(
        "div",
        { className: "app-footer-metrics" },
        React.createElement(
          MotionDeck,
          { className: "app-footer-grid" },
          footerMetric("Window", React.createElement(MotionNumber, { value: remainingLabel })),
          footerMetric("Route", routeLabel),
          footerMetric("Mode", state.isRunning ? "Live" : "Idle")
        )
      )
    )
  );
}

function footerMetric(label, value) {
  return React.createElement(
    "div",
    { className: "app-footer-pill app-footer-metric" },
    React.createElement("span", { className: "app-footer-metric-label" }, label),
    React.createElement("strong", { className: "app-footer-metric-value" }, value)
  );
}
