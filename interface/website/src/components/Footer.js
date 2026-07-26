import React from "react";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel } from "../utils/format.js";
import { ThinkingOrb } from "./ThinkingOrb.js";
import { BeamFrame, MotionNumber, MotionShimmerText } from "./MotionPrimitives.js";

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

  return React.createElement(
    "footer",
    { className: "app-footer px-4 pb-4 pt-2 shrink-0" },
    React.createElement(
      BeamFrame,
      { active: state.isRunning, tone: "mono", className: "app-footer-shell flex justify-between items-center rounded-[22px] px-4 py-3" },
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
        { className: "app-footer-metrics flex items-center gap-3" },
        React.createElement(
          "span",
          { className: "app-footer-pill font-code-sm text-[10px] text-on-surface-variant" },
          React.createElement(MotionNumber, { value: remainingLabel })
        )
      )
    )
  );
}
