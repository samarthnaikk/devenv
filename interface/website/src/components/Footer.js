import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel } from "../utils/format.js";

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
    { className: "p-4 bg-surface-container-highest border-t border-outline-variant flex justify-between items-center shrink-0" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2" },
      React.createElement("div", { className: `w-2 h-2 rounded-full ${state.isRunning ? "bg-primary glowing-pip animate-pulse" : "bg-primary glowing-pip"}` }),
      React.createElement(
        "div",
        { className: "flex flex-col" },
        React.createElement("span", { className: "font-label-caps text-[10px] text-on-surface" }, state.isRunning ? "Running" : backendReadyLabel),
        React.createElement("span", { className: "font-code-sm text-[9px] text-on-surface-variant" }, modelLabel)
      )
    ),
    React.createElement("span", { className: "font-code-sm text-[10px] text-on-surface-variant" }, remainingLabel)
  );
}
