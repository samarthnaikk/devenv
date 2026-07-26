import React from "react";

const ORB_STATES = new Set(["working", "searching", "solving", "listening", "composing", "shaping"]);
const DOTS = Array.from({ length: 24 }, (_, index) => index);

export function ThinkingOrb({ state = "working", size = 20, paused = false, label, className = "" }) {
  const orbState = ORB_STATES.has(state) ? state : "working";
  const dimension = size === 64 ? 64 : 20;
  return React.createElement(
    "span",
    {
      className: `thinking-orb thinking-orb-${orbState}${paused ? " is-paused" : ""} ${className}`.trim(),
      style: { width: `${dimension}px`, height: `${dimension}px` },
      role: "img",
      "aria-label": label || `${orbState} process`,
    },
    React.createElement("span", { className: "thinking-orb-core" }),
    React.createElement("span", { className: "thinking-orb-ring thinking-orb-ring-a" }),
    React.createElement("span", { className: "thinking-orb-ring thinking-orb-ring-b" }),
    React.createElement("span", { className: "thinking-orb-scan" }),
    React.createElement(
      "span",
      { className: "thinking-orb-dots", "aria-hidden": "true" },
      DOTS.map((index) => React.createElement("i", { key: index, style: { "--orb-index": index } }))
    )
  );
}

export function stateForThinkingStep(step, pending) {
  const kind = String(step?.kind || "").toLowerCase();
  const label = String(step?.label || "").toLowerCase();
  const text = String(step?.text || "").toLowerCase();
  if (kind.includes("search") || label.includes("search") || text.includes("search")) return "searching";
  if (label.includes("verify") || text.includes("verified")) return "shaping";
  if (label.includes("context") || label.includes("reason") || text.includes("reason")) return "solving";
  if (kind === "tool_call") return "working";
  if (pending && (label.includes("runtime") || text.includes("waiting"))) return "listening";
  if (!pending && (label.includes("trace") || text.includes("prepared"))) return "composing";
  return pending ? "working" : "listening";
}
