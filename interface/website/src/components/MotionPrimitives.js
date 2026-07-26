import React from "https://esm.sh/react@18.2.0";

export function BeamFrame({ children, active = true, tone = "ocean", className = "" }) {
  return React.createElement(
    "div",
    { className: `motion-beam motion-beam-${tone}${active ? " is-active" : ""} ${className}`.trim() },
    React.createElement("span", { className: "motion-beam-track", "aria-hidden": "true" }),
    children
  );
}

export function MetalSurface({ children, className = "" }) {
  return React.createElement(
    "div",
    { className: `motion-metal ${className}`.trim() },
    React.createElement("span", { className: "motion-metal-sheen", "aria-hidden": "true" }),
    children
  );
}

export function MotionReveal({ children, delay = 0, className = "" }) {
  return React.createElement(
    "div",
    { className: `motion-reveal ${className}`.trim(), style: { "--motion-delay": `${delay}ms` } },
    children
  );
}

export function MotionSwap({ children, className = "" }) {
  return React.createElement("span", { className: `motion-swap ${className}`.trim() }, children);
}

export function MotionNumber({ value, className = "" }) {
  return React.createElement("span", { className: `motion-number ${className}`.trim(), key: String(value) }, String(value));
}

export function MotionDeck({ children, className = "" }) {
  return React.createElement("div", { className: `motion-deck ${className}`.trim() }, children);
}

export function MotionStage({ children, className = "", delay = 0, axis = "y" }) {
  return React.createElement(
    "div",
    {
      className: `motion-stage motion-stage-${axis} ${className}`.trim(),
      style: { "--motion-delay": `${delay}ms` },
    },
    children
  );
}

export function MotionShimmerText({ children, className = "", active = true }) {
  return React.createElement(
    "span",
    {
      className: `motion-shimmer-text${active ? " is-active" : ""} ${className}`.trim(),
    },
    children
  );
}

export function MotionStack({ children, className = "" }) {
  return React.createElement("div", { className: `motion-stack ${className}`.trim() }, children);
}
