import React from "react";

export function BeamFrame({ children, active = true, tone = "ocean", className = "", ...props }) {
  return React.createElement(
    "div",
    { className: `motion-beam motion-beam-${tone}${active ? " is-active" : ""} ${className}`.trim(), ...props },
    React.createElement("span", { className: "motion-beam-track", "aria-hidden": "true" }),
    children
  );
}

export function MetalSurface({ children, className = "", ...props }) {
  return React.createElement(
    "div",
    { className: `motion-metal ${className}`.trim(), ...props },
    React.createElement("span", { className: "motion-metal-sheen", "aria-hidden": "true" }),
    children
  );
}

export function MotionReveal({ children, delay = 0, className = "", style = null, ...props }) {
  return React.createElement(
    "div",
    { className: `motion-reveal ${className}`.trim(), style: { ...(style || {}), "--motion-delay": `${delay}ms` }, ...props },
    children
  );
}

export function MotionSwap({ children, className = "", ...props }) {
  return React.createElement("span", { className: `motion-swap ${className}`.trim(), ...props }, children);
}

export function MotionNumber({ value, className = "", ...props }) {
  return React.createElement("span", { className: `motion-number ${className}`.trim(), key: String(value), ...props }, String(value));
}

export function MotionDeck({ children, className = "", ...props }) {
  return React.createElement("div", { className: `motion-deck ${className}`.trim(), ...props }, children);
}

export function MotionStage({ children, className = "", delay = 0, axis = "y", style = null, ...props }) {
  return React.createElement(
    "div",
    {
      className: `motion-stage motion-stage-${axis} ${className}`.trim(),
      style: { ...(style || {}), "--motion-delay": `${delay}ms` },
      ...props,
    },
    children
  );
}

export function MotionShimmerText({ children, className = "", active = true, ...props }) {
  return React.createElement(
    "span",
    {
      className: `motion-shimmer-text${active ? " is-active" : ""} ${className}`.trim(),
      ...props,
    },
    children
  );
}

export function MotionStack({ children, className = "", ...props }) {
  return React.createElement("div", { className: `motion-stack ${className}`.trim(), ...props }, children);
}

export function MotionTilt({ children, className = "", ...props }) {
  return React.createElement(
    "div",
    { className: `motion-tilt ${className}`.trim(), ...props },
    React.createElement("span", { className: "motion-tilt-glare", "aria-hidden": "true" }),
    children
  );
}

export function MotionBadge({ children, className = "", active = false, ...props }) {
  return React.createElement(
    "span",
    { className: `motion-badge${active ? " is-active" : ""} ${className}`.trim(), ...props },
    children
  );
}
