import React from "react";
import { useApp } from "../context/AppContext.js";
import { BeamFrame, MotionReveal } from "./MotionPrimitives.js";

export function Toast() {
  const { state } = useApp();
  if (!state.toast) return null;

  return React.createElement(
    MotionReveal,
    { delay: 120, className: "toast-banner-shell" },
    React.createElement(
      BeamFrame,
      { tone: "sunrise", className: "toast-banner markdown-body inline-markdown" },
      React.createElement("span", { className: "toast-banner-orbit", "aria-hidden": "true" }),
      React.createElement("span", { className: "material-symbols-outlined toast-banner-icon" }, "notifications_active"),
      React.createElement(
        "span",
        { className: "toast-banner-copy-wrap" },
        React.createElement("span", { className: "toast-banner-label" }, "Runtime update"),
        React.createElement("span", { className: "toast-banner-copy" }, state.toast)
      )
    )
  );
}
