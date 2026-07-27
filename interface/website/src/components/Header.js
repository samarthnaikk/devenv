import React from "react";
import { useApp } from "../context/AppContext.js";
import { MotionReveal, MotionSwap } from "./MotionPrimitives.js";

export function Header() {
  const { state, dispatch } = useApp();

  const toggleSettings = () => {
    dispatch({ type: "SET_SHOW_SETTINGS", payload: !state.showSettings });
  };

  return React.createElement(
    "header",
    { className: "app-header px-margin-desktop w-full z-50 shrink-0" },
    React.createElement(
      MotionReveal,
      { className: "w-full" },
      React.createElement(
        "div",
        { className: "app-header-shell app-header-shell-minimal" },
        React.createElement(
          "div",
          { className: "app-header-minimal-row" },
          React.createElement(
            "div",
            { className: "app-header-brand-minimal" },
            React.createElement(
              "div",
              { className: "app-header-mark" },
              React.createElement(
                MotionSwap,
                { className: "items-center justify-center" },
                React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, state.isRunning ? "bolt" : "auto_awesome")
              )
            ),
            React.createElement(
              "div",
              { className: "app-header-copy-minimal" },
              React.createElement("span", { className: "app-header-title" }, "Devenv"),
              React.createElement(
                "span",
                { className: "app-header-subtitle" },
                state.isRunning ? "Working" : "Ready"
              )
            )
          ),
          React.createElement(
            "button",
            {
              type: "button",
              "data-action": "toggle-settings",
              className: `app-header-button app-header-settings-button icon-button p-2 rounded-2xl text-on-surface-variant ${state.showSettings ? "is-active text-primary" : ""}`,
              onClick: toggleSettings,
              "aria-label": "Settings",
            },
            React.createElement("span", { className: "material-symbols-outlined text-[20px]" }, "settings")
          )
        )
      )
    )
  );
}

let toastTimeoutId = null;

export function showToast(dispatch, message) {
  dispatch({ type: "SET_TOAST", payload: message });
  if (toastTimeoutId) window.clearTimeout(toastTimeoutId);
  toastTimeoutId = window.setTimeout(() => {
    dispatch({ type: "SET_TOAST", payload: "" });
  }, 1600);
}
