import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { AccessCard } from "./AccessCard.js";
import { SessionsCard } from "./SessionsCard.js";
import { UsageCard } from "./UsageCard.js";
import { Footer } from "./Footer.js";

export function Sidebar() {
  const { state, dispatch } = useApp();
  const collapsed = Boolean(state.sidebarCollapsed);

  const toggleSidebar = () => {
    dispatch({ type: "SET_SIDEBAR_COLLAPSED", payload: !collapsed });
  };

  if (collapsed) {
    return React.createElement(
      "aside",
      { className: "w-12 shrink-0 flex flex-col items-center justify-center border-l border-outline-variant bg-surface-container-low" },
      React.createElement(
        "button",
        {
          type: "button",
          className: "w-8 h-16 rounded-l-xl rounded-r-md border border-outline-variant bg-surface-container-high text-on-surface-variant hover:text-primary hover:border-primary/40 transition-colors flex items-center justify-center",
          onClick: toggleSidebar,
          "aria-label": "Expand workspace panel",
          title: "Expand workspace panel",
        },
        React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "keyboard_double_arrow_left")
      )
    );
  }

  return React.createElement(
    "aside",
    { className: "w-[24rem] max-w-[32vw] min-w-[20rem] shrink-0 flex flex-col h-full bg-surface-container-low border-l border-outline-variant" },
    React.createElement(
      "div",
      { className: "flex-1 overflow-y-auto p-4 space-y-6" },
      React.createElement(
        "div",
        { className: "flex items-center justify-between gap-3 mb-2" },
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "DEVELOPER WORKSPACE"),
        React.createElement(
          "button",
          {
            type: "button",
            className: "w-8 h-8 rounded-lg border border-outline-variant bg-surface-container text-on-surface-variant hover:text-primary hover:border-primary/40 transition-colors flex items-center justify-center shrink-0",
            onClick: toggleSidebar,
            "aria-label": "Collapse workspace panel",
            title: "Collapse workspace panel",
          },
          React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "keyboard_double_arrow_right")
        )
      ),
      React.createElement(AccessCard, null),
      React.createElement(SessionsCard, null),
      React.createElement(UsageCard, null)
    ),
    React.createElement(Footer, null)
  );
}
