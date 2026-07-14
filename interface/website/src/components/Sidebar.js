import React from "https://esm.sh/react@18.2.0";
import { AccessCard } from "./AccessCard.js";
import { SessionsCard } from "./SessionsCard.js";
import { UsageCard } from "./UsageCard.js";
import { Footer } from "./Footer.js";

export function Sidebar() {
  return React.createElement(
    "aside",
    { className: "w-[35%] flex flex-col h-full bg-surface-container-low border-l border-outline-variant" },
    React.createElement(
      "div",
      { className: "flex-1 overflow-y-auto p-4 space-y-6" },
      React.createElement(
        "div",
        { className: "flex items-center gap-2 mb-2" },
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "DEVELOPER WORKSPACE")
      ),
      React.createElement(AccessCard, null),
      React.createElement(SessionsCard, null),
      React.createElement(UsageCard, null)
    ),
    React.createElement(Footer, null)
  );
}
