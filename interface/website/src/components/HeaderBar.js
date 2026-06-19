import React from "https://esm.sh/react@18";

export function HeaderBar({ workspacePath, status }) {
  return React.createElement(
    "header",
    { className: "header-bar" },
    React.createElement(
      "div",
      { className: "header-copy" },
      React.createElement("p", { className: "eyebrow" }, "Devenv Runtime"),
      React.createElement("h1", null, "Web Terminal"),
      React.createElement("span", { className: "workspace-path" }, workspacePath || "Loading workspace...")
    ),
    React.createElement(
      "div",
      { className: "status-pill" },
      React.createElement("span", { className: "status-dot" }),
      status || "Connecting"
    )
  );
}
