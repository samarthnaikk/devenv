import React from "https://esm.sh/react@18";

export function LogPanel({ title, badge, entries, tone = "neutral" }) {
  const rows = entries.length
    ? entries.map((entry, index) =>
        React.createElement(
          "div",
          { key: `${title}-${index}`, className: `log-line ${entry.source || tone}` },
          React.createElement("span", { className: "log-line-number" }, String(index + 1).padStart(3, "0")),
          React.createElement("span", { className: "log-line-tag" }, (entry.source || tone).toUpperCase()),
          React.createElement("span", { className: "log-line-message" }, entry.message || entry)
        )
      )
    : [
        React.createElement(
          "div",
          { key: "empty", className: "log-empty" },
          "No entries yet."
        ),
      ];

  return React.createElement(
    "section",
    { className: `log-panel ${tone}` },
    React.createElement(
      "div",
      { className: "log-panel-header" },
      React.createElement("h3", null, title),
      badge ? React.createElement("span", { className: "log-badge" }, badge) : null
    ),
    React.createElement("div", { className: "log-stream" }, rows)
  );
}
