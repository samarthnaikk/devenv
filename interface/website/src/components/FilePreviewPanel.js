import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";

export function FilePreviewPanel({ path, content, onClose }) {
  return React.createElement(
    "section",
    { className: "content-panel preview-panel" },
    React.createElement(
      "div",
      { className: "preview-header" },
      React.createElement(
        "div",
        null,
        React.createElement("div", { className: "panel-label" }, "File Preview"),
        React.createElement("h3", { className: "preview-title" }, path ? path.split("/").at(-1) : "No file selected"),
        React.createElement("div", { className: "preview-path" }, path || "No file selected")
      ),
      React.createElement(
        "button",
        { className: "preview-close", type: "button", onClick: onClose },
        "Hide Preview"
      )
    ),
    React.createElement(
      "div",
      { className: "preview-caption" },
      "Preview is manual only. Pick a file on the left and close it anytime."
    ),
    React.createElement("div", {
      className: "markdown-body preview-body",
      dangerouslySetInnerHTML: {
        __html: renderMarkdown(content || "No preview content loaded."),
      },
    })
  );
}
