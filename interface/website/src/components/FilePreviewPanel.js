import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";

export function FilePreviewPanel({ path, content }) {
  return React.createElement(
    "aside",
    { className: "content-panel preview-panel" },
    React.createElement("div", { className: "panel-label" }, path || "File Preview"),
    React.createElement("div", {
      className: "markdown-body preview-body",
      dangerouslySetInnerHTML: {
        __html: renderMarkdown(content || "Select a file from the left panel to inspect it here."),
      },
    })
  );
}
