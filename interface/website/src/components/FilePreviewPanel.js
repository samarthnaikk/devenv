import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";
import { FileTree } from "./FileTree.js";

export function FilePreviewPanel({
  nodes,
  expandedPaths,
  selectedPath,
  content,
  isPreviewVisible,
  onSelectFile,
  onToggleDirectory,
  onOpenPreview,
  onClose,
}) {
  return React.createElement(
    "section",
    { className: "content-panel preview-panel" },
    React.createElement(
      "div",
      { className: "editor-toolbar" },
      React.createElement(
        "div",
        { className: "editor-toolbar-group" },
        React.createElement("div", { className: "panel-label" }, "Explorer"),
        React.createElement("div", { className: "preview-path" }, selectedPath || "No file selected")
      ),
      React.createElement(
        "div",
        { className: "sidebar-actions" },
        React.createElement(
          "button",
          {
            className: "preview-button",
            type: "button",
            disabled: !selectedPath,
            onClick: onOpenPreview,
          },
          "Open Preview"
        ),
        React.createElement(
          "button",
          {
            className: "preview-button secondary",
            type: "button",
            disabled: !isPreviewVisible,
            onClick: onClose,
          },
          "Close Preview"
        )
      )
    ),
    React.createElement(
      "div",
      { className: "editor-body" },
      React.createElement(
        "aside",
        { className: "editor-explorer" },
        React.createElement(
          "div",
          { className: "sidebar-header" },
          React.createElement("div", { className: "panel-label" }, "Files"),
          React.createElement(
            "div",
            { className: "sidebar-caption" },
            "Select a file, then open it in the preview pane."
          )
        ),
        React.createElement(FileTree, {
          nodes,
          expandedPaths,
          selectedPath,
          onToggleDirectory,
          onSelectFile,
        })
      ),
      React.createElement(
        "section",
        { className: "editor-preview" },
        React.createElement("h3", { className: "preview-title" }, selectedPath ? selectedPath.split("/").at(-1) : "Preview"),
        React.createElement(
          "div",
          { className: "preview-caption" },
          isPreviewVisible
            ? "Manual preview is open for the selected file."
            : "Preview stays blank until you explicitly open a file."
        ),
        isPreviewVisible
          ? React.createElement("div", {
              className: "markdown-body preview-body",
              dangerouslySetInnerHTML: {
                __html: renderMarkdown(content || "No preview content loaded."),
              },
            })
          : React.createElement(
              "div",
              { className: "preview-empty" },
              "No file preview is open."
            )
      )
      )
  );
}
