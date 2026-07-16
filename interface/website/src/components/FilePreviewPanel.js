import React from "https://esm.sh/react@18.2.0";
import { renderFilePreview } from "../lib/file_preview.js";
import { FileTree } from "./FileTree.js";

export function FilePreviewPanel({
  nodes,
  expandedPaths,
  selectedPath,
  content,
  previewKind,
  contentType,
  onSelectFile,
  onToggleDirectory,
}) {
  const preview = previewKind === "text" ? renderFilePreview(content || "", selectedPath) : null;

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
            "Select a file to preview it instantly."
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
          selectedPath ? "Preview updates automatically as you browse files." : "Preview a file by selecting it in the explorer."
        ),
        selectedPath
          ? previewKind === "image"
            ? React.createElement("img", {
                className: "image-preview",
                src: content,
                alt: selectedPath || "Image preview",
              })
            : previewKind === "pdf"
              ? React.createElement(
                  "div",
                  { className: "pdf-preview-shell" },
                  React.createElement(
                    "div",
                    { className: "pdf-preview-meta" },
                    React.createElement("span", { className: "material-symbols-outlined text-[16px] text-primary" }, "picture_as_pdf"),
                    React.createElement(
                      "span",
                      { className: "text-[12px] text-on-surface-variant" },
                      "Inline PDF preview for generated documents"
                    )
                  ),
                  React.createElement("iframe", {
                    className: "pdf-preview-frame",
                    src: content,
                    title: selectedPath || "PDF preview",
                  })
                )
            : previewKind === "binary"
              ? React.createElement(
                  "div",
                  { className: "preview-empty" },
                  `Preview unavailable for ${contentType || "binary"} files.`
                )
              : React.createElement("div", {
                  className: preview.className,
                  dangerouslySetInnerHTML: {
                    __html: preview.html,
                  },
                })
          : React.createElement(
              "div",
              { className: "preview-empty" },
              "No file selected."
            )
      )
      )
  );
}
