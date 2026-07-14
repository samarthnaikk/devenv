import React from "https://esm.sh/react@18.2.0";

export function FileTree({
  nodes,
  expandedPaths,
  selectedPath,
  onToggleDirectory,
  onSelectFile,
  depth = 0,
}) {
  if (!nodes.length) {
    return React.createElement(
      "div",
      { className: "tree-empty" },
      "No files found in this directory."
    );
  }

  return React.createElement(
    "div",
    { className: "tree-group" },
    nodes.map((node) =>
      React.createElement(
        "div",
        { key: node.path || "__root__" },
        React.createElement(
          "button",
          {
            className: `tree-row${selectedPath === node.path ? " active" : ""}`,
            style: { paddingLeft: `${12 + depth * 14}px` },
            type: "button",
            onClick: () => (node.is_dir ? onToggleDirectory(node.path) : onSelectFile(node.path)),
          },
          React.createElement("span", { className: "tree-caret" }, node.is_dir ? (expandedPaths.has(node.path) ? "▾" : "▸") : "·"),
          React.createElement("span", { className: `tree-icon ${node.is_dir ? "directory" : "file"}` }, node.is_dir ? "DIR" : "FILE"),
          React.createElement(
            "span",
            { className: "tree-name-wrap" },
            React.createElement("span", { className: "tree-name" }, node.name || node.path || "workspace"),
            node.is_dir && node.children
              ? React.createElement(
                  "span",
                  { className: "tree-meta" },
                  `${node.children.length} ${node.children.length === 1 ? "item" : "items"}`
                )
              : null
          )
        ),
        node.is_dir && expandedPaths.has(node.path) && node.children
          ? React.createElement(FileTree, {
              nodes: node.children,
              expandedPaths,
              selectedPath,
              onToggleDirectory,
              onSelectFile,
              depth: depth + 1,
            })
          : null
      )
    )
  );
}
