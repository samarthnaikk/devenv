import React from "https://esm.sh/react@18";

export function FileTree({
  nodes,
  expandedPaths,
  selectedPath,
  onToggleDirectory,
  onSelectFile,
  depth = 0,
}) {
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
            style: { paddingLeft: `${12 + depth * 16}px` },
            type: "button",
            onClick: () => (node.is_dir ? onToggleDirectory(node.path) : onSelectFile(node.path)),
          },
          React.createElement("span", { className: "tree-caret" }, node.is_dir ? (expandedPaths.has(node.path) ? "▾" : "▸") : "·"),
          React.createElement("span", { className: "tree-icon" }, node.is_dir ? "DIR" : "FILE"),
          React.createElement("span", { className: "tree-name" }, node.name || node.path || "workspace")
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
