import React from "https://esm.sh/react@18";

export function FileSidebar({ entries, activePath, onSelectFile, onEnterDirectory, currentPath }) {
  const rows = entries.map((entry) =>
    React.createElement(
      "button",
      {
        key: entry.path,
        className: `file-row${activePath === entry.path ? " active" : ""}`,
        type: "button",
        onClick: () => (entry.is_dir ? onEnterDirectory(entry.path) : onSelectFile(entry.path)),
      },
      React.createElement("span", { className: "file-icon" }, entry.is_dir ? "DIR" : "FILE"),
      React.createElement("span", { className: "file-name" }, entry.name)
    )
  );

  return React.createElement(
    "aside",
    { className: "sidebar-panel" },
    React.createElement("div", { className: "panel-label" }, currentPath || "workspace"),
    currentPath
      ? React.createElement(
          "button",
          {
            className: "file-row back-row",
            type: "button",
            onClick: () => onEnterDirectory(parentPath(currentPath)),
          },
          React.createElement("span", { className: "file-icon" }, "UP"),
          React.createElement("span", { className: "file-name" }, "..")
        )
      : null,
    React.createElement("div", { className: "file-list" }, rows)
  );
}

function parentPath(path) {
  const parts = path.split("/").filter(Boolean);
  parts.pop();
  return parts.join("/");
}
