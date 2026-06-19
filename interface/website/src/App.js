import React from "https://esm.sh/react@18";
import { fetchFiles, fetchHealth } from "./api.js";
import { FileSidebar } from "./components/FileSidebar.js";
import { HeaderBar } from "./components/HeaderBar.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [entries, setEntries] = React.useState([]);
  const [selectedPath, setSelectedPath] = React.useState("");
  const [currentPath, setCurrentPath] = React.useState("");

  React.useEffect(() => {
    Promise.all([fetchHealth(), fetchFiles("")]).then(([healthPayload, filePayload]) => {
      setHealth(healthPayload);
      setEntries(filePayload.entries);
      setCurrentPath(filePayload.path);
    });
  }, []);

  if (!health) {
    return React.createElement("div", { className: "loading-shell" }, "Booting Devenv web interface...");
  }

  return React.createElement(
    "div",
    { className: "app-shell" },
    React.createElement(HeaderBar, {
      workspacePath: health.workspace_path,
      status: health.status,
    }),
    React.createElement(FileSidebar, {
      entries,
      activePath: selectedPath,
      currentPath,
      onSelectFile: setSelectedPath,
      onEnterDirectory: async (path) => {
        const filePayload = await fetchFiles(path);
        setEntries(filePayload.entries);
        setCurrentPath(filePayload.path);
      },
    }),
    React.createElement(
      "section",
      { className: "content-panel" },
      React.createElement("div", { className: "panel-label" }, "Terminal coming next"),
      React.createElement(
        "p",
        null,
        "Runtime is connected. Selected file: ",
        selectedPath || "none"
      )
    )
  );
}
