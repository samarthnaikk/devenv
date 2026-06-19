import React from "https://esm.sh/react@18";
import { fetchFile, fetchFiles, fetchHealth, runTurn } from "./api.js";
import { FilePreviewPanel } from "./components/FilePreviewPanel.js";
import { FileTree } from "./components/FileTree.js";
import { HeaderBar } from "./components/HeaderBar.js";
import { StepsPanel } from "./components/StepsPanel.js";
import { TerminalPanel } from "./components/TerminalPanel.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [tree, setTree] = React.useState([]);
  const [expandedPaths, setExpandedPaths] = React.useState(new Set());
  const [selectedPath, setSelectedPath] = React.useState("");
  const [selectedContent, setSelectedContent] = React.useState("");
  const [prompt, setPrompt] = React.useState("Tell me about this project.");
  const [transcript, setTranscript] = React.useState([]);
  const [steps, setSteps] = React.useState([]);
  const [usage, setUsage] = React.useState({});
  const [aiLogs, setAiLogs] = React.useState([]);
  const [systemLogs, setSystemLogs] = React.useState([]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [bootError, setBootError] = React.useState("");

  React.useEffect(() => {
    Promise.all([fetchHealth(), fetchFiles("")])
      .then(([healthPayload, filePayload]) => {
        setHealth(healthPayload);
        setTree(filePayload.entries.map((entry) => ({ ...entry, children: null })));
        setExpandedPaths(new Set([""]));
        setSystemLogs((current) => [...current, `Loaded workspace: ${healthPayload.workspace_path}`]);
      })
      .catch((error) => {
        setBootError(error.message);
      });
  }, []);

  if (bootError) {
    return React.createElement("div", { className: "loading-shell" }, `Failed to load interface: ${bootError}`);
  }

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
    React.createElement(
      "aside",
      { className: "sidebar-panel left-column" },
      React.createElement("div", { className: "panel-label" }, "Workspace Tree"),
      React.createElement(FileTree, {
        nodes: tree,
        expandedPaths,
        selectedPath,
        onSelectFile: (path) => {
          setSelectedPath(path);
          setSystemLogs((current) => [...current, `Selected file: ${path}`]);
        },
        onToggleDirectory: async (path) => {
          if (expandedPaths.has(path)) {
            const next = new Set(expandedPaths);
            next.delete(path);
            setExpandedPaths(next);
            return;
          }

          const payload = await fetchFiles(path);
          setTree((current) => attachChildren(current, path, payload.entries));
          const next = new Set(expandedPaths);
          next.add(path);
          setExpandedPaths(next);
          setSystemLogs((current) => [...current, `Expanded directory: ${path || "."}`]);
        },
      }),
      React.createElement(
        "button",
        {
          className: "preview-button",
          type: "button",
          disabled: !selectedPath,
          onClick: async () => {
            if (!selectedPath) {
              return;
            }
            const filePayload = await fetchFile(selectedPath);
            setSelectedContent(filePayload.content);
            setSystemLogs((current) => [...current, `Previewed file: ${selectedPath}`]);
          },
        },
        "Preview selected file"
      )
    ),
    React.createElement(
      "main",
      { className: "main-column" },
      React.createElement(TerminalPanel, {
        transcript,
        prompt,
        onPromptChange: setPrompt,
        isRunning,
        onSubmit: async () => {
          const nextPrompt = prompt.trim();
          if (!nextPrompt) {
            return;
          }
          setIsRunning(true);
          setTranscript((current) => [...current, { role: "user", content: nextPrompt }]);
          setSystemLogs((current) => [...current, `Submitted prompt: ${nextPrompt}`]);
          try {
            const result = await runTurn(nextPrompt);
            setTranscript((current) => [
              ...current,
              { role: "assistant", content: result.final_response || "No assistant response returned." },
            ]);
            setSteps(result.steps);
            setUsage(result.total_usage);
            setAiLogs(result.ai_logs || []);
            setSystemLogs((current) => [...current, ...(result.system_logs || [])]);
            setPrompt("");
          } catch (error) {
            setTranscript((current) => [
              ...current,
              { role: "assistant", content: `Request failed: ${error.message}` },
            ]);
            setSystemLogs((current) => [...current, `Request failed: ${error.message}`]);
          } finally {
            setIsRunning(false);
          }
        },
      }),
      selectedContent
        ? React.createElement(FilePreviewPanel, {
            path: selectedPath,
            content: selectedContent,
          })
        : null
    ),
    React.createElement(
      "aside",
      { className: "right-column" },
      React.createElement(StepsPanel, { title: "AI Logs", steps: [], usage, textLogs: aiLogs }),
      React.createElement(StepsPanel, { title: "System Logs", steps, usage: {}, textLogs: systemLogs })
    )
  );
}

function attachChildren(nodes, targetPath, children) {
  return nodes.map((node) => {
    if (node.path === targetPath) {
      return { ...node, children: children.map((entry) => ({ ...entry, children: null })) };
    }
    if (!node.children) {
      return node;
    }
    return { ...node, children: attachChildren(node.children, targetPath, children) };
  });
}
