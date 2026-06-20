import React from "https://esm.sh/react@18";
import { fetchFile, fetchFiles, fetchHealth, runTurn } from "./api.js";
import { FilePreviewPanel } from "./components/FilePreviewPanel.js";
import { HeaderBar } from "./components/HeaderBar.js";
import { LogPanel } from "./components/LogPanel.js";
import { TerminalPanel } from "./components/TerminalPanel.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [tree, setTree] = React.useState([]);
  const [expandedPaths, setExpandedPaths] = React.useState(new Set([""]));
  const [selectedPath, setSelectedPath] = React.useState("");
  const [selectedContent, setSelectedContent] = React.useState("");
  const [isPreviewVisible, setIsPreviewVisible] = React.useState(false);
  const [prompt, setPrompt] = React.useState("Tell me about this project.");
  const [transcript, setTranscript] = React.useState([
    {
      role: "assistant",
      content:
        "I’m connected to your workspace. Expand folders on the left, inspect files only when you want a preview, and use the center terminal to ask the runtime about the project.",
    },
  ]);
  const [logEntries, setLogEntries] = React.useState([]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [bootError, setBootError] = React.useState("");

  React.useEffect(() => {
    Promise.all([fetchHealth(), fetchFiles("")])
      .then(([healthPayload, filePayload]) => {
        setHealth(healthPayload);
        setTree(normalizeEntries(filePayload.entries));
        setLogEntries([
          createLogEntry("system", `Workspace loaded: ${healthPayload.workspace_path}`),
          createLogEntry("system", `Available tools: ${healthPayload.tools.join(", ")}`),
          createLogEntry("ai", "AI channel connected and waiting for prompts"),
        ]);
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
      { className: "left-column" },
      React.createElement(LogPanel, {
        title: "Runtime Log",
        badge: `${logEntries.length} lines`,
        entries: logEntries,
      })
    ),
    React.createElement(
      "main",
      { className: "main-column" },
      React.createElement(FilePreviewPanel, {
        nodes: tree,
        expandedPaths,
        selectedPath,
        content: selectedContent,
        isPreviewVisible,
        onSelectFile: (path) => {
          setSelectedPath(path);
          setLogEntries((current) => [...current, createLogEntry("system", `Selected file: ${path}`)]);
        },
        onToggleDirectory: async (path) => {
          const next = new Set(expandedPaths);
          if (next.has(path)) {
            next.delete(path);
            setExpandedPaths(next);
            setLogEntries((current) => [...current, createLogEntry("system", `Collapsed directory: ${path || "."}`)]);
            return;
          }

          const payload = await fetchFiles(path);
          setTree((current) => attachChildren(current, path, normalizeEntries(payload.entries)));
          next.add(path);
          setExpandedPaths(next);
          setLogEntries((current) => [...current, createLogEntry("system", `Expanded directory: ${path || "."}`)]);
        },
        onOpenPreview: async () => {
          if (!selectedPath) {
            return;
          }
          const filePayload = await fetchFile(selectedPath);
          setSelectedContent(filePayload.content);
          setIsPreviewVisible(true);
          setLogEntries((current) => [...current, createLogEntry("system", `Preview opened: ${selectedPath}`)]);
        },
        onClose: () => {
          setIsPreviewVisible(false);
          setLogEntries((current) => [...current, createLogEntry("system", "Preview closed")]);
        },
      })
    ),
    React.createElement(
      "aside",
      { className: "right-column" },
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
          setLogEntries((current) => [...current, createLogEntry("system", `Prompt submitted: ${nextPrompt}`)]);

          try {
            const result = await runTurn(nextPrompt);
            setTranscript((current) => [
              ...current,
              { role: "assistant", content: result.final_response || "No assistant response returned." },
            ]);
            setLogEntries((current) => [...current, ...buildLogEntries(result)]);
            setPrompt("");
          } catch (error) {
            setTranscript((current) => [
              ...current,
              { role: "assistant", content: `Request failed: ${error.message}` },
            ]);
            setLogEntries((current) => [...current, createLogEntry("system", `Request failed: ${error.message}`)]);
          } finally {
            setIsRunning(false);
          }
        },
      })
    )
  );
}

function normalizeEntries(entries) {
  return [...entries]
    .sort((left, right) => {
      if (left.is_dir !== right.is_dir) {
        return left.is_dir ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    })
    .map(toTreeNode);
}

function toTreeNode(entry) {
  return { ...entry, children: entry.children ? normalizeEntries(entry.children) : null };
}

function attachChildren(nodes, targetPath, children) {
  return nodes.map((node) => {
    if (node.path === targetPath) {
      return { ...node, children };
    }
    if (!node.children) {
      return node;
    }
    return { ...node, children: attachChildren(node.children, targetPath, children) };
  });
}

function buildLogEntries(result) {
  const systemLogs = result.system_logs?.length
    ? result.system_logs.map((entry) => createLogEntry("system", entry))
    : [createLogEntry("system", "No runtime system logs were returned for this turn.")];
  const aiLogs = result.ai_logs?.length
    ? result.ai_logs.map((entry) => createLogEntry("ai", entry))
    : [createLogEntry("ai", "No AI-side trace was emitted for this turn.")];
  const stepLogs = (result.steps || []).map((step, index) =>
    createLogEntry(
      step.success ? "system" : "error",
      `Step ${index + 1}: ${step.tool_name} ${step.success ? "completed successfully" : "failed"}`
    )
  );

  return [...systemLogs, ...aiLogs, ...stepLogs];
}

function createLogEntry(source, message) {
  return {
    source,
    message,
  };
}
