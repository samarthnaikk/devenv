import React from "https://esm.sh/react@18";
import { fetchFile, fetchFiles, fetchHealth, runTurn } from "./api.js";
import { FilePreviewPanel } from "./components/FilePreviewPanel.js";
import { FileTree } from "./components/FileTree.js";
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
        setTree(normalizeEntries(filePayload.entries));
        setAiLogs(["AI channel connected and waiting for prompts"]);
        setSystemLogs([
          `Workspace loaded: ${healthPayload.workspace_path}`,
          `Available tools: ${healthPayload.tools.join(", ")}`,
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
      { className: "sidebar-panel left-column" },
      React.createElement(
        "div",
        { className: "sidebar-header" },
        React.createElement("div", { className: "panel-label" }, "Workspace Tree"),
        React.createElement(
          "div",
          { className: "sidebar-caption" },
          selectedPath ? `Selected: ${selectedPath}` : "Pick a file or expand a directory"
        )
      ),
      React.createElement(FileTree, {
        nodes: tree,
        expandedPaths,
        selectedPath,
        onSelectFile: (path) => {
          setSelectedPath(path);
          setSystemLogs((current) => [...current, `Selected file: ${path}`]);
        },
        onToggleDirectory: async (path) => {
          const next = new Set(expandedPaths);
          if (next.has(path)) {
            next.delete(path);
            setExpandedPaths(next);
            setSystemLogs((current) => [...current, `Collapsed directory: ${path || "."}`]);
            return;
          }

          const payload = await fetchFiles(path);
          setTree((current) => attachChildren(current, path, normalizeEntries(payload.entries)));
          next.add(path);
          setExpandedPaths(next);
          setSystemLogs((current) => [...current, `Expanded directory: ${path || "."}`]);
        },
      }),
      React.createElement(
        "div",
        { className: "sidebar-actions" },
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
              setIsPreviewVisible(true);
              setSystemLogs((current) => [...current, `Preview opened: ${selectedPath}`]);
            },
          },
          "Preview File"
        ),
        React.createElement(
          "button",
          {
            className: "preview-button secondary",
            type: "button",
            disabled: !isPreviewVisible,
            onClick: () => {
              setIsPreviewVisible(false);
              setSystemLogs((current) => [...current, "Preview closed"]);
            },
          },
          "Hide Preview"
        )
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
          setSystemLogs((current) => [...current, `Prompt submitted: ${nextPrompt}`]);

          try {
            const result = await runTurn(nextPrompt);
            setTranscript((current) => [
              ...current,
              { role: "assistant", content: result.final_response || "No assistant response returned." },
            ]);
            setUsage(result.total_usage);
            setSteps(result.steps || []);
            setAiLogs(result.ai_logs?.length ? result.ai_logs : ["No AI-side trace was emitted for this turn."]);
            setSystemLogs(buildSystemFeed(result));
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
      isPreviewVisible
        ? React.createElement(FilePreviewPanel, {
            path: selectedPath,
            content: selectedContent,
            onClose: () => {
              setIsPreviewVisible(false);
              setSystemLogs((current) => [...current, "Preview closed"]);
            },
          })
        : null
    ),
    React.createElement(
      "aside",
      { className: "right-column" },
      React.createElement(LogPanel, {
        title: "AI Logs",
        badge: `${usage.total_tokens || 0} tokens`,
        entries: aiLogs,
        tone: "ai",
      }),
      React.createElement(LogPanel, {
        title: "System Logs",
        badge: `${steps.length} step${steps.length === 1 ? "" : "s"}`,
        entries: systemLogs,
        tone: "system",
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

function buildSystemFeed(result) {
  const baseLogs = result.system_logs?.length
    ? result.system_logs
    : ["No runtime system logs were returned for this turn."];
  const stepLogs = (result.steps || []).map(
    (step, index) =>
      `Step ${index + 1}: ${step.tool_name} ${step.success ? "completed successfully" : "failed"}`
  );

  return [...baseLogs, ...stepLogs];
}
