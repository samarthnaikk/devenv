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
  const [leftWidth, setLeftWidth] = React.useState(320);
  const [rightWidth, setRightWidth] = React.useState(380);
  const [leftCollapsed, setLeftCollapsed] = React.useState(false);
  const [rightCollapsed, setRightCollapsed] = React.useState(false);
  const dragStateRef = React.useRef(null);

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

  React.useEffect(() => {
    function handlePointerMove(event) {
      const dragState = dragStateRef.current;
      if (!dragState) {
        return;
      }

      const minPaneWidth = 220;
      const maxPaneWidth = Math.max(minPaneWidth, Math.floor(window.innerWidth * 0.45));

      if (dragState.side === "left") {
        const nextWidth = clamp(event.clientX, minPaneWidth, maxPaneWidth);
        setLeftCollapsed(false);
        setLeftWidth(nextWidth);
        return;
      }

      const nextWidth = clamp(window.innerWidth - event.clientX, minPaneWidth, maxPaneWidth);
      setRightCollapsed(false);
      setRightWidth(nextWidth);
    }

    function handlePointerUp() {
      dragStateRef.current = null;
      document.body.classList.remove("is-resizing");
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  if (bootError) {
    return React.createElement("div", { className: "loading-shell" }, `Failed to load interface: ${bootError}`);
  }

  if (!health) {
    return React.createElement("div", { className: "loading-shell" }, "Booting Devenv web interface...");
  }

  const gridTemplateColumns = [
    leftCollapsed ? "0px" : `${leftWidth}px`,
    "14px",
    "minmax(0, 1fr)",
    "14px",
    rightCollapsed ? "0px" : `${rightWidth}px`,
  ].join(" ");

  return React.createElement(
    "div",
    { className: "app-shell", style: { gridTemplateColumns } },
    React.createElement(HeaderBar, {
      workspacePath: health.workspace_path,
      status: health.status,
    }),
    React.createElement(
      "aside",
      { className: `left-column${leftCollapsed ? " collapsed" : ""}` },
      React.createElement(LogPanel, {
        title: "Runtime Log",
        badge: `${logEntries.length} lines`,
        entries: logEntries,
        onToggleCollapse: () => setLeftCollapsed((current) => !current),
        collapseLabel: leftCollapsed ? "Expand logs" : "Collapse logs",
        collapseGlyph: leftCollapsed ? ">" : "<",
      })
    ),
    React.createElement("div", {
      className: `pane-resizer left${leftCollapsed ? " collapsed" : ""}`,
      onPointerDown: () => {
        dragStateRef.current = { side: "left" };
        document.body.classList.add("is-resizing");
      },
      children: React.createElement(
        "button",
        {
          className: "pane-toggle",
          type: "button",
          onClick: (event) => {
            event.stopPropagation();
            setLeftCollapsed((current) => !current);
          },
          "aria-label": leftCollapsed ? "Expand logs" : "Collapse logs",
          title: leftCollapsed ? "Expand logs" : "Collapse logs",
        },
        leftCollapsed ? ">" : "<"
      ),
    }),
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
    React.createElement("div", {
      className: `pane-resizer right${rightCollapsed ? " collapsed" : ""}`,
      onPointerDown: () => {
        dragStateRef.current = { side: "right" };
        document.body.classList.add("is-resizing");
      },
      children: React.createElement(
        "button",
        {
          className: "pane-toggle",
          type: "button",
          onClick: (event) => {
            event.stopPropagation();
            setRightCollapsed((current) => !current);
          },
          "aria-label": rightCollapsed ? "Expand chat" : "Collapse chat",
          title: rightCollapsed ? "Expand chat" : "Collapse chat",
        },
        rightCollapsed ? "<" : ">"
      ),
    }),
    React.createElement(
      "aside",
      { className: `right-column${rightCollapsed ? " collapsed" : ""}` },
      React.createElement(TerminalPanel, {
        transcript,
        prompt,
        onPromptChange: setPrompt,
        isRunning,
        onToggleCollapse: () => setRightCollapsed((current) => !current),
        collapseLabel: rightCollapsed ? "Expand chat" : "Collapse chat",
        collapseGlyph: rightCollapsed ? "<" : ">",
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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
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
