import React from "https://esm.sh/react@18";
import { fetchFile, fetchFiles, fetchHealth, runTurn } from "./api.js";
import { FilePreviewPanel } from "./components/FilePreviewPanel.js";
import { HeaderBar } from "./components/HeaderBar.js";
import { TerminalPanel } from "./components/TerminalPanel.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [tree, setTree] = React.useState([]);
  const [expandedPaths, setExpandedPaths] = React.useState(new Set([""]));
  const [selectedPath, setSelectedPath] = React.useState("");
  const [selectedPreview, setSelectedPreview] = React.useState({
    kind: "text",
    content: "",
    contentType: "text/plain",
  });
  const [isPreviewVisible, setIsPreviewVisible] = React.useState(false);
  const [prompt, setPrompt] = React.useState("Tell me about this project.");
  const [transcript, setTranscript] = React.useState([
    {
      id: "intro",
      role: "assistant",
      content:
        "I’m connected to your workspace. Expand folders on the left, inspect files only when you want a preview, and use the center terminal to ask the runtime about the project.",
    },
  ]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [bootError, setBootError] = React.useState("");
  const [usage, setUsage] = React.useState({});
  const [healthMeta, setHealthMeta] = React.useState({ provider: "", model: "" });
  const [rightWidth, setRightWidth] = React.useState(380);
  const [rightCollapsed, setRightCollapsed] = React.useState(false);
  const dragStateRef = React.useRef(null);

  React.useEffect(() => {
    Promise.all([fetchHealth(), fetchFiles("")])
      .then(([healthPayload, filePayload]) => {
        setHealth(healthPayload);
        setHealthMeta({
          provider: healthPayload.ai_provider || "",
          model: healthPayload.ai_model || "",
        });
        setTree(normalizeEntries(filePayload.entries));
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
    "minmax(0, 1fr)",
    "14px",
    rightCollapsed ? "0px" : `${rightWidth}px`,
  ].join(" ");

  return React.createElement(
    "div",
    { className: "app-shell", style: { gridTemplateColumns } },
    React.createElement(HeaderBar, {
      workspacePath: health.workspace_path,
      provider: healthMeta.provider,
      model: healthMeta.model,
      usage,
    }),
    React.createElement(
      "main",
      { className: "main-column" },
      React.createElement(FilePreviewPanel, {
        nodes: tree,
        expandedPaths,
        selectedPath,
        content: selectedContent,
        previewKind: selectedPreview.kind,
        contentType: selectedPreview.contentType,
        isPreviewVisible,
        onSelectFile: (path) => {
          setSelectedPath(path);
        },
        onToggleDirectory: async (path) => {
          const next = new Set(expandedPaths);
          if (next.has(path)) {
            next.delete(path);
            setExpandedPaths(next);
            return;
          }

          const payload = await fetchFiles(path);
          setTree((current) => attachChildren(current, path, normalizeEntries(payload.entries)));
          next.add(path);
          setExpandedPaths(next);
        },
        onOpenPreview: async () => {
          if (!selectedPath) {
            return;
          }
          const filePayload = await fetchFile(selectedPath);
          setSelectedPreview({
            kind: filePayload.kind || "text",
            content: filePayload.content || "",
            contentType: filePayload.content_type || "text/plain",
          });
          setIsPreviewVisible(true);
        },
        onClose: () => {
          setIsPreviewVisible(false);
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
          setPrompt("");
          const thinkingId = `thinking-${Date.now()}`;
          const pendingLogs = [
            createLogEntry("system", `Prompt submitted: ${nextPrompt}`),
            createLogEntry("ai", "Waiting for runtime response..."),
          ];
          setTranscript((current) => [
            ...current,
            { id: `user-${Date.now()}`, role: "user", content: nextPrompt },
            { id: thinkingId, role: "thinking", content: formatThinkingBlock(pendingLogs) },
          ]);

          try {
            const result = await runTurn(nextPrompt);
            const turnLogs = buildLogEntries(result);
            setTranscript((current) => [
              ...current.map((entry) =>
                entry.id === thinkingId
                  ? { ...entry, content: formatThinkingBlock(turnLogs) }
                  : entry
              ),
              {
                id: `assistant-${Date.now()}`,
                role: "assistant",
                content: result.final_response || "No assistant response returned.",
              },
            ]);
            setUsage(result.total_usage || {});
          } catch (error) {
            setTranscript((current) => [
              ...current.map((entry) =>
                entry.id === thinkingId
                  ? {
                      ...entry,
                      content: formatThinkingBlock([
                        createLogEntry("system", `Prompt submitted: ${nextPrompt}`),
                        createLogEntry("error", `Request failed: ${error.message}`),
                      ]),
                    }
                  : entry
              ),
              { id: `assistant-${Date.now()}`, role: "assistant", content: `Request failed: ${error.message}` },
            ]);
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

function formatThinkingBlock(entries) {
  return ["```text", ...entries.map((entry) => `${String(entry.source).toUpperCase()}  ${entry.message}`), "```"].join(
    "\n"
  );
}
