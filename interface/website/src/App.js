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
  const [prompt, setPrompt] = React.useState("Tell me about this project.");
  const [transcript, setTranscript] = React.useState([
    {
      id: "intro",
      role: "assistant",
      content: "I’m connected to your workspace.",
    },
  ]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [bootError, setBootError] = React.useState("");
  const [usage, setUsage] = React.useState({});
  const [healthMeta, setHealthMeta] = React.useState({ provider: "", model: "" });
  const [blueprint, setBlueprint] = React.useState(null);
  const [runtimeState, setRuntimeState] = React.useState("PLANNING");
  const [rightWidth, setRightWidth] = React.useState(380);
  const [rightCollapsed, setRightCollapsed] = React.useState(false);
  const [usageWindow, setUsageWindow] = React.useState([]);
  const [rateLimitInfo, setRateLimitInfo] = React.useState(null);
  const [clock, setClock] = React.useState(Date.now());
  const [planModeEnabled, setPlanModeEnabled] = React.useState(false);
  const dragStateRef = React.useRef(null);

  React.useEffect(() => {
    const intervalId = window.setInterval(() => {
      setClock(Date.now());
      setUsageWindow((current) => current.filter((entry) => Date.now() - entry.timestamp < 60000));
      setRateLimitInfo((current) => {
        if (!current || current.resetAt > Date.now()) {
          return current;
        }
        return null;
      });
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, []);

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
  const contextBudget = buildContextBudget(usageWindow, rateLimitInfo, clock);

  return React.createElement(
    "div",
    { className: "app-shell", style: { gridTemplateColumns } },
    React.createElement(HeaderBar, {
      workspacePath: health.workspace_path,
      provider: healthMeta.provider,
      model: healthMeta.model,
      usage,
      contextBudget,
      planModeEnabled,
      onPlanModeChange: setPlanModeEnabled,
    }),
    React.createElement(
      "main",
      { className: "main-column" },
      React.createElement(FilePreviewPanel, {
        nodes: tree,
        expandedPaths,
        selectedPath,
        content: selectedPreview.content,
        previewKind: selectedPreview.kind,
        contentType: selectedPreview.contentType,
        onSelectFile: async (path) => {
          setSelectedPath(path);
          const filePayload = await fetchFile(path);
          setSelectedPreview({
            kind: filePayload.kind || "text",
            content: filePayload.content || "",
            contentType: filePayload.content_type || "text/plain",
          });
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
        blueprint,
        runtimeState,
        onPromptChange: setPrompt,
        isRunning,
        isCoolingDown: Boolean(rateLimitInfo && rateLimitInfo.resetAt > clock),
        cooldownLabel: rateLimitInfo ? formatDuration(Math.max(rateLimitInfo.resetAt - clock, 0)) : "",
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
            { id: thinkingId, role: "thinking", content: formatThinkingBlock(pendingLogs), pending: true },
          ]);

          try {
            const result = await runTurn(nextPrompt, planModeEnabled ? "force_plan" : "force_direct");
            const turnLogs = buildLogEntries(result);
            setTranscript((current) => [
              ...current.map((entry) =>
                entry.id === thinkingId
                  ? { ...entry, content: formatThinkingBlock(turnLogs), pending: false }
                  : entry
              ),
              {
                id: `assistant-${Date.now()}`,
                role: "assistant",
                content: result.final_response || "No assistant response returned.",
              },
            ]);
            setUsage(result.total_usage || {});
            setBlueprint(result.blueprint || null);
            setRuntimeState(result.state || "PLANNING");
            setUsageWindow((current) =>
              [...current, { timestamp: Date.now(), totalTokens: result.total_usage?.total_tokens || 0 }].filter(
                (entry) => Date.now() - entry.timestamp < 60000
              )
            );
            setRateLimitInfo(null);
          } catch (error) {
            const parsedRateLimit = parseRateLimitError(error.message);
            setTranscript((current) => [
              ...current.map((entry) =>
                entry.id === thinkingId
                  ? {
                      ...entry,
                      pending: false,
                      content: formatThinkingBlock([
                        createLogEntry("system", `Prompt submitted: ${nextPrompt}`),
                        createLogEntry("error", `Request failed: ${error.message}`),
                      ]),
                    }
                  : entry
              ),
              {
                id: `assistant-${Date.now()}`,
                role: parsedRateLimit ? "error" : "assistant",
                content: parsedRateLimit
                  ? [
                      `Rate limit reached.`,
                      ``,
                      `TPM limit: ${parsedRateLimit.limit}`,
                      `Used: ${parsedRateLimit.used}`,
                      `Requested: ${parsedRateLimit.requested}`,
                      `Retry in: ${formatDuration(parsedRateLimit.retryMs)}`,
                      `Resets at: ${formatTimestamp(parsedRateLimit.resetAt)}`,
                    ].join("\n")
                  : `Request failed: ${error.message}`,
              },
            ]);
            if (parsedRateLimit) {
              setRateLimitInfo(parsedRateLimit);
            }
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
    ? result.system_logs
        .filter((entry) => !entry.startsWith("Plan checkpoints:"))
        .map((entry) => createLogEntry("system", entry))
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

function parseRateLimitError(message) {
  const limitMatch = message.match(/Limit\s+(\d+)/i);
  const usedMatch = message.match(/Used\s+(\d+)/i);
  const requestedMatch = message.match(/Requested\s+(\d+)/i);
  const retryMatch = message.match(/try again in\s+([\d.]+)s/i);
  if (!limitMatch || !usedMatch || !requestedMatch || !retryMatch) {
    return null;
  }

  const retryMs = Math.ceil(Number(retryMatch[1]) * 1000);
  return {
    limit: Number(limitMatch[1]),
    used: Number(usedMatch[1]),
    requested: Number(requestedMatch[1]),
    retryMs,
    resetAt: Date.now() + retryMs,
  };
}

function buildContextBudget(usageWindow, rateLimitInfo, now) {
  const limit = rateLimitInfo?.limit || 12000;
  const recentUsage = usageWindow.reduce((sum, entry) => sum + entry.totalTokens, 0);
  const remaining = Math.max(limit - recentUsage, 0);
  const nextResetTimestamp = usageWindow.length ? Math.min(...usageWindow.map((entry) => entry.timestamp + 60000)) : null;
  const resetAt = rateLimitInfo?.resetAt || nextResetTimestamp;

  return {
    remaining,
    remainingLabel: `${remaining}/${limit}`,
    resetAt,
    resetLabel: resetAt ? formatTimestamp(resetAt) : "Idle",
    isLow: remaining / limit < 0.1,
  };
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.max(Math.ceil(milliseconds / 1000), 0);
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60);
  return minutes ? `${minutes}m ${String(seconds).padStart(2, "0")}s` : `${seconds}s`;
}

function formatTimestamp(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
