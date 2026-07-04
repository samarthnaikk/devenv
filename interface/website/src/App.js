import React from "https://esm.sh/react@18";
import { fetchHealth, runTurn, updateModel } from "./api.js";
import { TerminalPanel } from "./components/TerminalPanel.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [prompt, setPrompt] = React.useState("");
  const [transcript, setTranscript] = React.useState([]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [bootError, setBootError] = React.useState("");
  const [usage, setUsage] = React.useState({});
  const [healthMeta, setHealthMeta] = React.useState({ provider: "", model: "", availableModels: [] });
  const [blueprint, setBlueprint] = React.useState(null);
  const [runtimeState, setRuntimeState] = React.useState("PLANNING");
  const [stageTraces, setStageTraces] = React.useState([]);
  const [verificationResults, setVerificationResults] = React.useState([]);
  const [usageWindow, setUsageWindow] = React.useState([]);
  const [rateLimitInfo, setRateLimitInfo] = React.useState(null);
  const [clock, setClock] = React.useState(Date.now());
  const [planningMode, setPlanningMode] = React.useState("auto");
  const [localOnlyEnabled, setLocalOnlyEnabled] = React.useState(false);
  const [selectedTools, setSelectedTools] = React.useState([]);

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
    fetchHealth()
      .then((healthPayload) => {
        setHealth(healthPayload);
        setHealthMeta({
          provider: healthPayload.ai_provider || "",
          model: healthPayload.ai_model || "",
          availableModels: healthPayload.available_models || [],
        });
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

  const contextBudget = buildContextBudget(usageWindow, rateLimitInfo);
  const activeProvider = localOnlyEnabled ? "Local" : healthMeta.provider;
  const activeModel = localOnlyEnabled ? "heuristic-runtime" : healthMeta.model;

  return React.createElement(
    "div",
    { className: "app-shell chat-shell" },
    React.createElement(
      "main",
      { className: "chat-main" },
      React.createElement(TerminalPanel, {
        transcript,
        prompt,
        blueprint,
        runtimeState,
        stageTraces,
        verificationResults,
        workspacePath: health.workspace_path,
        provider: activeProvider,
        model: activeModel,
        availableModels: healthMeta.availableModels,
        contextBudget,
        planningMode,
        onPlanningModeChange: setPlanningMode,
        localOnlyEnabled,
        onLocalOnlyChange: setLocalOnlyEnabled,
        availableTools: health.tools || [],
        selectedTools,
        onSelectedToolsChange: setSelectedTools,
        onModelChange: async (nextModel) => {
          const payload = await updateModel(nextModel);
          setHealthMeta((current) => ({
            ...current,
            model: payload.ai_model || nextModel,
            availableModels: payload.available_models || current.availableModels,
          }));
        },
        onPromptChange: setPrompt,
        isRunning,
        isCoolingDown: Boolean(rateLimitInfo && rateLimitInfo.resetAt > clock),
        cooldownLabel: rateLimitInfo ? formatDuration(Math.max(rateLimitInfo.resetAt - clock, 0)) : "",
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
            const aggregateLogs = [];
            let continuePlan = false;
            let result = null;
            const refreshThinking = (pending) => {
              setTranscript((current) =>
                current.map((entry) =>
                  entry.id === thinkingId
                    ? { ...entry, content: formatThinkingBlock(aggregateLogs), pending }
                    : entry
                )
              );
            };

            let autoContinueCount = 0;
            do {
              while (true) {
                try {
                  result = await runTurn(nextPrompt, planningMode, continuePlan, localOnlyEnabled, selectedTools);
                  break;
                } catch (error) {
                  const parsedRateLimit = parseRateLimitError(error.message);
                  if (!parsedRateLimit) {
                    throw error;
                  }

                  setRateLimitInfo(parsedRateLimit);
                  const retryEntry = createLogEntry(
                    "error",
                    `Rate limit reached. Retrying in ${formatDuration(parsedRateLimit.retryMs)}`
                  );
                  aggregateLogs.push(retryEntry);
                  refreshThinking(true);
                  await waitForCooldown(parsedRateLimit.resetAt, (remainingMs) => {
                    retryEntry.message = `Rate limit reached. Retrying in ${formatDuration(remainingMs)}`;
                    refreshThinking(true);
                  });
                  retryEntry.message = "Cooldown finished. Retrying request now.";
                  refreshThinking(true);
                }
              }

              aggregateLogs.push(...buildLogEntries(result));
              setUsage(result.total_usage || {});
              setBlueprint(result.blueprint || null);
              setRuntimeState(result.state || "PLANNING");
              setStageTraces(result.stage_traces || []);
              setVerificationResults(result.verification_results || []);
              setUsageWindow((current) =>
                [...current, { timestamp: Date.now(), totalTokens: result.total_usage?.total_tokens || 0 }].filter(
                  (entry) => Date.now() - entry.timestamp < 60000
                )
              );
              const shouldContinue = shouldAutoContinue(result, autoContinueCount);
              refreshThinking(shouldContinue);
              continuePlan = shouldContinue;
              autoContinueCount += shouldContinue ? 1 : 0;
            } while (continuePlan);

            const finalMessage = selectVisibleAssistantResponse(result, aggregateLogs);
            setTranscript((current) => [
              ...current.map((entry) =>
                entry.id === thinkingId
                  ? { ...entry, content: formatThinkingBlock(aggregateLogs), pending: false }
                  : entry
              ),
              ...(result?.error_message
                ? [
                    {
                      id: `assistant-error-${Date.now()}`,
                      role: "error",
                      content: result.error_message,
                    },
                  ]
                : [
                    {
                      id: `assistant-${Date.now()}`,
                      role: "assistant",
                      content: finalMessage,
                    },
                  ]),
            ]);
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
                      "Rate limit reached.",
                      "",
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

function buildContextBudget(usageWindow, rateLimitInfo) {
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

function hasIncompleteTasks(blueprint) {
  return Boolean(blueprint?.tasks?.some((task) => !task.is_completed));
}

function shouldAutoContinue(result, autoContinueCount) {
  if (!hasIncompleteTasks(result?.blueprint)) {
    return false;
  }
  if (result?.state !== "EXECUTING") {
    return false;
  }
  if (autoContinueCount >= 24) {
    return false;
  }
  const systemLogs = result?.system_logs || [];
  if (systemLogs.some((entry) => String(entry).includes("Verification failed"))) {
    return false;
  }
  return true;
}

async function waitForCooldown(resetAt, onTick) {
  while (true) {
    const remainingMs = Math.max(resetAt - Date.now(), 0);
    onTick(remainingMs);
    if (remainingMs <= 0) {
      return;
    }
    await sleep(Math.min(remainingMs, 1000));
  }
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function selectVisibleAssistantResponse(result, aggregateLogs) {
  if (!result?.blueprint) {
    return result?.final_response || "No assistant response returned.";
  }

  const completedTasks = result.blueprint.tasks?.filter((task) => task.is_completed) || [];
  const verificationLines = aggregateLogs
    .filter((entry) => entry.source === "system" && String(entry.message).startsWith("Verification "))
    .map((entry) => `- ${entry.message}`);

  const sections = [];
  sections.push("Completed execution plan:");
  sections.push(...completedTasks.map((task) => `- [x] ${task.description}`));

  if (verificationLines.length) {
    sections.push("");
    sections.push("Verification:");
    sections.push(...verificationLines);
  }

  if (Array.isArray(result.stage_traces) && result.stage_traces.length) {
    sections.push("");
    sections.push("Pipeline:");
    sections.push(...result.stage_traces.map((trace) => `- ${trace.stage}: ${trace.summary}`));
  }

  if (!result.blueprint.verification_passed) {
    sections.push("");
    sections.push("Some verification checks still need attention.");
  }

  if (!completedTasks.length && result.final_response) {
    return result.final_response;
  }

  return sections.join("\n");
}
