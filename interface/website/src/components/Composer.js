import React from "react";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel, formatDuration } from "../utils/format.js";
import { ToolPicker } from "./ToolPicker.js?v=popup3";
import { validatePlanBlueprint } from "../utils/validation.js";
import { buildPlanModePrompt, extractPlanBlueprint, READ_ONLY_PLAN_TOOLS, shouldDisplayPlanResult } from "../utils/plans.js";
import { BeamFrame, MetalSurface, MotionReveal, MotionShimmerText, MotionStack, MotionSwap } from "./MotionPrimitives.js";

export function Composer() {
  const { state, dispatch } = useApp();
  const textareaRef = React.useRef(null);
  const isCoolingDown = Boolean(state.rateLimitInfo && state.rateLimitInfo.resetAt > state.clock);
  const isBudgetBlocked = Boolean(state.sessionBudgetTokens && state.sessionUsageTotal >= state.sessionBudgetTokens);
  const isDisabled = isCoolingDown || isBudgetBlocked;
  const pendingThinking = [...state.transcript].reverse().find((entry) => entry.role === "thinking" && entry.pending);
  const replyTarget = state.replyTarget;
  const pendingRunMode = state.isRunning
    ? state.pendingRunMode
    : inferPendingRunMode({
        prompt: state.prompt,
        selectedTools: state.selectedTools,
        planMode: state.planMode,
      });

  const placeholder = isCoolingDown
    ? `Cooldown active. Input unlocks in ${formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))}.`
    : isBudgetBlocked
      ? "Session budget reached. Increase the limit in the right rail to continue."
      : "Ask Devenv...";

  const handleInput = (e) => {
    dispatch({ type: "SET_PROMPT", payload: e.target.value });
    autosizeComposer(e.target);
  };

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const originalPrompt = state.prompt.trim();
    if (!originalPrompt || state.isRunning || isCoolingDown || isBudgetBlocked) return;
    const requestPrompt = buildRuntimePrompt(originalPrompt, replyTarget);
    const userEntry = {
      id: `user-${Date.now()}`,
      role: "user",
      content: originalPrompt,
      replyTo: replyTarget,
    };

    dispatch({ type: "SET_IS_RUNNING", payload: true });
    dispatch({ type: "SET_RUN_STARTED_AT", payload: Date.now() });
    dispatch({
      type: "SET_PENDING_RUN_MODE",
      payload: inferPendingRunMode({
        prompt: originalPrompt,
        selectedTools: state.selectedTools,
        planMode: state.planMode,
      }),
    });
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
    dispatch({ type: "SET_PROMPT", payload: "" });
    dispatch({ type: "SET_REPLY_TARGET", payload: null });

    const thinkingId = `thinking-${Date.now()}`;
    const pendingRunMode = inferPendingRunMode({
      prompt: originalPrompt,
      selectedTools: state.selectedTools,
      planMode: state.planMode,
    });
    const pendingLogs = pendingRunMode === "web"
      ? [
          { source: "tool_call", message: "tool: web_search" },
          { source: "web_search", message: `query: ${originalPrompt}` },
          { source: "ai", message: "Checking live sources for the latest answer" },
        ]
      : pendingRunMode === "knowledge"
        ? [
            { source: "tool_call", message: "tool: knowledge_search" },
            { source: "knowledge_search", message: `query: ${originalPrompt}` },
            { source: "ai", message: "Looking for repos, videos, docs, and threads" },
          ]
      : [
          { source: "system", message: "Checking Devenv memory" },
          { source: "ai", message: "Looking for prior session matches" },
        ];
    if (replyTarget?.excerpt) {
      pendingLogs.unshift({ source: "reply", message: `focus: ${replyTarget.author}: ${replyTarget.excerpt}` });
    }

    dispatch({ type: "APPEND_TRANSCRIPT", payload: userEntry });
    dispatch({
      type: "APPEND_TRANSCRIPT",
      payload: { id: thinkingId, role: "thinking", content: formatThinkingBlock(pendingLogs), pending: true },
    });

    try {
      const { runPlan, runTurn } = await import("../api.js");
      let result = null;
      let planValidationError = null;
      const planOnlyMode = Boolean(state.planMode);

      while (true) {
        try {
          result = planOnlyMode
            ? await runPlan({
                prompt: buildPlanModePrompt(requestPrompt),
                selectedTools: READ_ONLY_PLAN_TOOLS,
                backendPreference: state.preferredBackend || "opencode",
              })
            : await runTurn({
                prompt: requestPrompt,
                planningMode: "auto",
                selectedTools: state.selectedTools,
                backendPreference: state.preferredBackend || "opencode",
                sessionBudgetTokens: state.sessionBudgetTokens,
              });
          break;
        } catch (error) {
          const parsedRateLimit = parseRateLimitError(error.message);
          if (!parsedRateLimit) throw error;
          dispatch({ type: "SET_RATE_LIMIT_INFO", payload: parsedRateLimit });
          updateThinking(thinkingId, dispatch, [
            { source: "system", message: "Rate limit reached" },
            { source: "ai", message: `Retrying in ${formatDuration(parsedRateLimit.retryMs)}` },
          ]);
          await waitForCooldown(parsedRateLimit.resetAt, (remainingMs) => {
            updateThinking(thinkingId, dispatch, [
              { source: "system", message: "Rate limit reached" },
              { source: "ai", message: `Retrying in ${formatDuration(remainingMs)}` },
            ]);
          });
        }
      }

      const shouldShowPlan = shouldDisplayPlanResult(result, planOnlyMode);
      const planBlueprint = shouldShowPlan ? extractPlanBlueprint(result, planOnlyMode) : null;

      if (shouldShowPlan && planBlueprint && (planBlueprint.tasks || planBlueprint.nodes)) {
        const validation = validatePlanBlueprint(planBlueprint);
        planValidationError = validation.valid ? null : validation.error;
      } else if (shouldShowPlan) {
        planValidationError = "No valid plan JSON was returned.";
      }

      const turnTokens = Number(result.total_usage?.total_tokens || 0);
      dispatch({ type: "SET_LATEST_TURN_TOKENS", payload: turnTokens });
      dispatch({ type: "SET_LATEST_ELAPSED_MS", payload: Number(result.elapsed_ms || Date.now() - state.runStartedAt) });
      dispatch({ type: "SET_ACTIVE_BACKEND", payload: result.backend_used || result.metadata?.backend_used || state.activeBackend });

      const now = Date.now();
      dispatch({
        type: "SET_USAGE_WINDOW",
        payload: [...state.usageWindow, { timestamp: now, totalTokens: turnTokens }].filter((e) => now - e.timestamp < 60000),
      });

      dispatch({
        type: "SET_RETRIEVAL_STATUS",
        payload: buildRetrievalStatus(result.metadata || {}),
      });

      const budgetState = result.metadata?.budget_state || null;
      if (budgetState) {
        dispatch({ type: "SET_SESSION_USAGE_TOTAL", payload: Number(budgetState.used || state.sessionUsageTotal) });
      } else {
        dispatch({ type: "SET_SESSION_USAGE_TOTAL", payload: state.sessionUsageTotal + turnTokens });
      }

      dispatch({
        type: "UPDATE_TRANSCRIPT_ENTRY",
        payload: { id: thinkingId, updates: { content: formatThinkingFromResult(result), pending: false } },
      });

      if (shouldShowPlan && planBlueprint && planValidationError === null) {
        dispatch({ type: "SET_PLAN_BLUEPRINT", payload: planBlueprint });
        dispatch({
          type: "APPEND_TRANSCRIPT",
          payload: {
            id: `plan-${Date.now()}`,
            role: "plan",
            content: "",
            blueprint: planBlueprint,
            mode: planOnlyMode ? "forced" : "auto",
          },
        });
      } else if (shouldShowPlan && planValidationError) {
        dispatch({ type: "SET_PLAN_BLUEPRINT", payload: null });
        dispatch({
          type: "APPEND_TRANSCRIPT",
          payload: {
            id: `plan-error-${Date.now()}`,
            role: "error",
            content: `Plan mode expected a valid multi-node flowchart JSON response, but the UI could not render it.\n\nLast error: ${planValidationError}`,
            diagnostics: buildMessageDiagnostics({
              result,
              pendingRunMode,
              planOnlyMode,
              overrideSourceLabel: "Plan render issue",
              overrideDetail: planValidationError,
            }),
          },
        });
      }

      const visibleAssistantResponse = planOnlyMode && shouldShowPlan && planValidationError === null
        ? `Generated an execution plan for: ${originalPrompt}`
        : selectVisibleAssistantResponse(result);

      if (visibleAssistantResponse) {
        dispatch({
          type: "APPEND_TRANSCRIPT",
          payload: {
            id: `assistant-${Date.now()}`,
            role: result?.error_message ? "error" : "assistant",
            content: visibleAssistantResponse,
            diagnostics: buildMessageDiagnostics({
              result,
              pendingRunMode,
              planOnlyMode,
              overrideSourceLabel: result?.error_message ? "Runtime issue" : null,
              overrideDetail: result?.error_message || null,
            }),
          },
        });
      }

      dispatch({ type: "SET_RATE_LIMIT_INFO", payload: null });
      if (budgetState?.blocked) {
        const { showToast } = await import("./Header.js");
        showToast(dispatch, "Session budget reached");
      }
    } catch (error) {
      const parsedRateLimit = parseRateLimitError(error.message);
      updateThinking(thinkingId, dispatch, [
        { source: "system", message: "Memory retrieval failed" },
        { source: "error", message: error.message },
      ]);
      dispatch({
        type: "APPEND_TRANSCRIPT",
        payload: {
          id: `assistant-${Date.now()}`,
          role: parsedRateLimit ? "error" : "assistant",
          content: parsedRateLimit ? "Rate limit reached while checking Devenv memory." : `Request failed: ${error.message}`,
          diagnostics: buildFailureDiagnostics({
            error,
            pendingRunMode,
            planOnlyMode: Boolean(state.planMode),
            preferredBackend: state.preferredBackend || state.activeBackend || "opencode",
            parsedRateLimit,
          }),
        },
      });
      if (parsedRateLimit) {
        dispatch({ type: "SET_RATE_LIMIT_INFO", payload: parsedRateLimit });
      }
    } finally {
      dispatch({ type: "SET_IS_RUNNING", payload: false });
      dispatch({ type: "SET_PENDING_RUN_MODE", payload: "direct" });
    }
  };

  React.useEffect(() => {
    if (textareaRef.current) {
      autosizeComposer(textareaRef.current);
    }
  }, [state.prompt]);

  React.useEffect(() => {
    const handler = () => {
      if (!textareaRef.current) return;
      textareaRef.current.focus();
      const length = textareaRef.current.value.length;
      textareaRef.current.setSelectionRange(length, length);
    };
    window.addEventListener("opencode-suggestion", handler);
    return () => window.removeEventListener("opencode-suggestion", handler);
  }, []);

  return React.createElement(
    "form",
    {
      className: "composer-shell p-margin-desktop bg-surface-container-low border-t border-outline-variant",
      onSubmit: handleSubmit,
    },
    React.createElement(
      "div",
      { className: "max-w-4xl mx-auto flex flex-col gap-3" },
      React.createElement(
        BeamFrame,
        {
          active: state.isRunning,
          tone: pendingThinking ? "ocean" : "mono",
          className: "composer-frame relative inset-terminal rounded-[26px] border border-outline-variant p-4",
        },
        React.createElement("span", { className: "composer-ribbon", "aria-hidden": "true" }),
        React.createElement("div", { className: "composer-backdrop composer-backdrop-one", "aria-hidden": "true" }),
        React.createElement("div", { className: "composer-backdrop composer-backdrop-two", "aria-hidden": "true" }),
        React.createElement(
          "div",
          { className: "composer-topline" },
          React.createElement(
            "div",
            { className: "composer-topline-copy" },
            React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, state.planMode ? "Plan-first" : "Live prompt"),
            React.createElement(
              MotionShimmerText,
              { active: state.isRunning, className: "composer-topline-detail text-on-surface-variant" },
              describeComposerState(state, { isCoolingDown, isBudgetBlocked })
            )
          ),
          React.createElement(
            "div",
            { className: "composer-topline-pills" },
            React.createElement("span", { className: "composer-pill" }, formatBackendLabel(state.preferredBackend || "opencode")),
            React.createElement("span", { className: "composer-pill" }, describeRouteChip(state)),
            React.createElement("span", { className: "composer-pill" }, state.planMode ? "plan mode" : "direct/auto")
          )
        ),
        replyTarget
          ? React.createElement(
              MotionReveal,
              { className: "mb-3" },
              React.createElement(
                MetalSurface,
                { className: "composer-reply flex items-start gap-3 rounded-2xl border border-primary/30 bg-surface-container px-3 py-2" },
                React.createElement("span", { className: "material-symbols-outlined text-primary text-[16px] mt-0.5" }, "reply"),
                React.createElement(
                  "div",
                  { className: "min-w-0 flex-1" },
                  React.createElement("div", { className: "font-label-caps text-label-caps text-primary" }, `Replying to ${replyTarget.author}`),
                  React.createElement("div", { className: "truncate text-[12px] text-on-surface-variant" }, replyTarget.excerpt)
                ),
                React.createElement(
                  "button",
                  {
                    type: "button",
                    className: "rounded-full p-1 text-on-surface-variant transition-colors hover:bg-surface-container-high hover:text-on-surface",
                    onClick: () => dispatch({ type: "SET_REPLY_TARGET", payload: null }),
                    title: "Clear reply",
                  },
                  React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "close")
                )
              )
            )
          : null,
        React.createElement(
          MotionStack,
          { className: "composer-signal-row" },
          React.createElement("span", { className: "composer-signal-dot" }),
          React.createElement("span", { className: "composer-signal-dot" }),
          React.createElement("span", { className: "composer-signal-dot" })
        ),
        React.createElement(
          "div",
          { className: "composer-input-shell" },
          React.createElement("div", { className: "composer-input-orbit composer-input-orbit-one", "aria-hidden": "true" }),
          React.createElement("div", { className: "composer-input-orbit composer-input-orbit-two", "aria-hidden": "true" }),
          React.createElement("textarea", {
            ref: textareaRef,
            className: "composer-input w-full bg-transparent border-none focus:ring-0 font-body-md text-body-md text-on-surface resize-none h-20 placeholder:text-outline outline-none",
            placeholder,
            disabled: isDisabled,
            value: state.prompt,
            onChange: handleInput,
            onKeyDown: handleKeyDown,
          }),
          React.createElement(
            "div",
            { className: "composer-input-meta" },
            React.createElement("span", { className: "composer-input-meta-pill" }, state.planMode ? "Blueprint only" : "Live runtime"),
            React.createElement("span", { className: "composer-input-meta-pill" }, state.selectedTools.length ? `${state.selectedTools.length} route${state.selectedTools.length === 1 ? "" : "s"}` : "Auto route"),
            React.createElement("span", { className: "composer-input-meta-pill" }, `${state.prompt.trim().length} chars`)
          )
        ),
        React.createElement(
          MotionSwap,
          { className: "composer-route-strip" },
          React.createElement("span", { className: "composer-route-strip-label" }, state.planMode ? "Plan lane" : "Run lane"),
          React.createElement("span", { className: "composer-route-strip-copy" }, state.selectedTools.length ? describeRouteChip(state) : "Let Devenv decide between memory, tools, and live search.")
        ),
        React.createElement(
          "div",
          { className: "composer-toolbar flex justify-between items-center mt-2 pt-2 border-t border-outline-variant/30" },
          React.createElement(
            "div",
            { className: "composer-toolbar-left flex items-center gap-2" },
            React.createElement(ToolPicker, null),
            React.createElement(
              "div",
              { className: "composer-hint text-on-surface-variant" },
              "Enter to type, Cmd/Ctrl+Enter to run"
            )
          ),
          React.createElement(
            "button",
            {
              type: "submit",
              className: "composer-submit px-6 py-2 rounded-full font-label-caps text-label-caps font-bold",
              disabled: state.isRunning || isDisabled || !state.prompt.trim(),
            },
            state.isRunning
              ? React.createElement(
                  MotionSwap,
                  { className: "items-center gap-2" },
                  React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, pendingRunMode === "knowledge" ? "hub" : pendingRunMode === "web" ? "public" : "bolt"),
                  React.createElement(MotionShimmerText, { className: "composer-submit-copy" }, runningVerbForMode(pendingRunMode))
                )
              : isCoolingDown
                ? formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))
                : isBudgetBlocked
                  ? "Blocked"
                  : "Ask"
          )
        ),
        state.isRunning && pendingThinking
          ? React.createElement(MotionReveal, { className: "mt-2" }, renderRunningTicker(dispatch, state, pendingThinking))
          : null
      )
    )
  );
}

function updateThinking(thinkingId, dispatch, logs) {
  dispatch({
    type: "UPDATE_TRANSCRIPT_ENTRY",
    payload: {
      id: thinkingId,
      updates: {
        content: logs.map((e) => `${String(e.source).toUpperCase()}  ${e.message}`).join("\n"),
      },
    },
  });
}

function formatThinkingBlock(entries) {
  return ["```text", ...entries.map((entry) => `${String(entry.source).toUpperCase()}  ${entry.message}`), "```"].join("\n");
}

function parseRateLimitError(message) {
  const limitMatch = message.match(/Limit\s+(\d+)/i);
  const usedMatch = message.match(/Used\s+(\d+)/i);
  const requestedMatch = message.match(/Requested\s+(\d+)/i);
  const retryMatch = message.match(/try again in\s+([\d.]+)s/i);
  if (!limitMatch || !usedMatch || !requestedMatch || !retryMatch) return null;
  const retryMs = Math.ceil(Number(retryMatch[1]) * 1000);
  return { limit: Number(limitMatch[1]), used: Number(usedMatch[1]), requested: Number(requestedMatch[1]), retryMs, resetAt: Date.now() + retryMs };
}

async function waitForCooldown(resetAt, onTick) {
  while (true) {
    const remainingMs = Math.max(resetAt - Date.now(), 0);
    onTick(remainingMs);
    if (remainingMs <= 0) return;
    await sleep(Math.min(remainingMs, 1000));
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function inferPendingRunMode({ prompt, selectedTools = [], planMode = false }) {
  if (planMode) return "plan";
  const toolSet = new Set(Array.isArray(selectedTools) ? selectedTools : []);
  if (toolSet.has("knowledge_search")) return "knowledge";
  if (toolSet.has("web_search")) return "web";
  if (toolSet.size) return "direct";
  const lowered = String(prompt || "").toLowerCase();
  const webMarkers = ["today", "latest", "current", "currently", "recent", "president", "prime minister", "ceo", "who is"];
  const knowledgeMarkers = ["github", "repo", "reference", "references", "youtube", "reddit", "stackoverflow", "quora", "similar project", "find examples"];
  if (knowledgeMarkers.some((marker) => lowered.includes(marker))) return "knowledge";
  if (webMarkers.some((marker) => lowered.includes(marker))) return "web";
  return "direct";
}

function formatThinkingFromResult(result) {
  const lines = [];
  const metadata = result.metadata || {};
  if (metadata.external_context_state === "privacy_blocked") {
    lines.push({ source: "system", message: "Privacy mode blocked prior memory for this turn" });
  }
  const toolSteps = Array.isArray(result.steps) ? result.steps : [];
  if (!toolSteps.length && Array.isArray(result.stage_traces) && result.stage_traces.length) {
    for (const trace of result.stage_traces.slice(0, 5)) {
      const summary = summarizeStageTrace(trace);
      if (summary) lines.push(summary);
      const traceLogs = Array.isArray(trace.logs) ? trace.logs : [];
      for (const log of traceLogs.slice(0, 2)) {
        const normalizedLog = summarizeTraceLog(trace.stage, log);
        if (normalizedLog) lines.push(normalizedLog);
      }
    }
  }
  for (const step of toolSteps.slice(0, 5)) {
    if (step.tool_name) {
      const args = step.arguments || {};
      const path = typeof args.path === "string" ? ` (${args.path})` : "";
      lines.push({ source: "tool_call", message: `tool: ${step.tool_name}${path}` });
      if (step.tool_name === "web_search" && typeof step.data?.query === "string") {
        lines.push({ source: "web_search", message: `query: ${step.data.query}` });
        const results = Array.isArray(step.data?.results) ? step.data.results : [];
        for (const item of results.slice(0, 5)) {
          lines.push({ source: "web_search", message: `result: ${String(item.title || item.url || "Result")} - ${String(item.url || "").trim()}` });
        }
      }
      if (step.tool_name === "knowledge_search") {
        if (typeof step.data?.query === "string") {
          lines.push({ source: "knowledge_search", message: `query: ${step.data.query}` });
        }
        const groups = Array.isArray(step.data?.resources) ? step.data.resources : extractKnowledgeResources(step.output);
        lines.push({ source: "trace", message: summarizeKnowledgeResources(groups) });
        for (const group of groups.slice(0, 7)) {
          if (!group || typeof group !== "object") continue;
          const source = String(group.source || "general").trim() || "general";
          const query = String(group.query || "").trim();
          lines.push({ source: "knowledge_search", message: `source: ${JSON.stringify({ source, query })}` });
          const results = Array.isArray(group.results) ? group.results : [];
          for (const item of results.slice(0, 5)) {
            lines.push({
              source: "knowledge_search",
              message: `result: ${JSON.stringify({ source, title: String(item.title || item.url || "Result"), url: String(item.url || ""), query: String(item.query || query || "") })}`,
            });
          }
        }
      }
    }
    if (step.output) lines.push({ source: "trace", message: step.output.split("\n")[0] });
  }
  if (!lines.length) {
    lines.push({ source: "ai", message: result.final_response ? "Finished shaping the response" : "Checked Devenv context" });
  }
  return formatThinkingBlock(lines);
}

function summarizeStageTrace(trace) {
  if (!trace || typeof trace !== "object") return null;
  const stage = String(trace.stage || "").trim().toLowerCase();
  const payload = trace.payload && typeof trace.payload === "object" ? trace.payload : {};
  if (stage === "checkpoint_creation") {
    const checkpointCount = Number(payload.checkpoint_count || 0);
    const continued = Boolean(payload.continued);
    return {
      source: "ai",
      message: continued
        ? checkpointCount > 0
          ? `Reused the active ${checkpointCount}-step execution plan`
          : "Reused the active execution plan"
        : checkpointCount > 0
          ? `Mapped the request into ${checkpointCount} execution step${checkpointCount === 1 ? "" : "s"}`
          : "Mapped the request into an execution plan",
    };
  }
  if (stage === "context_memory") {
    const workspaceFacts = Number(payload.workspace_fact_count || 0);
    return {
      source: "ai",
      message: workspaceFacts > 0 ? "Built a grounded context packet from memory and workspace facts" : "Built a distilled context packet",
    };
  }
  if (stage === "metadata") {
    const touched = Array.isArray(payload.files_touched) ? payload.files_touched : [];
    const destination = String(payload.output_destination || "").trim();
    return {
      source: "ai",
      message: touched.length
        ? `Recorded runtime output for ${touched.length} touched file${touched.length === 1 ? "" : "s"}`
        : destination
          ? `Recorded runtime output for ${destination.replaceAll("_", " ")}`
          : "Recorded runtime output details",
    };
  }
  if (stage === "verification") {
    return {
      source: "ai",
      message: trace.success === false ? "Verification flagged an issue" : "Verified the runtime result",
    };
  }
  if (stage === "brain") {
    return {
      source: "ai",
      message: "Executed the active checkpoint",
    };
  }
  if (typeof trace.summary === "string" && trace.summary.trim()) {
    return { source: "ai", message: trace.summary.trim() };
  }
  return null;
}

function summarizeTraceLog(stage, log) {
  const text = String(log || "").trim();
  if (!text) return null;
  const normalizedStage = String(stage || "").trim().toLowerCase();
  if (normalizedStage === "metadata") {
    if (/^Output destination:/i.test(text)) {
      return { source: "trace", message: text.replace(/^Output destination:\s*/i, "Answer destination: ") };
    }
    if (/^Touched files:/i.test(text)) {
      return { source: "trace", message: text.replace(/^Touched files:\s*/i, "Touched files recorded: ") };
    }
  }
  if (normalizedStage === "context_memory") {
    if (/^Checkpoint objective:/i.test(text)) {
      return { source: "trace", message: text.replace(/^Checkpoint objective:\s*/i, "Focused on: ") };
    }
    if (/workspace scan/i.test(text)) {
      return { source: "trace", message: "Scanned the workspace to ground the current step" };
    }
  }
  if (normalizedStage === "checkpoint_creation" && /^Checkpoint count:/i.test(text)) {
    return { source: "trace", message: text.replace(/^Checkpoint count:\s*/i, "Planned checkpoints: ") };
  }
  if (normalizedStage === "verification" && /non-empty answer returned/i.test(text)) {
    return { source: "trace", message: "Verification confirmed the response was non-empty" };
  }
  return { source: "trace", message: text };
}

function extractKnowledgeResources(output) {
  const payload = extractTrailingJsonObject(output);
  return Array.isArray(payload?.resources) ? payload.resources : [];
}

function extractTrailingJsonObject(output) {
  const text = String(output || "");
  const start = text.lastIndexOf("\n{");
  const candidate = start >= 0 ? text.slice(start + 1).trim() : text.trim();
  if (!candidate.startsWith("{")) return null;
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

function summarizeKnowledgeResources(groups) {
  const validGroups = Array.isArray(groups) ? groups.filter((group) => group && typeof group === "object") : [];
  const resultCount = validGroups.reduce((total, group) => total + (Array.isArray(group.results) ? group.results.length : 0), 0);
  const sources = validGroups
    .filter((group) => Array.isArray(group.results) && group.results.length > 0)
    .map((group) => String(group.source || "general"));
  return resultCount
    ? `Collected ${resultCount} reference result${resultCount === 1 ? "" : "s"} from ${sources.join(", ")}`
    : "No reference results were found in the selected sources";
}

function selectVisibleAssistantResponse(result) {
  const text = String(result?.final_response || "").trim();
  if (text) return text;
  const errText = String(result?.error_message || "").trim();
  if (errText) return errText;
  return "No memory answer was returned.";
}

function buildRetrievalStatus(metadata) {
  if (metadata.external_context_state === "reused_prior_context") {
    const count = Number(metadata.external_context_session_count || 0);
    return {
      mode: "reused_prior_context",
      label: count > 0 ? `Reused prior context${count > 1 ? ` (${count})` : ""}` : "Reused prior context",
      detail: metadata.external_context_reason || "A prior Devenv session matched this request.",
    };
  }
  return {
    mode: "new_context",
    label: "New context",
    detail: metadata.external_context_reason || "No strong prior Devenv session match was found.",
  };
}

function buildMessageDiagnostics({
  result,
  pendingRunMode,
  planOnlyMode,
  overrideSourceLabel = null,
  overrideDetail = null,
}) {
  const metadata = result?.metadata || {};
  const retrieval = buildRetrievalStatus(metadata);
  const steps = Array.isArray(result?.steps) ? result.steps : [];
  const backendRaw = result?.backend_used || metadata.backend_used || result?.backend || "local";
  const backendLabel = backendRaw === "local" ? "Local runtime" : formatBackendLabel(backendRaw);
  const hasWeb = pendingRunMode === "web" || steps.some((step) => step?.tool_name === "web_search");
  const hasKnowledge = pendingRunMode === "knowledge" || steps.some((step) => step?.tool_name === "knowledge_search");
  const toolCount = steps.filter((step) => step?.tool_name).length;
  const localRuntime = String(metadata.backend_used || backendRaw) === "local";
  const evidenceItems = extractEvidenceItems(steps);

  let routeLabel = planOnlyMode ? "Plan mode" : "Direct route";
  let sourceLabel = "Direct answer";
  let detail = overrideDetail || retrieval.detail || "";

  if (planOnlyMode) {
    routeLabel = "Plan mode";
    sourceLabel = "Execution plan";
    detail = overrideDetail || "This turn stayed in plan mode and returned a renderable execution flow.";
  } else if (hasWeb) {
    routeLabel = "Web route";
    sourceLabel = "Live web";
    detail = overrideDetail || "This answer used fetched live-source results instead of relying on stale memory.";
  } else if (hasKnowledge) {
    routeLabel = "Knowledge route";
    sourceLabel = "Reference search";
    detail = overrideDetail || "This answer pulled external references such as repos, docs, or discussion threads.";
  } else if (metadata.external_context_state === "reused_prior_context") {
    routeLabel = "Memory route";
    sourceLabel = "Prior context";
    detail = overrideDetail || retrieval.detail || "A prior Devenv session was reused to answer this turn.";
  } else if (metadata.external_context_state === "privacy_blocked") {
    routeLabel = "Privacy route";
    sourceLabel = "Memory blocked";
    detail = overrideDetail || retrieval.detail || "Prior session memory was intentionally blocked for this turn.";
  } else if (localRuntime && toolCount > 0) {
    routeLabel = "Tool route";
    sourceLabel = "Workspace tools";
    detail = overrideDetail || "The runtime stayed local and answered from bounded workspace inspection.";
  } else if (localRuntime) {
    routeLabel = "Local route";
    sourceLabel = "Local runtime";
    detail = overrideDetail || "The runtime answered locally without handing the turn to a remote backend.";
  }

  return {
    badgeLabel: planOnlyMode ? "Plan" : overrideSourceLabel ? "Issue" : "Answer",
    kicker: overrideSourceLabel ? "runtime trace" : sourceLabel.toLowerCase(),
    routeLabel,
    backendLabel,
    sourceLabel: overrideSourceLabel || sourceLabel,
    toolLabel: toolCount ? `${toolCount} tool step${toolCount === 1 ? "" : "s"}` : "",
    retrievalLabel: retrieval.label || "",
    detail,
    evidenceItems,
  };
}

function buildFailureDiagnostics({
  error,
  pendingRunMode,
  planOnlyMode,
  preferredBackend,
  parsedRateLimit,
}) {
  const routeLabel = planOnlyMode ? "Plan mode" : pendingRunMode === "web" ? "Web route" : pendingRunMode === "knowledge" ? "Knowledge route" : "Direct route";
  const backendLabel = preferredBackend === "local" ? "Local runtime" : formatBackendLabel(preferredBackend || "opencode");
  return {
    badgeLabel: "Issue",
    kicker: "runtime trace",
    routeLabel,
    backendLabel,
    sourceLabel: parsedRateLimit ? "Rate limited" : "Runtime failure",
    toolLabel: "",
    retrievalLabel: "",
    detail: parsedRateLimit ? "The backend hit a rate limit before the turn could finish." : String(error?.message || "The request failed before a normal answer was produced."),
    evidenceItems: [],
  };
}

function extractEvidenceItems(steps) {
  const items = [];
  for (const step of Array.isArray(steps) ? steps : []) {
    if (!step || typeof step !== "object") continue;
    if (step.tool_name === "web_search") {
      const results = Array.isArray(step.data?.results) ? step.data.results : [];
      for (const result of results.slice(0, 3)) {
        const title = String(result?.title || result?.url || "Web result").trim();
        const url = String(result?.url || "").trim();
        if (!title || !url) continue;
        items.push({ kind: "web", label: "Web", title, url, meta: "" });
      }
    }
    if (step.tool_name === "knowledge_search") {
      const resources = Array.isArray(step.data?.resources) ? step.data.resources : [];
      for (const group of resources.slice(0, 3)) {
        const source = String(group?.source || "Reference").trim();
        const results = Array.isArray(group?.results) ? group.results : [];
        for (const result of results.slice(0, 2)) {
          const title = String(result?.title || result?.url || "Reference").trim();
          const url = String(result?.url || "").trim();
          if (!title || !url) continue;
          items.push({ kind: "knowledge", label: source, title, url, meta: "" });
        }
      }
    }
    if (items.length >= 4) break;
  }
  return items.slice(0, 4);
}

const RUNNING_STATUS_FRAMES = [
  "Scanning stored sessions",
  "Matching prior projects",
  "Collecting relevant details",
  "Drafting the memory answer",
];

const WEB_RUNNING_STATUS_FRAMES = [
  "Searching the web",
  "Checking live sources",
  "Reading the relevant page",
  "Summarizing the result",
];

const KNOWLEDGE_RUNNING_STATUS_FRAMES = [
  "Searching GitHub and the web",
  "Collecting repos and references",
  "Grouping videos, docs, and threads",
  "Preparing the research summary",
];

function renderRunningTicker(dispatch, state, pendingThinking) {
  void dispatch;
  const clock = Date.now();
  const content = String(pendingThinking.content || "");
  const useKnowledge = state.pendingRunMode === "knowledge" || /knowledge_search|source:/i.test(content);
  const useGlobe = state.pendingRunMode === "web" || /query:|result:|searching the web/i.test(content);
  const steps = parseThinkingText(content);
  let frame;
  if (steps.length) {
    const recent = Array.from(new Set(steps.slice(-4)));
    const index = Math.floor(clock / 1600) % recent.length;
    frame = recent[index];
  } else {
    const frames = useKnowledge ? KNOWLEDGE_RUNNING_STATUS_FRAMES : state.pendingRunMode === "web" ? WEB_RUNNING_STATUS_FRAMES : RUNNING_STATUS_FRAMES;
    frame = frames[Math.floor(clock / 1200) % frames.length];
  }
  return React.createElement(
    MetalSurface,
    { className: "composer-running inline-flex items-center gap-2 px-4 py-2 rounded-full border border-outline-variant" },
    React.createElement("span", { className: `material-symbols-outlined text-primary text-[16px] animate-pulse` }, useKnowledge ? "hub" : useGlobe ? "public" : "bolt"),
    React.createElement("span", { className: "font-body-md text-body-md text-on-surface" }, frame),
    React.createElement(
      "span",
      { className: "inline-flex gap-1" },
      React.createElement("span", { className: "composer-running-dot animate-bounce", style: { animationDelay: "0s" } }),
      React.createElement("span", { className: "composer-running-dot animate-bounce", style: { animationDelay: "0.18s" } }),
      React.createElement("span", { className: "composer-running-dot animate-bounce", style: { animationDelay: "0.36s" } })
    )
  );
}

function parseThinkingText(content) {
  return String(content || "")
    .split("\n")
    .filter((l) => l && !l.startsWith("```"))
    .map((l) => l.replace(/^[A-Z_]+\s+/, "").trim())
    .filter(Boolean);
}

function buildRuntimePrompt(prompt, replyTarget) {
  if (!replyTarget?.excerpt) return prompt;
  return [
    "Reply context:",
    `You are replying to an earlier ${replyTarget.role} message from ${replyTarget.author}.`,
    `Quoted message: """${replyTarget.excerpt}"""`,
    "Use this replied message as the primary local context for the next answer.",
    "",
    "User request:",
    prompt,
  ].join("\n");
}

function autosizeComposer(textarea) {
  textarea.style.height = "0px";
  textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 72), 220)}px`;
}

function runningVerbForMode(mode) {
  if (mode === "knowledge") return "Researching";
  if (mode === "web") return "Searching";
  return "Thinking";
}

function describeComposerState(state, { isCoolingDown, isBudgetBlocked }) {
  if (isCoolingDown) return "Cooling down after a rate limit";
  if (isBudgetBlocked) return "Session budget reached";
  if (state.isRunning) return "Executing the current turn";
  if (state.planMode) return "Will inspect the repo and return a flowchart before execution";
  if (state.selectedTools.includes("track_symbol")) return "Biased toward tracing how a symbol moves through the codebase";
  if (state.selectedTools.includes("inspect_symbols")) return "Biased toward definitions, exports, and structural code inspection";
  if (state.selectedTools.includes("search_text")) return "Biased toward repo-wide string and usage search";
  if (state.selectedTools.includes("read_file")) return "Biased toward opening exact files before answering";
  if (state.selectedTools.includes("locate_files")) return "Biased toward finding the right files before deeper inspection";
  if (state.selectedTools.includes("list_directory")) return "Biased toward mapping folders and workspace structure";
  if (state.selectedTools.includes("knowledge_search")) return "Biased toward repos, docs, videos, and reference gathering";
  if (state.selectedTools.includes("web_search")) return "Biased toward live web results and current facts";
  if (state.selectedTools.includes("generate_pdf")) return "Biased toward producing a polished PDF artifact";
  if (state.selectedTools.includes("generate_prompt")) return "Biased toward generating a stronger prompt output";
  return "Auto-routes between memory, plan, tools, and live search";
}

function describeRouteChip(state) {
  if (state.planMode) return "repo plan";
  if (!state.selectedTools.length) return "auto route";
  if (state.selectedTools.length === 1) {
    const selected = state.selectedTools[0];
    if (selected === "list_directory") return "files route";
    if (selected === "locate_files") return "locate route";
    if (selected === "read_file") return "read route";
    if (selected === "search_text") return "search route";
    if (selected === "inspect_symbols") return "symbols route";
    if (selected === "track_symbol") return "trace route";
    if (selected === "knowledge_search") return "knowledge route";
    if (selected === "web_search") return "web route";
    if (selected === "generate_pdf") return "pdf route";
    if (selected === "generate_prompt") return "prompt route";
  }
  return `${state.selectedTools.length} routes`;
}
