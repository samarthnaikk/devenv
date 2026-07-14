import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { formatDuration } from "../utils/format.js";
import { ToolPicker } from "./ToolPicker.js";
import { validatePlanBlueprint } from "../utils/validation.js";
import { extractPlanBlueprint, READ_ONLY_PLAN_TOOLS, shouldDisplayPlanResult } from "../utils/plans.js";

export function Composer() {
  const { state, dispatch } = useApp();
  const textareaRef = React.useRef(null);
  const isCoolingDown = Boolean(state.rateLimitInfo && state.rateLimitInfo.resetAt > state.clock);
  const isBudgetBlocked = Boolean(state.sessionBudgetTokens && state.sessionUsageTotal >= state.sessionBudgetTokens);
  const isDisabled = isCoolingDown || isBudgetBlocked;
  const pendingThinking = [...state.transcript].reverse().find((entry) => entry.role === "thinking" && entry.pending);

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

    dispatch({ type: "SET_IS_RUNNING", payload: true });
    dispatch({ type: "SET_RUN_STARTED_AT", payload: Date.now() });
    dispatch({ type: "SET_PENDING_RUN_MODE", payload: inferPendingRunMode(originalPrompt) });
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
    dispatch({ type: "SET_PROMPT", payload: "" });

    const thinkingId = `thinking-${Date.now()}`;
    const pendingLogs = state.pendingRunMode === "web"
      ? [
          { source: "tool_call", message: "tool: web_search" },
          { source: "web_search", message: `query: ${originalPrompt}` },
          { source: "ai", message: "Checking live sources for the latest answer" },
        ]
      : [
          { source: "system", message: "Checking Devenv memory" },
          { source: "ai", message: "Looking for prior session matches" },
        ];

    dispatch({ type: "APPEND_TRANSCRIPT", payload: { id: `user-${Date.now()}`, role: "user", content: originalPrompt } });
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
                prompt: originalPrompt,
                selectedTools: READ_ONLY_PLAN_TOOLS,
                backendPreference: state.preferredBackend || "opencode",
              })
            : await runTurn({
                prompt: originalPrompt,
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
        },
      });
      if (parsedRateLimit) {
        dispatch({ type: "SET_RATE_LIMIT_INFO", payload: parsedRateLimit });
      }
    } finally {
      dispatch({ type: "SET_IS_RUNNING", payload: false });
      dispatch({ type: "SET_PENDING_RUN_MODE", payload: "memory" });
    }
  };

  React.useEffect(() => {
    if (textareaRef.current) {
      autosizeComposer(textareaRef.current);
    }
  }, [state.prompt]);

  return React.createElement(
    "form",
    {
      className: "p-margin-desktop bg-surface-container-low border-t border-outline-variant",
      onSubmit: handleSubmit,
    },
    React.createElement(
      "div",
      { className: "max-w-4xl mx-auto flex flex-col gap-3" },
      React.createElement(
        "div",
        { className: "relative inset-terminal rounded-xl border border-outline-variant p-4 focus-within:border-primary transition-all" },
        React.createElement("textarea", {
          ref: textareaRef,
          className: "w-full bg-transparent border-none focus:ring-0 font-body-md text-body-md text-on-surface resize-none h-20 placeholder:text-outline outline-none",
          placeholder,
          disabled: isDisabled,
          value: state.prompt,
          onChange: handleInput,
          onKeyDown: handleKeyDown,
        }),
        React.createElement(
          "div",
          { className: "flex justify-between items-center mt-2 pt-2 border-t border-outline-variant/30" },
          React.createElement(
            "div",
            { className: "flex items-center gap-2" },
            React.createElement(ToolPicker, null)
          ),
          React.createElement(
            "button",
            {
              type: "submit",
              className: "px-6 py-2 bg-primary text-on-primary rounded-full font-label-caps text-label-caps font-bold hover:opacity-90 transition-opacity",
              disabled: state.isRunning || isDisabled || !state.prompt.trim(),
            },
            state.isRunning
              ? `Searching${".".repeat((Math.floor(Date.now() / 350) % 3) + 1)}`
              : isCoolingDown
                ? formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))
                : isBudgetBlocked
                  ? "Blocked"
                  : "Ask"
          )
        ),
        state.isRunning && pendingThinking
          ? React.createElement("div", { className: "mt-2" }, renderRunningTicker(dispatch, state, pendingThinking))
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

function inferPendingRunMode(prompt) {
  const lowered = String(prompt || "").toLowerCase();
  const webMarkers = ["today", "latest", "current", "currently", "recent", "president", "prime minister", "ceo", "who is"];
  return webMarkers.some((marker) => lowered.includes(marker)) ? "web" : "memory";
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
      if (trace.summary) lines.push({ source: "ai", message: trace.summary });
    }
  }
  for (const step of toolSteps.slice(0, 5)) {
    if (step.tool_name) {
      const args = step.arguments || {};
      const path = typeof args.path === "string" ? ` (${args.path})` : "";
      lines.push({ source: "tool_call", message: `tool: ${step.tool_name}${path}` });
    }
  }
  if (!lines.length) {
    lines.push({ source: "ai", message: result.final_response ? "Prepared the final answer" : "Checked Devenv context" });
  }
  return formatThinkingBlock(lines);
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

function renderRunningTicker(dispatch, state, pendingThinking) {
  const clock = Date.now();
  const content = String(pendingThinking.content || "");
  const useGlobe = state.pendingRunMode === "web" || /query:|result:|searching the web/i.test(content);
  const steps = parseThinkingText(content);
  let frame;
  if (steps.length) {
    const recent = Array.from(new Set(steps.slice(-4)));
    const index = Math.floor(clock / 1600) % recent.length;
    frame = recent[index];
  } else {
    const frames = state.pendingRunMode === "web" ? WEB_RUNNING_STATUS_FRAMES : RUNNING_STATUS_FRAMES;
    frame = frames[Math.floor(clock / 1200) % frames.length];
  }
  return React.createElement(
    "span",
    { className: "inline-flex items-center gap-2 px-4 py-2 bg-surface-container rounded-full border border-outline-variant" },
    React.createElement("span", { className: `material-symbols-outlined text-primary text-[16px] animate-pulse` }, useGlobe ? "public" : "bolt"),
    React.createElement("span", { className: "font-body-md text-body-md text-on-surface" }, frame),
    React.createElement(
      "span",
      { className: "inline-flex gap-1" },
      React.createElement("span", { className: "w-1 h-1 rounded-full bg-on-surface/25 animate-bounce", style: { animationDelay: "0s" } }),
      React.createElement("span", { className: "w-1 h-1 rounded-full bg-on-surface/25 animate-bounce", style: { animationDelay: "0.18s" } }),
      React.createElement("span", { className: "w-1 h-1 rounded-full bg-on-surface/25 animate-bounce", style: { animationDelay: "0.36s" } })
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

function autosizeComposer(textarea) {
  textarea.style.height = "0px";
  textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 72), 220)}px`;
}
