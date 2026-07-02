const STORAGE_THEME_KEY = "devenv-ui-theme";

const state = {
  health: null,
  prompt: "",
  transcript: [],
  isRunning: false,
  bootError: "",
  healthMeta: { provider: "", model: "", availableModels: [] },
  blueprint: null,
  runtimeState: "PLANNING",
  stageTraces: [],
  verificationResults: [],
  usageWindow: [],
  rateLimitInfo: null,
  planningMode: "auto",
  localOnlyEnabled: false,
  showThinking: false,
  clock: Date.now(),
  theme: loadTheme(),
  toast: "",
};

const root = document.getElementById("root");
const SUGGESTIONS = [
  "Build a classic Snake game in this repo.",
  "Find and fix a bug in my code.",
  "Summarize this app in a one-page note.",
];
const REASONING_OPTIONS = [
  { value: "force_direct", label: "High" },
  { value: "auto", label: "Medium" },
  { value: "force_plan", label: "Max" },
];

let renderQueued = false;
let toastTimeoutId = null;

bootstrap();

async function bootstrap() {
  window.setInterval(() => {
    const nextClock = Date.now();
    const nextUsageWindow = state.usageWindow.filter((entry) => nextClock - entry.timestamp < 60000);
    const nextRateLimitInfo =
      state.rateLimitInfo && state.rateLimitInfo.resetAt > nextClock ? state.rateLimitInfo : null;

    if (
      nextClock !== state.clock &&
      (state.isRunning || nextRateLimitInfo || nextUsageWindow.length !== state.usageWindow.length)
    ) {
      state.clock = nextClock;
      state.usageWindow = nextUsageWindow;
      state.rateLimitInfo = nextRateLimitInfo;
      scheduleRender();
    } else {
      state.clock = nextClock;
    }
  }, 1000);

  bindEvents();
  scheduleRender();

  try {
    const healthPayload = await request("/api/health");
    state.health = healthPayload;
    state.healthMeta = {
      provider: healthPayload.ai_provider || "",
      model: healthPayload.ai_model || "",
      availableModels: healthPayload.available_models || [],
    };
  } catch (error) {
    state.bootError = error.message;
  }

  scheduleRender();
}

function bindEvents() {
  root.addEventListener("click", async (event) => {
    const suggestion = event.target.closest("[data-suggestion]");
    if (suggestion) {
      state.prompt = suggestion.getAttribute("data-suggestion") || "";
      scheduleRender({ focusComposer: true, moveCaretToEnd: true });
      return;
    }

    const submit = event.target.closest("[data-submit]");
    if (submit) {
      event.preventDefault();
      await submitPrompt();
      return;
    }

    const action = event.target.closest("[data-action]");
    if (action) {
      event.preventDefault();
      await handleAction(action.getAttribute("data-action"));
    }
  });

  root.addEventListener("input", (event) => {
    if (event.target.matches("[data-prompt-input]")) {
      state.prompt = event.target.value;
      syncComposerState();
      autosizeComposer(event.target);
      return;
    }
  });

  root.addEventListener("keydown", async (event) => {
    if (!event.target.matches("[data-prompt-input]")) {
      return;
    }
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      await submitPrompt();
    }
  });

  root.addEventListener("change", async (event) => {
    if (event.target.matches("[data-permissions-select]")) {
      state.localOnlyEnabled = event.target.value === "local";
      scheduleRender({ preserveComposerFocus: true });
      return;
    }

    if (event.target.matches("[data-model-select]")) {
      const payload = await request("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: event.target.value }),
      });
      state.healthMeta.model = payload.ai_model || event.target.value;
      state.healthMeta.availableModels = payload.available_models || state.healthMeta.availableModels;
      showToast("Model updated");
      scheduleRender({ preserveComposerFocus: true });
      return;
    }

    if (event.target.matches("[data-reasoning-select]")) {
      state.planningMode = event.target.value;
      scheduleRender({ preserveComposerFocus: true });
      return;
    }

    if (event.target.matches("[data-thinking-toggle]")) {
      state.showThinking = Boolean(event.target.checked);
      scheduleRender({ preserveComposerFocus: true });
    }
  });

  root.addEventListener("submit", async (event) => {
    if (event.target.matches("[data-composer-form]")) {
      event.preventDefault();
      await submitPrompt();
    }
  });
}

async function handleAction(action) {
  if (action === "theme") {
    state.theme = state.theme === "dark" ? "light" : "dark";
    persistTheme(state.theme);
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "new-thread") {
    state.prompt = "";
    state.transcript = [];
    state.blueprint = null;
    state.runtimeState = "PLANNING";
    state.stageTraces = [];
    state.verificationResults = [];
    showToast("Started a new thread");
    scheduleRender({ focusComposer: true });
    return;
  }

  if (action === "copy-thread") {
    if (!state.transcript.length) {
      showToast("Nothing to copy yet");
      return;
    }

    const transcriptText = state.transcript
      .filter((entry) => entry.role !== "thinking" || state.showThinking)
      .map((entry) => `${roleLabel(entry)}\n${String(entry.content || "").trim()}`)
      .join("\n\n");

    try {
      await navigator.clipboard.writeText(transcriptText);
      showToast("Thread copied");
    } catch (error) {
      showToast("Clipboard access failed");
    }
  }
}

async function submitPrompt() {
  const nextPrompt = state.prompt.trim();
  if (!nextPrompt || state.isRunning || isCoolingDown()) {
    return;
  }

  state.isRunning = true;
  state.prompt = "";
  const thinkingId = `thinking-${Date.now()}`;
  const pendingLogs = [
    createLogEntry("system", `Prompt submitted: ${nextPrompt}`),
    createLogEntry("ai", "Waiting for runtime response..."),
  ];
  state.transcript.push({ id: `user-${Date.now()}`, role: "user", content: nextPrompt });
  state.transcript.push({ id: thinkingId, role: "thinking", content: formatThinkingBlock(pendingLogs), pending: true });
  scheduleRender({ focusComposer: true });

  try {
    const aggregateLogs = [];
    let continuePlan = false;
    let autoContinueCount = 0;
    let result = null;

    do {
      while (true) {
        try {
          result = await request("/api/turn", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompt: nextPrompt,
              planning_mode: state.planningMode,
              continue_plan: continuePlan,
              local_only: state.localOnlyEnabled,
            }),
          });
          break;
        } catch (error) {
          const parsedRateLimit = parseRateLimitError(error.message);
          if (!parsedRateLimit) {
            throw error;
          }

          state.rateLimitInfo = parsedRateLimit;
          const retryEntry = createLogEntry(
            "error",
            `Rate limit reached. Retrying in ${formatDuration(parsedRateLimit.retryMs)}`
          );
          aggregateLogs.push(retryEntry);
          updateThinkingEntry(thinkingId, formatThinkingBlock(aggregateLogs), true);
          scheduleRender({ preserveComposerFocus: true });
          await waitForCooldown(parsedRateLimit.resetAt, (remainingMs) => {
            retryEntry.message = `Rate limit reached. Retrying in ${formatDuration(remainingMs)}`;
            updateThinkingEntry(thinkingId, formatThinkingBlock(aggregateLogs), true);
            scheduleRender({ preserveComposerFocus: true });
          });
          retryEntry.message = "Cooldown finished. Retrying request now.";
        }
      }

      aggregateLogs.push(...buildLogEntries(result));
      state.blueprint = result.blueprint || null;
      state.runtimeState = result.state || "PLANNING";
      state.stageTraces = result.stage_traces || [];
      state.verificationResults = result.verification_results || [];
      state.usageWindow = [...state.usageWindow, { timestamp: Date.now(), totalTokens: result.total_usage?.total_tokens || 0 }].filter(
        (entry) => Date.now() - entry.timestamp < 60000
      );
      continuePlan = shouldAutoContinue(result, autoContinueCount);
      autoContinueCount += continuePlan ? 1 : 0;
      updateThinkingEntry(thinkingId, formatThinkingBlock(aggregateLogs), continuePlan);
      scheduleRender({ preserveComposerFocus: true });
    } while (continuePlan);

    state.transcript = state.transcript.map((entry) =>
      entry.id === thinkingId ? { ...entry, content: formatThinkingBlock(aggregateLogs), pending: false } : entry
    );
    state.transcript.push({
      id: `assistant-${Date.now()}`,
      role: result?.error_message ? "error" : "assistant",
      content: result?.error_message || selectVisibleAssistantResponse(result, aggregateLogs),
    });
    state.rateLimitInfo = null;
  } catch (error) {
    const parsedRateLimit = parseRateLimitError(error.message);
    state.transcript = state.transcript.map((entry) =>
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
    );
    state.transcript.push({
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
    });
    if (parsedRateLimit) {
      state.rateLimitInfo = parsedRateLimit;
    }
  } finally {
    state.isRunning = false;
    scheduleRender({ focusComposer: true });
  }
}

function updateThinkingEntry(thinkingId, content, pending) {
  state.transcript = state.transcript.map((entry) => (entry.id === thinkingId ? { ...entry, content, pending } : entry));
}

function scheduleRender(options = {}) {
  state.__renderOptions = {
    ...(state.__renderOptions || {}),
    ...options,
  };
  if (renderQueued) {
    return;
  }
  renderQueued = true;
  window.requestAnimationFrame(() => {
    renderQueued = false;
    render(state.__renderOptions || {});
    state.__renderOptions = null;
  });
}

function render(options = {}) {
  const composerState = captureComposerState();

  document.body.dataset.theme = state.theme;

  if (state.bootError) {
    root.innerHTML = `<div class="loading-shell">Failed to load interface: ${escapeHtml(state.bootError)}</div>`;
    return;
  }

  if (!state.health) {
    root.innerHTML = `<div class="loading-shell">Booting Codex workspace...</div>`;
    return;
  }

  const contextBudget = buildContextBudget(state.usageWindow, state.rateLimitInfo);
  const provider = state.localOnlyEnabled ? "Local" : state.healthMeta.provider;
  const model = state.localOnlyEnabled ? "heuristic-runtime" : state.healthMeta.model;
  const branchName = state.blueprint?.target_branch || "main";
  const visibleMessages = state.transcript.filter((item) => item.role !== "thinking" || state.showThinking);

  root.innerHTML = `
    <div class="app-shell chat-shell">
      <main class="chat-main">
        <section class="content-panel terminal-panel${state.transcript.length ? " has-messages" : ""}">
          <div class="codex-window-chrome" aria-hidden="true">
            <div class="mac-controls">
              <span class="mac-dot red"></span>
              <span class="mac-dot yellow"></span>
              <span class="mac-dot green"></span>
            </div>
            <div class="thread-title">${state.transcript.length ? "Current thread" : "New chat"}</div>
            <div class="top-actions">
              <button type="button" class="ghost-action" data-action="theme">${state.theme === "dark" ? "Light" : "Dark"}</button>
              <button type="button" class="ghost-action" data-action="new-thread">Open</button>
              <button type="button" class="ghost-action" data-action="copy-thread">Commit</button>
            </div>
          </div>
          <div class="terminal-scroll-region">
            ${
              state.transcript.length
                ? renderTranscript(visibleMessages)
                : `
                  <div class="codex-empty-state">
                    <div class="hero-stack">
                      <div class="codex-glyph" aria-hidden="true">
                        <svg viewBox="0 0 28 28" fill="none">
                          <path d="M10.1 22.2c-3.1 0-5.8-2.5-5.8-5.7 0-2.9 2-5.2 4.8-5.7.7-3.2 3.4-5.4 6.9-5.4 4 0 7.2 3.1 7.2 7.1v.3c1.7.8 2.8 2.5 2.8 4.5 0 2.7-2.2 4.9-5 4.9H10.1Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                          <path d="M12 14h.01M17 14h.01" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                      </div>
                      <h1 class="hero-title">What should we build?</h1>
                      <div class="hero-subtitle">${escapeHtml(projectName(state.health.workspace_path || "Project"))}</div>
                    </div>
                    <div class="suggestion-row">
                      ${SUGGESTIONS.map(
                        (suggestion) =>
                          `<button type="button" class="suggestion-card" data-suggestion="${escapeAttribute(suggestion)}">${escapeHtml(suggestion)}</button>`
                      ).join("")}
                    </div>
                  </div>
                `
            }
          </div>
          <form class="terminal-form codex-composer" data-composer-form>
            <div class="composer-shell">
              <textarea
                class="terminal-input composer-input"
                rows="${state.transcript.length ? 3 : 2}"
                data-prompt-input
                placeholder="${escapeAttribute(
                  isCoolingDown()
                    ? `Cooldown active. Input unlocks in ${formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))}.`
                    : "Ask Codex anything, @ to use files, / for commands"
                )}"
                ${isCoolingDown() ? "disabled" : ""}
              >${escapeHtml(state.prompt)}</textarea>
              <div class="composer-toolbar">
                <div class="composer-toolbar-left">
                  <label class="toolbar-select compact">
                    <span>Permissions</span>
                    <select data-permissions-select>
                      <option value="default" ${state.localOnlyEnabled ? "" : "selected"}>Default permissions</option>
                      <option value="local" ${state.localOnlyEnabled ? "selected" : ""}>Work locally</option>
                    </select>
                  </label>
                  <label class="toolbar-select compact">
                    <span>Model</span>
                    <select data-model-select ${state.localOnlyEnabled ? "disabled" : ""}>
                      ${renderModelOptions(model)}
                    </select>
                  </label>
                  <label class="toolbar-select compact">
                    <span>Reasoning</span>
                    <select data-reasoning-select>
                      ${REASONING_OPTIONS.map(
                        (option) =>
                          `<option value="${option.value}" ${state.planningMode === option.value ? "selected" : ""}>${option.label}</option>`
                      ).join("")}
                    </select>
                  </label>
                </div>
                <div class="composer-toolbar-right">
                  <label class="terminal-toggle inline-toggle${state.showThinking ? " enabled" : ""}">
                    <input type="checkbox" data-thinking-toggle ${state.showThinking ? "checked" : ""} />
                    <span>Show thinking</span>
                  </label>
                  <div class="composer-meta">${escapeHtml(`${provider || "Unknown"} · ${contextBudget.remainingLabel} · ${branchName}`)}</div>
                  <button
                    class="terminal-submit composer-submit"
                    type="submit"
                    data-submit
                    ${state.isRunning || isCoolingDown() || !state.prompt.trim() ? "disabled" : ""}
                  >${
                    isCoolingDown()
                      ? formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))
                      : state.isRunning
                        ? "Working"
                        : "Send"
                  }</button>
                </div>
              </div>
            </div>
            <div class="composer-hint">Press Cmd/Ctrl + Enter to send</div>
          </form>
          ${state.toast ? `<div class="toast-banner">${escapeHtml(state.toast)}</div>` : ""}
        </section>
      </main>
    </div>
  `;

  const textarea = root.querySelector("[data-prompt-input]");
  if (textarea) {
    autosizeComposer(textarea);
  }
  restoreComposerState(composerState, options);
  syncComposerState();
}

function renderTranscript(messages) {
  return `
    <div class="chat-thread">
      ${messages
        .map(
          (item) => `
            <article class="thread-message ${item.role}">
              <div class="thread-message-role">${escapeHtml(roleLabel(item))}</div>
              <div class="thread-message-body markdown-body">${renderRichText(item.content)}</div>
            </article>
          `
        )
        .join("")}
      ${
        state.blueprint?.tasks?.length
          ? `
            <aside class="thread-plan-summary">
              <div class="thread-plan-heading">Plan</div>
              <div class="thread-plan-state">${escapeHtml(state.runtimeState || "Ready")}</div>
              <ul class="thread-plan-list">
                ${state.blueprint.tasks
                  .map((task) => `<li>${escapeHtml(`${task.is_completed ? "Done" : "Next"}: ${task.description}`)}</li>`)
                  .join("")}
              </ul>
              ${
                state.stageTraces.length
                  ? `<div class="thread-plan-meta">${escapeHtml(
                      state.stageTraces.map((trace) => `${trace.stage}: ${trace.summary}`).join(" · ")
                    )}</div>`
                  : ""
              }
              ${
                state.verificationResults.length
                  ? `<div class="thread-plan-meta">${
                      state.verificationResults.every((entry) => entry.success) ? "Verification passed" : "Verification pending"
                    }</div>`
                  : ""
              }
            </aside>
          `
          : ""
      }
    </div>
  `;
}

function renderModelOptions(activeModel) {
  const models = state.healthMeta.availableModels?.length ? state.healthMeta.availableModels : [activeModel || ""];
  return models
    .map(
      (modelName) =>
        `<option value="${escapeAttribute(modelName)}" ${activeModel === modelName ? "selected" : ""}>${escapeHtml(
          simplifyModelLabel(modelName)
        )}</option>`
    )
    .join("");
}

function renderRichText(content) {
  const text = String(content || "");
  if (text.includes("```")) {
    return text
      .split(/```/)
      .map((chunk, index) => (index % 2 ? `<pre><code>${escapeHtml(chunk.replace(/^\w+\n/, ""))}</code></pre>` : renderParagraphs(chunk)))
      .join("");
  }
  return renderParagraphs(text);
}

function renderParagraphs(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) {
        return "";
      }
      if (trimmed.startsWith("- ")) {
        return `<ul>${trimmed
          .split("\n")
          .map((line) => `<li>${escapeHtml(line.replace(/^- /, ""))}</li>`)
          .join("")}</ul>`;
      }
      return `<p>${escapeHtml(trimmed).replace(/\n/g, "<br />")}</p>`;
    })
    .join("");
}

function syncComposerState() {
  const button = root.querySelector("[data-submit]");
  if (button) {
    button.disabled = state.isRunning || isCoolingDown() || !state.prompt.trim();
  }
  const textarea = root.querySelector("[data-prompt-input]");
  if (textarea && textarea.value !== state.prompt) {
    textarea.value = state.prompt;
    autosizeComposer(textarea);
  }
}

function captureComposerState() {
  const activeElement = document.activeElement;
  const textarea = root.querySelector("[data-prompt-input]");
  const scroller = root.querySelector(".terminal-scroll-region");
  return {
    composerFocused: Boolean(activeElement && textarea && activeElement === textarea),
    selectionStart: textarea?.selectionStart ?? null,
    selectionEnd: textarea?.selectionEnd ?? null,
    scrollTop: scroller?.scrollTop ?? 0,
    nearBottom: scroller ? scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 40 : false,
  };
}

function restoreComposerState(previous, options = {}) {
  const textarea = root.querySelector("[data-prompt-input]");
  const scroller = root.querySelector(".terminal-scroll-region");
  if (scroller) {
    scroller.scrollTop = previous?.nearBottom ? scroller.scrollHeight : previous?.scrollTop || 0;
  }
  if (!textarea) {
    return;
  }
  if (options.focusComposer || options.preserveComposerFocus || previous?.composerFocused) {
    textarea.focus();
    const end = textarea.value.length;
    const start = options.moveCaretToEnd ? end : previous?.selectionStart ?? end;
    const finish = options.moveCaretToEnd ? end : previous?.selectionEnd ?? end;
    textarea.setSelectionRange(start, finish);
  }
}

function autosizeComposer(textarea) {
  textarea.style.height = "0px";
  textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 72), 220)}px`;
}

function roleLabel(item) {
  if (item.role === "user") {
    return "You";
  }
  if (item.role === "thinking") {
    return state.showThinking ? "Thinking" : "Working";
  }
  if (item.role === "error") {
    return "System";
  }
  return "Codex";
}

function simplifyModelLabel(modelName) {
  const value = String(modelName || "").trim();
  return value ? value.replace(/^.*\//, "").replace(/-/g, " ") : "Unknown";
}

function projectName(workspacePath) {
  const cleaned = String(workspacePath || "").replace(/\\/g, "/");
  const parts = cleaned.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "Project";
}

function createLogEntry(source, message) {
  return { source, message };
}

function buildLogEntries(result) {
  const systemLogs = result.system_logs?.length
    ? result.system_logs.filter((entry) => !entry.startsWith("Plan checkpoints:")).map((entry) => createLogEntry("system", entry))
    : [createLogEntry("system", "No runtime system logs were returned for this turn.")];
  const aiLogs = result.ai_logs?.length
    ? result.ai_logs.map((entry) => createLogEntry("ai", entry))
    : [createLogEntry("ai", "No AI-side trace was emitted for this turn.")];
  const stepLogs = (result.steps || []).map((step, index) =>
    createLogEntry(step.success ? "system" : "error", `Step ${index + 1}: ${step.tool_name} ${step.success ? "completed successfully" : "failed"}`)
  );
  return [...systemLogs, ...aiLogs, ...stepLogs];
}

function formatThinkingBlock(entries) {
  return ["```text", ...entries.map((entry) => `${String(entry.source).toUpperCase()}  ${entry.message}`), "```"].join("\n");
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
  return {
    remaining,
    remainingLabel: `${remaining}/${limit}`,
  };
}

function shouldAutoContinue(result, autoContinueCount) {
  const hasIncompleteTasks = Boolean(result?.blueprint?.tasks?.some((task) => !task.is_completed));
  if (!hasIncompleteTasks || result?.state !== "EXECUTING" || autoContinueCount >= 24) {
    return false;
  }
  return !(result?.system_logs || []).some((entry) => String(entry).includes("Verification failed"));
}

function selectVisibleAssistantResponse(result, aggregateLogs) {
  if (!result?.blueprint) {
    return result?.final_response || "No assistant response returned.";
  }

  const completedTasks = result.blueprint.tasks?.filter((task) => task.is_completed) || [];
  const verificationLines = aggregateLogs
    .filter((entry) => entry.source === "system" && String(entry.message).startsWith("Verification "))
    .map((entry) => `- ${entry.message}`);

  const sections = ["Completed execution plan:", ...completedTasks.map((task) => `- ${task.description}`)];
  if (verificationLines.length) {
    sections.push("", "Verification:", ...verificationLines);
  }
  if (Array.isArray(result.stage_traces) && result.stage_traces.length) {
    sections.push("", "Pipeline:", ...result.stage_traces.map((trace) => `- ${trace.stage}: ${trace.summary}`));
  }
  if (!result.blueprint.verification_passed) {
    sections.push("", "Some verification checks still need attention.");
  }
  return sections.join("\n");
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

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function isCoolingDown() {
  return Boolean(state.rateLimitInfo && state.rateLimitInfo.resetAt > state.clock);
}

function loadTheme() {
  try {
    return window.localStorage.getItem(STORAGE_THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function persistTheme(theme) {
  try {
    window.localStorage.setItem(STORAGE_THEME_KEY, theme);
  } catch {}
}

function showToast(message) {
  state.toast = message;
  if (toastTimeoutId) {
    window.clearTimeout(toastTimeoutId);
  }
  toastTimeoutId = window.setTimeout(() => {
    state.toast = "";
    scheduleRender({ preserveComposerFocus: true });
  }, 1600);
  scheduleRender({ preserveComposerFocus: true });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}
