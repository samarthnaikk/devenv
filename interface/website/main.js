const STORAGE_THEME_KEY = "devenv-ui-theme";

const state = {
  health: null,
  prompt: "",
  transcript: [],
  isRunning: false,
  bootError: "",
  healthMeta: { provider: "", model: "", availableModels: [] },
  usageWindow: [],
  rateLimitInfo: null,
  showThinking: false,
  clock: Date.now(),
  theme: loadTheme(),
  toast: "",
  retrievalStatus: {
    mode: "new_context",
    label: "New context",
    detail: "No prior Devenv session has been reused yet.",
  },
};

const root = document.getElementById("root");
const SUGGESTIONS = [
  "Do you remember anything about the old retrieval logic for this project?",
  "What prior Codex session context is relevant to infinite memory here?",
  "Is this a new context or does it match an older Devenv session?",
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

    const shouldRender =
      state.isRunning ||
      Boolean(nextRateLimitInfo) ||
      nextUsageWindow.length !== state.usageWindow.length ||
      nextClock !== state.clock;

    state.clock = nextClock;
    state.usageWindow = nextUsageWindow;
    state.rateLimitInfo = nextRateLimitInfo;
    if (shouldRender) {
      scheduleRender();
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

  root.addEventListener("change", (event) => {
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
    state.retrievalStatus = {
      mode: "new_context",
      label: "New context",
      detail: "No prior Devenv session has been reused yet.",
    };
    showToast("Started a new retrieval thread");
    scheduleRender({ focusComposer: true });
    return;
  }

  if (action === "copy-thread") {
    if (!state.transcript.length) {
      showToast("Nothing to copy yet");
      return;
    }
    const transcriptText = state.transcript
      .map((entry) => `${roleLabel(entry)}\n${String(entry.content || "").trim()}`)
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(transcriptText);
      showToast("Thread copied");
    } catch {
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
    createLogEntry("system", "Checking Devenv memory"),
    createLogEntry("ai", "Looking for prior session matches"),
  ];
  state.transcript.push({ id: `user-${Date.now()}`, role: "user", content: nextPrompt });
  state.transcript.push({ id: thinkingId, role: "thinking", content: formatThinkingBlock(pendingLogs), pending: true });
  scheduleRender({ focusComposer: true });

  try {
    let result = null;
    while (true) {
      try {
        result = await request("/api/turn", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: nextPrompt,
            planning_mode: "auto",
            continue_plan: false,
            local_only: false,
          }),
        });
        break;
      } catch (error) {
        const parsedRateLimit = parseRateLimitError(error.message);
        if (!parsedRateLimit) {
          throw error;
        }
        state.rateLimitInfo = parsedRateLimit;
        updateThinkingEntry(
          thinkingId,
          formatThinkingBlock([
            createLogEntry("system", "Rate limit reached"),
            createLogEntry("ai", `Retrying in ${formatDuration(parsedRateLimit.retryMs)}`),
          ]),
          true
        );
        scheduleRender({ preserveComposerFocus: true });
        await waitForCooldown(parsedRateLimit.resetAt, (remainingMs) => {
          updateThinkingEntry(
            thinkingId,
            formatThinkingBlock([
              createLogEntry("system", "Rate limit reached"),
              createLogEntry("ai", `Retrying in ${formatDuration(remainingMs)}`),
            ]),
            true
          );
          scheduleRender({ preserveComposerFocus: true });
        });
      }
    }

    state.usageWindow = [...state.usageWindow, { timestamp: Date.now(), totalTokens: result.total_usage?.total_tokens || 0 }].filter(
      (entry) => Date.now() - entry.timestamp < 60000
    );
    state.retrievalStatus = buildRetrievalStatus(result.metadata || {});
    updateThinkingEntry(thinkingId, formatThinkingFromResult(result), false);
    state.transcript = state.transcript.map((entry) => (entry.id === thinkingId ? { ...entry, pending: false } : entry));
    state.transcript.push({
      id: `assistant-${Date.now()}`,
      role: result?.error_message ? "error" : "assistant",
      content: selectVisibleAssistantResponse(result),
    });
    state.rateLimitInfo = null;
  } catch (error) {
    const parsedRateLimit = parseRateLimitError(error.message);
    updateThinkingEntry(
      thinkingId,
      formatThinkingBlock([
        createLogEntry("system", "Memory retrieval failed"),
        createLogEntry("error", error.message),
      ]),
      false
    );
    state.transcript.push({
      id: `assistant-${Date.now()}`,
      role: parsedRateLimit ? "error" : "assistant",
      content: parsedRateLimit ? "Rate limit reached while checking Devenv memory." : `Request failed: ${error.message}`,
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
    root.innerHTML = `<div class="loading-shell">Booting Devenv memory retrieval...</div>`;
    return;
  }

  const contextBudget = buildContextBudget(state.usageWindow, state.rateLimitInfo);
  const provider = state.healthMeta.provider || "Unknown";

  root.innerHTML = `
    <div class="app-shell chat-shell">
      <main class="chat-main">
        <section class="content-panel terminal-panel${state.transcript.length ? " has-messages" : ""}">
          <div class="codex-window-chrome">
            <div class="brand-slot">
              <span class="brand-name">Devenv</span>
            </div>
            <div class="thread-title">${state.transcript.length ? "Memory thread" : "New memory lookup"}</div>
            <div class="top-actions">
              <button type="button" class="ghost-action icon-action" data-action="theme" aria-label="Toggle theme">
                ${state.theme === "dark" ? sunIcon() : moonIcon()}
              </button>
              <button type="button" class="ghost-action" data-action="new-thread">New</button>
              <button type="button" class="ghost-action" data-action="copy-thread">Copy</button>
            </div>
          </div>
          <div class="terminal-scroll-region">
            ${
              state.transcript.length
                ? renderTranscript()
                : `
                  <div class="codex-empty-state">
                    <div class="hero-stack">
                      <div class="codex-glyph" aria-hidden="true">${devenvCloudIcon()}</div>
                      <h1 class="hero-title">What should we recall?</h1>
                      <div class="hero-subtitle">Focused on Devenv infinite-memory retrieval from prior Codex sessions.</div>
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
                    : "Ask Devenv what it remembers from earlier Codex sessions"
                )}"
                ${isCoolingDown() ? "disabled" : ""}
              >${escapeHtml(state.prompt)}</textarea>
              <div class="composer-toolbar">
                <div class="composer-toolbar-left">
                  <div class="status-chip">
                    <span class="status-chip-label">Context</span>
                    <strong>${escapeHtml(state.retrievalStatus.label)}</strong>
                  </div>
                  <div class="status-chip status-chip-detail">
                    ${escapeHtml(state.retrievalStatus.detail)}
                  </div>
                </div>
                <div class="composer-toolbar-right">
                  <label class="terminal-toggle inline-toggle${state.showThinking ? " enabled" : ""}">
                    <input type="checkbox" data-thinking-toggle ${state.showThinking ? "checked" : ""} />
                    <span>Show raw thinking</span>
                  </label>
                  <div class="composer-meta">${escapeHtml(`${provider} · ${contextBudget.remainingLabel}`)}</div>
                  <button
                    class="terminal-submit composer-submit"
                    type="submit"
                    data-submit
                    ${state.isRunning || isCoolingDown() || !state.prompt.trim() ? "disabled" : ""}
                  >${
                    isCoolingDown()
                      ? formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))
                      : state.isRunning
                        ? "Searching"
                        : "Ask"
                  }</button>
                </div>
              </div>
            </div>
            <div class="composer-hint">Press Cmd/Ctrl + Enter to search memory</div>
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

function renderTranscript() {
  return `
    <div class="chat-thread">
      ${state.transcript
        .map((item) => {
          const body =
            item.role === "thinking" && !state.showThinking ? renderThinkingSummary(item.content, item.pending) : renderRichText(item.content);
          return `
            <article class="thread-message ${item.role}">
              <div class="thread-message-role">${escapeHtml(roleLabel(item))}</div>
              <div class="thread-message-body markdown-body">${body}</div>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderThinkingSummary(content, pending) {
  const steps = parseThinkingEntries(content);
  const selected = steps.slice(-4);
  const headline = pending ? "Devenv is checking memory" : "Devenv finished checking memory";
  return `
    <div class="thinking-card">
      <strong>${escapeHtml(headline)}</strong>
      <ul>
        ${selected.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function parseThinkingEntries(content) {
  return String(content || "")
    .split("\n")
    .map((line) => line.replace(/^```(?:text)?/, "").replace(/```$/, "").trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[A-Z_]+\s+/, ""))
    .map(humanizeThinkingLine);
}

function humanizeThinkingLine(line) {
  const lowered = line.toLowerCase();
  if (lowered.includes("queued prompt")) {
    return "Queued your lookup";
  }
  if (lowered.includes("memory context chars")) {
    return "Built the memory context packet";
  }
  if (lowered.includes("prior-session")) {
    return "Matched prior Devenv sessions";
  }
  if (lowered.includes("new context")) {
    return "Detected a new context";
  }
  if (lowered.includes("checkpoint blueprint") || lowered.includes("checkpoint")) {
    return "Reasoned through the retrieval flow";
  }
  if (lowered.includes("verification passed")) {
    return "Verified the response";
  }
  if (lowered.includes("waiting for runtime response")) {
    return "Waiting for Devenv runtime";
  }
  if (lowered.includes("retrying in")) {
    return line;
  }
  return line;
}

function formatThinkingFromResult(result) {
  const lines = [];
  const metadata = result.metadata || {};
  if (metadata.external_context_state === "reused_prior_context") {
    lines.push(createLogEntry("system", `Prior-session match found in ${metadata.external_context_session_count || 0} session(s)`));
  } else {
    lines.push(createLogEntry("system", "New context detected; no strong prior-session match reused"));
  }
  if (Array.isArray(result.stage_traces) && result.stage_traces.length) {
    for (const trace of result.stage_traces.slice(0, 5)) {
      if (trace.summary) {
        lines.push(createLogEntry("ai", trace.summary));
      }
    }
  }
  if (!lines.length) {
    lines.push(createLogEntry("ai", "Retrieved Devenv memory context"));
  }
  return formatThinkingBlock(lines);
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
    return "Devenv status";
  }
  if (item.role === "error") {
    return "Devenv";
  }
  return "Devenv";
}

function createLogEntry(source, message) {
  return { source, message };
}

function formatThinkingBlock(entries) {
  return ["```text", ...entries.map((entry) => `${String(entry.source).toUpperCase()}  ${entry.message}`), "```"].join("\n");
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

function selectVisibleAssistantResponse(result) {
  if (result?.final_response && String(result.final_response).trim()) {
    return String(result.final_response).trim();
  }
  if (result?.error_message && String(result.error_message).trim()) {
    return String(result.error_message).trim();
  }
  return "No memory answer was returned.";
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
  const forcedTheme = new URLSearchParams(window.location.search).get("theme");
  if (forcedTheme === "dark" || forcedTheme === "light") {
    return forcedTheme;
  }
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

function moonIcon() {
  return `
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M13.9 2.8a6.8 6.8 0 1 0 3.3 12.8A7.9 7.9 0 1 1 13.9 2.8Z" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;
}

function sunIcon() {
  return `
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <circle cx="10" cy="10" r="3.5" stroke="currentColor" stroke-width="1.7"/>
      <path d="M10 1.8V4M10 16v2.2M18.2 10H16M4 10H1.8M15.9 4.1 14.3 5.7M5.7 14.3 4.1 15.9M15.9 15.9 14.3 14.3M5.7 5.7 4.1 4.1" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
    </svg>
  `;
}

function devenvCloudIcon() {
  return `
    <svg viewBox="0 0 28 28" fill="none">
      <path d="M10.1 22.2c-3.1 0-5.8-2.5-5.8-5.7 0-2.9 2-5.2 4.8-5.7.7-3.2 3.4-5.4 6.9-5.4 4 0 7.2 3.1 7.2 7.1v.3c1.7.8 2.8 2.5 2.8 4.5 0 2.7-2.2 4.9-5 4.9H10.1Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M12 14h.01M17 14h.01" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;
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
