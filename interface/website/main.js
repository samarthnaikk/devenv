const STORAGE_THEME_KEY = "devenv-ui-theme";
const STORAGE_ACCESS_KEY = "devenv-ui-access";
const STORAGE_BACKEND_KEY = "devenv-ui-backend";

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
  accessPolicy: { session_access: { codex: false, opencode: false }, backend_access: { opencode: false } },
  persistedAccess: loadPersistedAccess(),
  backends: {},
  activeBackend: "opencode",
  preferredBackend: loadPreferredBackend(),
  selectedProvider: "codex",
  visibleSessionProviders: { codex: false, opencode: false },
  providerSessions: { codex: [], opencode: [] },
  selectedSessionId: "",
  sessionDetails: {},
  sessionLoading: false,
  accessUpdating: false,
  sessionBudgetTokens: 25000,
  budgetInput: "25000",
  sessionUsageTotal: 0,
  latestTurnTokens: 0,
  latestElapsedMs: 0,
  runStartedAt: 0,
  healthRefreshPending: false,
};

const root = document.getElementById("root");
const SUGGESTIONS = [
  "Do you remember anything about the old retrieval logic for this project?",
  "What prior Codex session context is relevant to infinite memory here?",
  "Is this a new context or does it match an older Devenv session?",
];

let renderQueued = false;
let toastTimeoutId = null;
const RUNNING_STATUS_FRAMES = [
  "Scanning stored sessions",
  "Matching prior projects",
  "Collecting relevant details",
  "Drafting the memory answer",
];

bootstrap();

async function bootstrap() {
  window.setInterval(() => {
    const nextClock = Date.now();
    const nextUsageWindow = state.usageWindow.filter((entry) => nextClock - entry.timestamp < 60000);
    const nextRateLimitInfo = state.rateLimitInfo && state.rateLimitInfo.resetAt > nextClock ? state.rateLimitInfo : null;
    const shouldRender =
      nextUsageWindow.length !== state.usageWindow.length ||
      Boolean(state.rateLimitInfo) !== Boolean(nextRateLimitInfo);
    state.clock = nextClock;
    state.usageWindow = nextUsageWindow;
    state.rateLimitInfo = nextRateLimitInfo;
    if (state.isRunning) {
      updateLiveRuntimeUI();
    }
    if (shouldRender) {
      scheduleRender({ preserveComposerFocus: true });
    }
    if (shouldPollHealthDuringBoot()) {
      void refreshHealth({ silent: true });
    }
  }, 1000);

  bindEvents();
  scheduleRender();

  try {
    await refreshHealth();
    await reapplyPersistedAccess();
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
      await handleAction(action.getAttribute("data-action"), action);
    }
  });

  root.addEventListener("input", (event) => {
    if (event.target.matches("[data-prompt-input]")) {
      state.prompt = event.target.value;
      syncComposerState();
      autosizeComposer(event.target);
      return;
    }
    if (event.target.matches("[data-budget-input]")) {
      state.budgetInput = event.target.value;
      scheduleRender({ preserveComposerFocus: true });
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
    if (event.target.matches("[data-thinking-toggle]")) {
      state.showThinking = Boolean(event.target.checked);
      scheduleRender({ preserveComposerFocus: true });
      return;
    }
    if (event.target.matches("[data-provider-select]")) {
      state.selectedProvider = event.target.value;
      state.selectedSessionId = "";
      await refreshProviderSessions(state.selectedProvider);
      return;
    }
    if (event.target.matches("[data-backend-select]")) {
      state.preferredBackend = event.target.value || "auto";
      persistPreferredBackend(state.preferredBackend);
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

async function handleAction(action, element) {
  if (action === "theme") {
    state.theme = state.theme === "dark" ? "light" : "dark";
    persistTheme(state.theme);
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "toggle-thinking-details") {
    state.showThinking = !state.showThinking;
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
    state.sessionUsageTotal = 0;
    state.latestTurnTokens = 0;
    state.latestElapsedMs = 0;
    showToast("Started a new retrieval thread");
    scheduleRender({ focusComposer: true });
    return;
  }

  if (action === "copy-thread") {
    if (!state.transcript.length) {
      showToast("Nothing to copy yet");
      return;
    }
    const transcriptText = state.transcript.map((entry) => `${roleLabel(entry)}\n${String(entry.content || "").trim()}`).join("\n\n");
    try {
      await navigator.clipboard.writeText(transcriptText);
      showToast("Thread copied");
    } catch {
      showToast("Clipboard access failed");
    }
    return;
  }

  if (action === "copy-message") {
    const messageId = element?.getAttribute("data-message-id") || "";
    const message = state.transcript.find((entry) => entry.id === messageId);
    if (!message) {
      showToast("Message not found");
      return;
    }
    try {
      await navigator.clipboard.writeText(String(message.content || "").trim());
      showToast(`${roleLabel(message)} copied`);
    } catch {
      showToast("Clipboard access failed");
    }
    return;
  }

  if (action === "grant-session" || action === "revoke-session") {
    const provider = element?.getAttribute("data-provider") || "";
    await updateSessionAccess(provider, action === "grant-session");
    return;
  }

  if (action === "grant-backend" || action === "revoke-backend") {
    await updateBackendAccess("opencode", action === "grant-backend");
    return;
  }

  if (action === "select-session") {
    const sessionId = element?.getAttribute("data-session-id") || "";
    const provider = element?.getAttribute("data-provider") || state.selectedProvider;
    if (!sessionId) {
      return;
    }
    state.selectedProvider = provider;
    state.selectedSessionId = sessionId;
    await refreshSelectedSession();
    return;
  }

  if (action === "refresh-sessions") {
    const refreshedCount = await refreshVisibleSessions();
    showToast(refreshedCount ? "Refreshed open session lists" : "Open a provider list to load its sessions");
    return;
  }

  if (action === "toggle-session-provider") {
    const provider = element?.getAttribute("data-provider") || "";
    if (!provider) {
      return;
    }
    const nextVisible = !state.visibleSessionProviders[provider];
    state.visibleSessionProviders[provider] = nextVisible;
    if (!nextVisible && state.selectedProvider === provider) {
      state.selectedSessionId = "";
    }
    scheduleRender({ preserveComposerFocus: true });
    if (nextVisible && state.accessPolicy.session_access?.[provider] && !state.providerSessions[provider].length) {
      await refreshProviderSessions(provider);
    }
    return;
  }

  if (action === "apply-budget") {
    const nextValue = Number.parseInt(state.budgetInput, 10);
    state.sessionBudgetTokens = Number.isFinite(nextValue) && nextValue > 0 ? nextValue : null;
    showToast(state.sessionBudgetTokens ? "Session budget updated" : "Session budget cleared");
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "increase-budget") {
    const increment = Number.parseInt(element?.getAttribute("data-increase") || "1000", 10) || 1000;
    const current = state.sessionBudgetTokens || 0;
    state.sessionBudgetTokens = current + increment;
    state.budgetInput = String(state.sessionBudgetTokens);
    showToast(`Budget increased to ${state.sessionBudgetTokens}`);
    scheduleRender({ preserveComposerFocus: true });
  }
}

async function submitPrompt() {
  const nextPrompt = state.prompt.trim();
  if (!nextPrompt || state.isRunning || isCoolingDown() || isBudgetBlocked()) {
    return;
  }

  state.isRunning = true;
  state.runStartedAt = Date.now();
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
            backend_preference: state.preferredBackend,
            session_budget_tokens: state.sessionBudgetTokens,
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

    const turnTokens = Number(result.total_usage?.total_tokens || 0);
    state.latestTurnTokens = turnTokens;
    state.latestElapsedMs = Number(result.elapsed_ms || Date.now() - state.runStartedAt);
    state.activeBackend = result.backend_used || result.metadata?.backend_used || state.activeBackend;
    state.usageWindow = [...state.usageWindow, { timestamp: Date.now(), totalTokens: turnTokens }].filter(
      (entry) => Date.now() - entry.timestamp < 60000
    );
    state.retrievalStatus = buildRetrievalStatus(result.metadata || {});
    const budgetState = result.metadata?.budget_state || null;
    if (budgetState) {
      state.sessionUsageTotal = Number(budgetState.used || state.sessionUsageTotal);
    } else {
      state.sessionUsageTotal += turnTokens;
    }
    updateThinkingEntry(thinkingId, formatThinkingFromResult(result), false);
    state.transcript = state.transcript.map((entry) => (entry.id === thinkingId ? { ...entry, pending: false } : entry));
    state.transcript.push({
      id: `assistant-${Date.now()}`,
      role: result?.error_message ? "error" : "assistant",
      content: selectVisibleAssistantResponse(result),
    });
    state.rateLimitInfo = null;
    if (budgetState?.blocked) {
      showToast("Session budget reached");
    }
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

async function refreshHealth(options = {}) {
  if (state.healthRefreshPending) {
    return state.health;
  }
  state.healthRefreshPending = true;
  const previousIndexing = state.health?.indexing || null;
  const healthPayload = await request("/api/health");
  try {
    state.health = healthPayload;
    state.healthMeta = {
      provider: healthPayload.ai_provider || "",
      model: healthPayload.ai_model || "",
      availableModels: healthPayload.available_models || [],
    };
    state.accessPolicy = healthPayload.access_policy || state.accessPolicy;
    state.backends = healthPayload.ai_backends || {};
    state.activeBackend = healthPayload.active_backend || "opencode";
    if (!isValidBackendPreference(state.preferredBackend)) {
      state.preferredBackend = healthPayload.preferred_backend || "auto";
      persistPreferredBackend(state.preferredBackend);
    }
    if (!state.selectedProvider) {
      state.selectedProvider = "codex";
    }
    const nextIndexing = healthPayload.indexing || null;
    const indexingChanged = JSON.stringify(previousIndexing || {}) !== JSON.stringify(nextIndexing || {});
    if (!options.silent || indexingChanged) {
      scheduleRender({ preserveComposerFocus: true });
    }
    return healthPayload;
  } finally {
    state.healthRefreshPending = false;
  }
}

async function reapplyPersistedAccess() {
  const persisted = state.persistedAccess;
  const sessionEntries = Object.entries(persisted.session_access || {});
  const backendEntries = Object.entries(persisted.backend_access || {});
  for (const [provider, allowed] of sessionEntries) {
    if (allowed) {
      await request("/api/session-access", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, allowed: true }),
      });
    }
  }
  for (const [backend, allowed] of backendEntries) {
    if (allowed) {
      await request("/api/backend-access", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend, allowed: true }),
      });
    }
  }
  await refreshHealth();
}

async function updateSessionAccess(provider, allowed) {
  state.accessUpdating = true;
  scheduleRender({ preserveComposerFocus: true });
  try {
    const payload = await request("/api/session-access", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, allowed }),
    });
    state.accessPolicy = payload;
    persistAccess(state.accessPolicy);
    if (!allowed) {
      state.providerSessions[provider] = [];
      state.visibleSessionProviders[provider] = false;
      if (state.selectedProvider === provider) {
        state.selectedSessionId = "";
      }
    }
    await refreshHealth();
    showToast(`${provider} session access ${allowed ? "granted" : "revoked"}`);
  } finally {
    state.accessUpdating = false;
    scheduleRender({ preserveComposerFocus: true });
  }
}

async function updateBackendAccess(backend, allowed) {
  state.accessUpdating = true;
  scheduleRender({ preserveComposerFocus: true });
  try {
    const payload = await request("/api/backend-access", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, allowed }),
    });
    state.accessPolicy = payload;
    persistAccess(state.accessPolicy);
    if (!allowed && state.preferredBackend === "opencode") {
      state.preferredBackend = "auto";
      persistPreferredBackend(state.preferredBackend);
    }
    await refreshHealth();
    showToast(`OpenCode backend ${allowed ? "enabled" : "disabled"}`);
  } finally {
    state.accessUpdating = false;
    scheduleRender({ preserveComposerFocus: true });
  }
}

async function refreshVisibleSessions() {
  const providers = Object.keys(state.providerSessions).filter(
    (provider) => state.visibleSessionProviders[provider] && state.accessPolicy.session_access?.[provider]
  );
  for (const provider of providers) {
    await refreshProviderSessions(provider);
  }
  return providers.length;
}

async function refreshProviderSessions(provider) {
  if (!state.accessPolicy.session_access?.[provider]) {
    state.providerSessions[provider] = [];
    state.visibleSessionProviders[provider] = false;
    scheduleRender({ preserveComposerFocus: true });
    return;
  }
  state.sessionLoading = true;
  scheduleRender({ preserveComposerFocus: true });
  try {
    const payload = await request(`/api/context-sources/${encodeURIComponent(provider)}/sessions`);
    state.providerSessions[provider] = payload.sessions || [];
    if (!state.selectedSessionId && state.providerSessions[provider][0]) {
      state.selectedSessionId = state.providerSessions[provider][0].session_id;
      await refreshSelectedSession();
    }
  } finally {
    state.sessionLoading = false;
    scheduleRender({ preserveComposerFocus: true });
  }
}

async function refreshSelectedSession() {
  const provider = state.selectedProvider;
  const sessionId = state.selectedSessionId;
  if (!provider || !sessionId || !state.accessPolicy.session_access?.[provider]) {
    return;
  }
  const payload = await request(
    `/api/context-sources/${encodeURIComponent(provider)}/sessions/${encodeURIComponent(sessionId)}`
  );
  state.sessionDetails[`${provider}:${sessionId}`] = payload;
  scheduleRender({ preserveComposerFocus: true });
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
  const uiState = captureUIState();
  document.body.dataset.theme = state.theme;

  if (state.bootError) {
    root.innerHTML = `<div class="loading-shell">Failed to load interface: ${escapeHtml(state.bootError)}</div>`;
    return;
  }

  if (!state.health) {
    root.innerHTML = `<div class="loading-shell">${renderStartupShell(null)}</div>`;
    return;
  }

  if (shouldShowStartupShell()) {
    root.innerHTML = `<div class="loading-shell">${renderStartupShell(state.health.indexing || null)}</div>`;
    return;
  }

  const contextBudget = buildContextBudget(state.usageWindow, state.rateLimitInfo);
  const provider = state.healthMeta.provider || "Unknown";

  root.innerHTML = `
    <div class="app-shell chat-shell">
      <main class="chat-main">
        <div class="workspace-grid">
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
                        <div class="hero-subtitle markdown-body">Ask Devenv to search memory, inspect prior sessions, or route turns through OpenCode with explicit consent.</div>
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
                      : isBudgetBlocked()
                        ? "Session budget reached. Increase the limit in the right rail to continue."
                        : "Ask Devenv what it remembers from earlier Codex or OpenCode sessions"
                  )}"
                  ${isCoolingDown() || isBudgetBlocked() ? "disabled" : ""}
                >${escapeHtml(state.prompt)}</textarea>
                <div class="composer-toolbar">
                  <div class="composer-toolbar-left">
                    <div class="status-chip">
                      <span class="status-chip-label">Context</span>
                      <strong>${escapeHtml(state.retrievalStatus.label)}</strong>
                    </div>
                    <div class="status-chip status-chip-detail markdown-body inline-markdown">
                      ${renderRichText(state.retrievalStatus.detail)}
                    </div>
                  </div>
                  <div class="composer-toolbar-right">
                    <label class="terminal-toggle inline-toggle${state.showThinking ? " enabled" : ""}">
                      <input type="checkbox" data-thinking-toggle ${state.showThinking ? "checked" : ""} />
                      <span>Show thinking details</span>
                    </label>
                    <button
                      class="terminal-submit composer-submit"
                      type="submit"
                      data-submit
                      ${state.isRunning ? 'data-live-submit-label="true"' : ""}
                      ${state.isRunning || isCoolingDown() || isBudgetBlocked() || !state.prompt.trim() ? "disabled" : ""}
                    >${
                      isCoolingDown()
                        ? formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))
                        : isBudgetBlocked()
                          ? "Blocked"
                        : state.isRunning
                          ? runningButtonLabel()
                          : "Ask"
                    }</button>
                  </div>
                </div>
                ${state.isRunning ? `<div class="composer-running-line">${renderRunningTicker(currentPendingThinkingContent())}</div>` : ""}
              </div>
              <div class="composer-hint markdown-body inline-markdown">${renderRichText("Press Cmd/Ctrl + Enter to search memory")}</div>
            </form>
            ${state.toast ? `<div class="toast-banner markdown-body inline-markdown">${renderRichText(state.toast)}</div>` : ""}
          </section>
          <aside class="side-rail">
            ${renderAccessCard()}
            ${renderSessionsCard()}
            ${renderUsageCard(contextBudget)}
          </aside>
        </div>
      </main>
    </div>
  `;

  const textarea = root.querySelector("[data-prompt-input]");
  if (textarea) {
    autosizeComposer(textarea);
  }
  restoreUIState(uiState, options);
  syncComposerState();
}

function shouldPollHealthDuringBoot() {
  if (!state.health || state.isRunning) {
    return false;
  }
  const indexing = state.health.indexing || null;
  if (!indexing) {
    return false;
  }
  return Boolean(indexing.active);
}

function shouldShowStartupShell() {
  if (!state.health || state.transcript.length) {
    return false;
  }
  const indexing = state.health.indexing || null;
  if (!indexing) {
    return false;
  }
  const hasProviderAccess = Object.values(state.accessPolicy.session_access || {}).some(Boolean);
  return Boolean(indexing.active || (hasProviderAccess && !indexing.completed && Number(indexing.total_sessions || 0) > 0));
}

function renderStartupShell(indexing) {
  const percent = Math.max(0, Math.min(100, Number(indexing?.percent || 0)));
  const processed = Number(indexing?.processed_sessions || 0);
  const total = Number(indexing?.total_sessions || 0);
  const eta = indexing && indexing.eta_seconds != null ? formatDuration(Number(indexing.eta_seconds || 0) * 1000) : "Estimating…";
  const message = indexing?.message || "Preparing Devenv memory retrieval…";
  const providerLabel = (indexing?.providers || []).length ? String(indexing.providers.join(" + ")).toUpperCase() : "LOCAL";
  return `
    <div class="startup-shell">
      <div class="startup-card">
        <div class="startup-kicker">${escapeHtml(providerLabel)} CHUNKING</div>
        <h1 class="startup-title">Preparing session memory</h1>
        <div class="startup-copy markdown-body">${renderRichText(message)}</div>
        <div class="startup-progress-track" aria-hidden="true">
          <div class="startup-progress-fill" style="width:${percent}%;"></div>
        </div>
        <div class="startup-progress-meta">
          <strong>${escapeHtml(`${percent}%`)}</strong>
          <span>${escapeHtml(total ? `${processed}/${total} sessions` : "Counting sessions")}</span>
          <span>${escapeHtml(`ETA ${eta}`)}</span>
        </div>
      </div>
    </div>
  `;
}

function renderTranscript() {
  return `
    <div class="chat-thread">
      ${state.transcript
        .map((item) => {
          const body =
            item.role === "thinking"
              ? state.showThinking
                ? renderThinkingDetail(item.content, item.pending)
                : renderThinkingSummary(item.content, item.pending)
              : renderRichText(item.content);
          return `
            <article class="thread-message ${item.role}">
              <div class="thread-message-header">
                <div class="thread-message-role">${escapeHtml(roleLabel(item))}</div>
                ${
                  item.role === "user" || item.role === "assistant" || item.role === "error"
                    ? `
                        <button
                          type="button"
                          class="message-copy-button"
                          data-action="copy-message"
                          data-message-id="${escapeAttribute(item.id)}"
                          aria-label="Copy ${escapeAttribute(roleLabel(item))} message"
                          title="Copy"
                        >
                          ${copyIcon()}
                        </button>
                      `
                    : ""
                }
              </div>
              <div class="thread-message-body markdown-body">${body}</div>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function copyIcon() {
  return `
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="9" y="9" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1.8"></rect>
      <path d="M7 15H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
}

function renderAccessCard() {
  const codexAllowed = Boolean(state.accessPolicy.session_access?.codex);
  const opencodeSessionAllowed = Boolean(state.accessPolicy.session_access?.opencode);
  const opencodeBackendAllowed = Boolean(state.accessPolicy.backend_access?.opencode);
  const opencode = state.backends.opencode || {};
  const activeBackendLabel = formatBackendLabel(state.activeBackend);
  const preferredBackendLabel = formatBackendLabel(state.preferredBackend);
  return `
    <section class="rail-card">
      <div class="rail-card-header">
        <div>
          <div class="panel-label">Access & Providers</div>
          <h2 class="rail-title">Consent and backend state</h2>
        </div>
        <div class="backend-badge ${escapeAttribute(state.activeBackend)}">${escapeHtml(activeBackendLabel)}</div>
      </div>
      <div class="provider-grid">
        ${renderProviderAccessRow("codex", "Codex sessions", codexAllowed)}
        ${renderProviderAccessRow("opencode", "OpenCode sessions", opencodeSessionAllowed)}
      </div>
      <div class="backend-card">
        <div class="backend-copy">
          <strong>Current backend</strong>
          <span class="markdown-body inline-markdown">${renderRichText(
            `${activeBackendLabel}${state.preferredBackend !== "auto" ? ` · preferred ${preferredBackendLabel}` : ""}`
          )}</span>
        </div>
        <div class="backend-actions">
          <select class="backend-select" data-backend-select>
            <option value="auto" ${state.preferredBackend === "auto" ? "selected" : ""}>Auto</option>
            <option value="opencode" ${state.preferredBackend === "opencode" ? "selected" : ""}>OpenCode</option>
          </select>
          <button type="button" class="context-action-button ${opencodeBackendAllowed ? "" : "primary"}" data-action="${opencodeBackendAllowed ? "revoke-backend" : "grant-backend"}" ${state.accessUpdating ? "disabled" : ""}>
            ${opencodeBackendAllowed ? "Revoke" : "Grant"}
          </button>
        </div>
      </div>
      <div class="backend-summary">
        <div class="markdown-body inline-markdown">${renderRichText(`**OpenCode:** ${opencode.detail || (opencode.available ? "Available" : "Unavailable")}`)}</div>
      </div>
    </section>
  `;
}

function renderProviderAccessRow(provider, label, allowed) {
  return `
      <div class="provider-row">
        <div class="provider-copy">
          <strong>${escapeHtml(label)}</strong>
          <span class="markdown-body inline-markdown">${renderRichText(
            allowed ? "Granted" : "Permission required before Devenv can read these sessions."
          )}</span>
        </div>
      <button
        type="button"
        class="context-action-button ${allowed ? "" : "primary"}"
        data-action="${allowed ? "revoke-session" : "grant-session"}"
        data-provider="${escapeAttribute(provider)}"
        ${state.accessUpdating ? "disabled" : ""}
      >
        ${allowed ? "Revoke" : "Grant"}
      </button>
    </div>
  `;
}

function renderSessionsCard() {
  const selectedProviderVisible = Boolean(state.visibleSessionProviders[state.selectedProvider]);
  const detail = selectedProviderVisible ? state.sessionDetails[`${state.selectedProvider}:${state.selectedSessionId}`] || null : null;
  return `
    <section class="rail-card">
      <div class="rail-card-header">
        <div>
          <div class="panel-label">Sessions</div>
          <h2 class="rail-title">Browsable history</h2>
        </div>
        <button type="button" class="context-action-button" data-action="refresh-sessions" ${state.sessionLoading ? "disabled" : ""}>Refresh</button>
      </div>
      <div class="provider-group-stack">
        ${renderProviderSessionGroup("codex", "Codex")}
        ${renderProviderSessionGroup("opencode", "OpenCode")}
      </div>
      <div class="session-detail">
        ${
          detail
            ? `
              <div class="session-detail-header">
                <strong>${escapeHtml(detail.summary?.title || "Untitled session")}</strong>
                <span>${escapeHtml(detail.summary?.workspace_path || detail.summary?.updated_at || "No workspace hint")}</span>
              </div>
              <div class="session-detail-messages">
                ${(detail.messages || [])
                  .slice(0, 10)
                  .map(
                    (message) => `
                      <article class="context-message ${escapeAttribute(message.role)}">
                        <div class="bubble-role">${escapeHtml(message.role)}</div>
                        <div class="markdown-body">${renderRichText(message.content)}</div>
                      </article>
                    `
                  )
                  .join("")}
              </div>
            `
            : `<div class="rail-empty">${
                selectedProviderVisible
                  ? "Select a session to inspect the transcript summary."
                  : "Open Codex or OpenCode history when you want to browse sessions."
              }</div>`
        }
      </div>
    </section>
  `;
}

function renderProviderSessionGroup(provider, label) {
  const allowed = Boolean(state.accessPolicy.session_access?.[provider]);
  const visible = Boolean(state.visibleSessionProviders[provider]);
  const sessions = state.providerSessions[provider] || [];
  return `
    <section class="provider-session-group">
      <div class="provider-group-header">
        <div class="provider-group-copy">
          <strong>${escapeHtml(label)}</strong>
          <div class="panel-caption provider-group-caption markdown-body inline-markdown">${renderRichText(
            allowed
              ? visible
                ? `${sessions.length} loaded session${sessions.length === 1 ? "" : "s"}`
                : "Hidden until you choose to view it"
              : "Grant access before loading sessions"
          )}</div>
        </div>
        <button
          type="button"
          class="context-action-button"
          data-action="toggle-session-provider"
          data-provider="${escapeAttribute(provider)}"
          ${!allowed ? "disabled" : ""}
        >
          ${visible ? "Hide" : "Show"}
        </button>
      </div>
      ${
        !allowed
          ? `<div class="rail-empty markdown-body">Grant ${escapeHtml(label)} access to load these sessions.</div>`
          : !visible
            ? `<div class="rail-empty compact markdown-body">${renderRichText(`${label} history stays collapsed until you open it.`)}</div>`
            : `
            <div class="session-list ${state.sessionLoading ? "loading" : ""}">
              ${
                sessions.length
                  ? sessions
                      .map(
                        (session) => `
                          <button
                            type="button"
                            class="session-card${state.selectedProvider === provider && state.selectedSessionId === session.session_id ? " selected" : ""}"
                            data-action="select-session"
                            data-provider="${escapeAttribute(provider)}"
                            data-session-id="${escapeAttribute(session.session_id)}"
                          >
                            <strong>${escapeHtml(session.title || "Untitled session")}</strong>
                            <span>${escapeHtml(session.updated_at || session.workspace_path || "Unknown update time")}</span>
                            <div class="markdown-body inline-markdown"><p>${renderInlineMarkdown(session.preview || session.workspace_path || "No preview available.")}</p></div>
                          </button>
                        `
                      )
                      .join("")
                  : `<div class="rail-empty">${state.sessionLoading ? "Loading sessions..." : "No sessions available."}</div>`
              }
            </div>
          `
      }
    </section>
  `;
}

function renderUsageCard(contextBudget) {
  const budgetState = buildBudgetState();
  const usageSummary = buildUsageSummary();
  return `
    <section class="rail-card">
      <div class="rail-card-header">
        <div>
          <div class="panel-label">Usage & Runtime</div>
          <h2 class="rail-title">Tokens, budget, and status</h2>
        </div>
        <div class="runtime-pill ${state.isRunning ? "running" : ""}">
          ${state.isRunning ? "Running" : "Idle"}
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-box">
          <span>Last request</span>
          <strong>${escapeHtml(String(state.latestTurnTokens || 0))}</strong>
        </div>
        <div class="metric-box">
          <span>Session total</span>
          <strong>${escapeHtml(String(state.sessionUsageTotal || 0))}</strong>
        </div>
        <div class="metric-box">
          <span>Elapsed</span>
          <strong data-live-elapsed>${escapeHtml(
            formatDuration(state.isRunning ? Date.now() - state.runStartedAt : state.latestElapsedMs || 0)
          )}</strong>
        </div>
      </div>
      <div class="chart-stack">
        <div class="chart-block">
          <div class="chart-label">Per-turn tokens</div>
          ${renderUsageBars(state.usageWindow)}
          <div class="chart-footer markdown-body inline-markdown">${renderRichText(usageSummary.turnsLabel)}</div>
        </div>
        <div class="chart-block">
          <div class="chart-label">Session token trend</div>
          ${renderCumulativeUsageChart(state.usageWindow)}
          <div class="context-note markdown-body inline-markdown">${renderRichText(usageSummary.sessionLabel)}</div>
        </div>
      </div>
      <div class="budget-editor">
        <label>
          <span>Session token budget</span>
          <input class="budget-input" data-budget-input type="number" min="0" placeholder="No limit" value="${escapeAttribute(state.budgetInput)}" />
        </label>
        <button type="button" class="context-action-button primary" data-action="apply-budget">Apply</button>
      </div>
      <div class="budget-status markdown-body inline-markdown ${budgetState.blocked ? "blocked" : ""}">
        ${renderRichText(budgetState.label)}
      </div>
      ${budgetState.blocked ? `<button type="button" class="context-action-button" data-action="increase-budget" data-increase="1000">Increase by 1000</button>` : ""}
      <div class="runtime-summary">
        ${
          state.isRunning
            ? renderRunningTicker(currentPendingThinkingContent())
            : `<div class="thinking-live-text markdown-body inline-markdown">${renderRichText(
                `${formatBackendLabel(state.activeBackend)} ready · ${contextBudget.remainingLabel} in the current minute`
              )}</div>`
        }
      </div>
    </section>
  `;
}

function renderUsageBars(entries) {
  const points = entries.slice(-12);
  const maxValue = Math.max(...points.map((entry) => entry.totalTokens), 1);
  const bars = points
    .map((entry, index) => {
      const height = Math.max(Math.round((entry.totalTokens / maxValue) * 52), 4);
      return `<rect x="${index * 12}" y="${60 - height}" width="8" height="${height}" rx="3"></rect>`;
    })
    .join("");
  const placeholders = Array.from({ length: Math.max(12 - points.length, 0) }, (_, index) => {
    const x = (points.length + index) * 12;
    return `<rect class="placeholder-bar" x="${x}" y="54" width="8" height="6" rx="3"></rect>`;
  }).join("");
  return `<svg class="usage-chart" viewBox="0 0 144 60" preserveAspectRatio="none">
    <line class="usage-grid-line" x1="0" y1="59" x2="144" y2="59"></line>
    ${placeholders}
    ${bars}
  </svg>`;
}

function buildBudgetState() {
  if (!state.sessionBudgetTokens) {
    return { blocked: false, label: "Set a session token budget to avoid unexpected usage spikes." };
  }
  const remaining = Math.max(state.sessionBudgetTokens - state.sessionUsageTotal, 0);
  if (remaining <= 0) {
    return { blocked: true, label: `Budget reached at ${state.sessionUsageTotal}/${state.sessionBudgetTokens} tokens.` };
  }
  return { blocked: false, label: `${remaining} tokens remaining before the session budget stops new turns.` };
}

function renderThinkingSummary(content, pending) {
  const steps = parseThinkingEntries(content);
  const latest = steps[steps.length - 1] || (pending ? "Checking memory" : "Finished checking memory");
  const headline = pending ? "Devenv is checking memory" : "Devenv finished checking memory";
  return `
    <div class="thinking-card thinking-card-collapsed">
      <div class="thinking-card-topline">
        <strong>${escapeHtml(headline)}</strong>
        <button type="button" class="thinking-toggle-button" data-action="toggle-thinking-details">
          Show details
        </button>
      </div>
      ${
        pending
          ? `<div class="thinking-live-row">${renderRunningTicker(content)}</div>`
          : `<div class="thinking-collapsed-note markdown-body inline-markdown">${renderRichText(latest)}</div>`
      }
    </div>
  `;
}

function renderThinkingDetail(content, pending) {
  const steps = parseThinkingEntries(content);
  const headline = pending ? "Live reasoning trace" : "Completed reasoning trace";
  return `
    <div class="thinking-card thinking-card-detailed">
      <div class="thinking-card-topline">
        <strong>${escapeHtml(headline)}</strong>
        <button type="button" class="thinking-toggle-button" data-action="toggle-thinking-details">
          Close details
        </button>
      </div>
      <div class="thinking-detail-meta">
        <span data-live-elapsed>${escapeHtml(
          formatDuration(state.isRunning ? Date.now() - state.runStartedAt : state.latestElapsedMs || 0)
        )}</span>
        <span>${escapeHtml(formatBackendLabel(state.activeBackend))}</span>
      </div>
      ${pending ? `<div class="thinking-live-row">${renderRunningTicker(content)}</div>` : ""}
      <ol class="thinking-step-list">
        ${steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
      </ol>
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

function renderRunningTicker(content = "") {
  return `
    <span class="thinking-live-indicator" aria-hidden="true">⚡</span>
    <span class="thinking-live-text" data-running-frame>${escapeHtml(currentRunningFrame(content))}</span>
    <span class="thinking-live-dots" aria-hidden="true">
      <span></span><span></span><span></span>
    </span>
  `;
}

function currentRunningFrame(content = "") {
  const steps = parseThinkingEntries(content).filter(Boolean);
  if (steps.length) {
    const recentSteps = Array.from(new Set(steps.slice(-4)));
    const index = Math.floor(state.clock / 1400) % recentSteps.length;
    return recentSteps[index];
  }
  const index = Math.floor(state.clock / 1200) % RUNNING_STATUS_FRAMES.length;
  return RUNNING_STATUS_FRAMES[index];
}

function runningButtonLabel() {
  const dots = ".".repeat((Math.floor(state.clock / 350) % 3) + 1);
  return `Searching${dots}`;
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
    button.disabled = state.isRunning || isCoolingDown() || isBudgetBlocked() || !state.prompt.trim();
  }
  const textarea = root.querySelector("[data-prompt-input]");
  if (textarea && textarea.value !== state.prompt) {
    textarea.value = state.prompt;
    autosizeComposer(textarea);
  }
}

function updateLiveRuntimeUI() {
  const elapsed = formatDuration(Date.now() - state.runStartedAt);
  for (const node of root.querySelectorAll("[data-live-elapsed]")) {
    node.textContent = elapsed;
  }
  const frame = currentRunningFrame(currentPendingThinkingContent());
  for (const node of root.querySelectorAll("[data-running-frame]")) {
    node.textContent = frame;
  }
  const button = root.querySelector("[data-live-submit-label]");
  if (button) {
    button.textContent = runningButtonLabel();
  }
}

function currentPendingThinkingContent() {
  const pendingEntry = [...state.transcript].reverse().find((entry) => entry.role === "thinking" && entry.pending);
  return pendingEntry ? String(pendingEntry.content || "") : "";
}

function captureUIState() {
  const activeElement = document.activeElement;
  const textarea = root.querySelector("[data-prompt-input]");
  const transcriptScroller = root.querySelector(".terminal-scroll-region");
  const railScroller = root.querySelector(".side-rail");
  return {
    composerFocused: Boolean(activeElement && textarea && activeElement === textarea),
    selectionStart: textarea?.selectionStart ?? null,
    selectionEnd: textarea?.selectionEnd ?? null,
    transcriptScrollTop: transcriptScroller?.scrollTop ?? 0,
    transcriptNearBottom: transcriptScroller
      ? transcriptScroller.scrollHeight - transcriptScroller.scrollTop - transcriptScroller.clientHeight < 40
      : false,
    railScrollTop: railScroller?.scrollTop ?? 0,
  };
}

function restoreUIState(previous, options = {}) {
  const textarea = root.querySelector("[data-prompt-input]");
  const transcriptScroller = root.querySelector(".terminal-scroll-region");
  const railScroller = root.querySelector(".side-rail");
  if (transcriptScroller) {
    transcriptScroller.scrollTop = previous?.transcriptNearBottom
      ? transcriptScroller.scrollHeight
      : previous?.transcriptScrollTop || 0;
  }
  if (railScroller) {
    railScroller.scrollTop = previous?.railScrollTop || 0;
  }
  if (!textarea) {
    return;
  }
  if (options.focusComposer || options.preserveComposerFocus || previous?.composerFocused) {
    textarea.focus({ preventScroll: true });
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
      if (/^#{1,3}\s+/.test(trimmed)) {
        return trimmed
          .split("\n")
          .map((line) => {
            const match = line.match(/^(#{1,3})\s+(.*)$/);
            if (!match) {
              return `<p>${renderInlineMarkdown(line)}</p>`;
            }
            const level = String(match[1].length);
            return `<h${level}>${renderInlineMarkdown(match[2])}</h${level}>`;
          })
          .join("");
      }
      if (trimmed.startsWith("- ")) {
        return `<ul>${trimmed
          .split("\n")
          .map((line) => `<li>${renderInlineMarkdown(line.replace(/^- /, ""))}</li>`)
          .join("")}</ul>`;
      }
      if (/^\d+\.\s/.test(trimmed)) {
        return `<ol>${trimmed
          .split("\n")
          .map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\.\s/, ""))}</li>`)
          .join("")}</ol>`;
      }
      if (trimmed.startsWith("> ")) {
        return `<blockquote>${trimmed
          .split("\n")
          .map((line) => renderInlineMarkdown(line.replace(/^>\s?/, "")))
          .join("<br />")}</blockquote>`;
      }
      return `<p>${trimmed.split("\n").map((line) => renderInlineMarkdown(line)).join("<br />")}</p>`;
    })
    .join("");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text || "")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^\*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderCumulativeUsageChart(entries) {
  const points = entries.slice(-12);
  const cumulative = [];
  let running = 0;
  for (const entry of points) {
    running += entry.totalTokens;
    cumulative.push(running);
  }
  const maxValue = Math.max(...cumulative, 1);
  const path = cumulative
    .map((value, index) => {
      const x = points.length === 1 ? 72 : (index / Math.max(points.length - 1, 1)) * 144;
      const y = 56 - (value / maxValue) * 44;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return `<svg class="usage-chart cumulative" viewBox="0 0 144 60" preserveAspectRatio="none">
    <line class="usage-grid-line" x1="0" y1="59" x2="144" y2="59"></line>
    ${path ? `<path class="usage-line" d="${path}"></path>` : `<path class="usage-line usage-line-empty" d="M 0 56 L 144 56"></path>`}
  </svg>`;
}

function buildUsageSummary() {
  const turns = state.usageWindow.length;
  const sessionTotal = state.sessionUsageTotal || 0;
  return {
    turnsLabel: turns
      ? `${turns} recent request${turns === 1 ? "" : "s"} sampled · includes prompt, memory context, and answer tokens`
      : "Waiting for the first request to plot usage",
    sessionLabel: sessionTotal ? `${sessionTotal} total session tokens tracked so far` : "Session graph will rise as turns complete",
  };
}

function formatBackendLabel(value) {
  if (value === "opencode") {
    return "OpenCode";
  }
  if (value === "groq") {
    return "Devenv";
  }
  if (value === "auto") {
    return "Auto";
  }
  return String(value || "Unknown");
}

function selectVisibleAssistantResponse(result) {
  const cleanedFinalResponse = sanitizeAssistantResponse(result?.final_response);
  if (cleanedFinalResponse) {
    return cleanedFinalResponse;
  }
  const cleanedErrorMessage = sanitizeAssistantResponse(result?.error_message);
  if (cleanedErrorMessage) {
    return cleanedErrorMessage;
  }
  return "No memory answer was returned.";
}

function sanitizeAssistantResponse(content) {
  const text = String(content || "").trim();
  if (!text) {
    return "";
  }
  const replay = extractReadableReplayText(text);
  if (replay.isReplay) {
    return replay.text;
  }
  return text;
}

function extractReadableReplayText(content) {
  const text = String(content || "").trim();
  if (!text || !text.startsWith("{") || !text.includes("\n")) {
    return { isReplay: false, text: "" };
  }
  const readableParts = [];
  const errors = [];
  const toolFailures = [];
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    let payload = null;
    try {
      payload = JSON.parse(line);
    } catch {
      continue;
    }
    if (!payload || typeof payload !== "object") {
      continue;
    }
    if (payload.part && payload.part.type === "text" && payload.part.text) {
      readableParts.push(String(payload.part.text).trim());
      continue;
    }
    if (payload.payload && payload.payload.type === "agent_message" && payload.payload.message) {
      readableParts.push(String(payload.payload.message).trim());
      continue;
    }
    if (payload.type === "error" && payload.error && typeof payload.error === "object") {
      const errorData = payload.error.data && typeof payload.error.data === "object" ? payload.error.data : {};
      errors.push(normalizeReplayError(errorData.message || payload.error.message || payload.error.name || ""));
      continue;
    }
    if (payload.type === "tool_use" && payload.part && payload.part.tool === "invalid") {
      const state = payload.part.state && typeof payload.part.state === "object" ? payload.part.state : {};
      const input = state.input && typeof state.input === "object" ? state.input : {};
      if (input.error) {
        toolFailures.push(String(input.error).trim());
      }
    }
  }
  const uniqueParts = [];
  for (const part of readableParts) {
    if (part && !uniqueParts.includes(part)) {
      uniqueParts.push(part);
    }
  }
  if (uniqueParts.length) {
    return { isReplay: true, text: uniqueParts.join("\n\n") };
  }
  if (errors.length) {
    return { isReplay: true, text: errors[0] };
  }
  if (toolFailures.length) {
    return { isReplay: true, text: "A required tool call was unavailable while replaying that answer." };
  }
  return { isReplay: true, text: "I couldn't produce a readable answer from that replay." };
}

function normalizeReplayError(message) {
  const cleaned = String(message || "").trim().replace(/\s+/g, " ");
  if (!cleaned) {
    return "I couldn't complete that replayed answer.";
  }
  if (cleaned.toLowerCase().includes("user rejected permission to use this specific tool call")) {
    return "Permission to use a required tool call was denied.";
  }
  return cleaned.endsWith(".") ? cleaned : `${cleaned}.`;
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

function isBudgetBlocked() {
  return Boolean(state.sessionBudgetTokens && state.sessionUsageTotal >= state.sessionBudgetTokens);
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

function loadPersistedAccess() {
  try {
    const raw = window.localStorage.getItem(STORAGE_ACCESS_KEY);
    if (!raw) {
      return { session_access: {}, backend_access: {} };
    }
    const payload = JSON.parse(raw);
    return typeof payload === "object" && payload ? payload : { session_access: {}, backend_access: {} };
  } catch {
    return { session_access: {}, backend_access: {} };
  }
}

function persistAccess(accessPolicy) {
  try {
    window.localStorage.setItem(STORAGE_ACCESS_KEY, JSON.stringify(accessPolicy));
  } catch {}
}

function loadPreferredBackend() {
  try {
    const value = window.localStorage.getItem(STORAGE_BACKEND_KEY) || "auto";
    return isValidBackendPreference(value) ? value : "auto";
  } catch {
    return "auto";
  }
}

function persistPreferredBackend(value) {
  try {
    window.localStorage.setItem(STORAGE_BACKEND_KEY, isValidBackendPreference(value) ? value : "auto");
  } catch {}
}

function isValidBackendPreference(value) {
  return value === "auto" || value === "opencode";
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
