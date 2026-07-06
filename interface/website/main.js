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
  preferredBackend: "opencode",
  selectedProvider: "codex",
  visibleSessionProviders: { codex: false, opencode: false },
  providerSessions: { codex: [], opencode: [] },
  selectedSessionId: "",
  sessionDetails: {},
  sessionLoading: false,
  accessUpdating: false,
  performanceMode: "medium",
  privacyMode: { no_memory: false, incognito: false },
  sessionBudgetTokens: 25000,
  budgetInput: "25000",
  sessionUsageTotal: 0,
  latestTurnTokens: 0,
  latestElapsedMs: 0,
  runStartedAt: 0,
  healthRefreshPending: false,
  pendingRunMode: "memory",
  selectedTools: [],
  toolPickerOpen: false,
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
const WEB_RUNNING_STATUS_FRAMES = [
  "Searching the web",
  "Checking live sources",
  "Reading the relevant page",
  "Summarizing the result",
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
    if (event.target.matches("[data-provider-select]")) {
      state.selectedProvider = event.target.value;
      state.selectedSessionId = "";
      await refreshProviderSessions(state.selectedProvider);
      return;
    }
    if (event.target.matches("[data-backend-select]")) {
      state.preferredBackend = "opencode";
      persistPreferredBackend(state.preferredBackend);
      scheduleRender({ preserveComposerFocus: true });
      return;
    }
    if (event.target.matches("[data-performance-select]")) {
      await updatePerformanceMode(event.target.value || "medium");
      return;
    }
    if (event.target.matches("[data-incognito-toggle]")) {
      await updatePrivacyMode({ incognito: Boolean(event.target.checked) });
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

  if (action === "toggle-tool-picker") {
    state.toolPickerOpen = !state.toolPickerOpen;
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "clear-tool-selection") {
    state.selectedTools = [];
    state.toolPickerOpen = false;
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "toggle-tool") {
    const toolName = String(element?.getAttribute("data-tool-name") || "").trim();
    if (!toolName) {
      return;
    }
    const next = new Set(state.selectedTools);
    if (next.has(toolName)) {
      next.delete(toolName);
    } else {
      next.add(toolName);
    }
    state.selectedTools = Array.from(next).sort();
    scheduleRender({ preserveComposerFocus: true });
    return;
  }

  if (action === "new-thread") {
    try {
      await request("/api/thread/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    } catch {
      showToast("Backend thread reset failed");
      return;
    }
    state.prompt = "";
    state.transcript = [];
    state.toolPickerOpen = false;
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

  if (action === "generate-prompt") {
    await generatePromptFromComposer();
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
  state.pendingRunMode = inferPendingRunMode(nextPrompt);
  state.toolPickerOpen = false;
  state.prompt = "";
  const thinkingId = `thinking-${Date.now()}`;
  const pendingLogs =
    state.pendingRunMode === "web"
      ? [
          createLogEntry("tool_call", "tool: web_search"),
          createLogEntry("web_search", `query: ${nextPrompt}`),
          createLogEntry("ai", "Checking live sources for the latest answer"),
        ]
      : [createLogEntry("system", "Checking Devenv memory"), createLogEntry("ai", "Looking for prior session matches")];
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
            selected_tools: state.selectedTools,
            backend_preference: "opencode",
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
    state.pendingRunMode = "memory";
    scheduleRender({ focusComposer: true });
  }
}

async function generatePromptFromComposer() {
  const task = state.prompt.trim();
  if (!task) {
    showToast("Write a task first to generate a prompt");
    return;
  }
  try {
    const result = await request("/api/tool", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_name: "generate_prompt",
        arguments: {
          task,
          allow_memory: "true",
          allow_web_search: "false",
          output_format: "strict",
        },
      }),
    });
    const promptText = String(result.data?.prompt || result.output || "").trim();
    state.transcript.push({
      id: `prompt-tool-${Date.now()}`,
      role: result.success ? "assistant" : "error",
      content: result.success ? `## Generated Prompt\n\n\`\`\`text\n${promptText}\n\`\`\`` : `Prompt generation failed: ${result.output}`,
    });
    showToast(result.success ? "Prompt generated" : "Prompt generation failed");
    scheduleRender({ focusComposer: true });
  } catch (error) {
    showToast("Prompt generation failed");
    state.transcript.push({
      id: `prompt-tool-error-${Date.now()}`,
      role: "error",
      content: `Prompt generation failed: ${error.message}`,
    });
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
    state.performanceMode = healthPayload.performance_mode || "medium";
    state.privacyMode = healthPayload.privacy || state.privacyMode;
    if (!isValidBackendPreference(state.preferredBackend)) {
      state.preferredBackend = "opencode";
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
      state.preferredBackend = "opencode";
      persistPreferredBackend(state.preferredBackend);
    }
    await refreshHealth();
    showToast(`OpenCode backend ${allowed ? "enabled" : "disabled"}`);
  } finally {
    state.accessUpdating = false;
    scheduleRender({ preserveComposerFocus: true });
  }
}

async function updatePerformanceMode(performanceMode) {
  const payload = await request("/api/performance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ performance_mode: performanceMode }),
  });
  state.performanceMode = payload.performance_mode || "medium";
  await refreshHealth({ silent: true });
  showToast(`Performance set to ${state.performanceMode}`);
  scheduleRender({ preserveComposerFocus: true });
}

async function updatePrivacyMode({ incognito }) {
  const payload = await request("/api/privacy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ no_memory: Boolean(incognito), incognito: Boolean(incognito) }),
  });
  state.privacyMode = payload.privacy || state.privacyMode;
  await refreshHealth({ silent: true });
  showToast(state.privacyMode.incognito ? "Incognito mode enabled" : "Incognito mode disabled");
  scheduleRender({ preserveComposerFocus: true });
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
    <div class="flex flex-col h-screen overflow-hidden bg-background">
      <!-- TopAppBar -->
      <header class="flex justify-between items-center h-14 px-margin-desktop w-full z-50 bg-surface border-b border-outline-variant shrink-0">
        <div class="flex items-center gap-4">
          <span class="font-headline-md text-headline-md font-bold text-on-surface">Devenv</span>
        </div>
        <div class="absolute left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-1.5 rounded-full bg-surface-container-high border border-outline-variant">
          <span class="font-label-caps text-label-caps text-on-surface-variant uppercase">${state.transcript.length ? "Memory thread" : "New memory lookup"}</span>
          <div class="h-1.5 w-1.5 rounded-full bg-primary glowing-pip"></div>
        </div>
        <div class="flex items-center gap-3">
          <button type="button" class="p-2 rounded-lg hover:bg-surface-variant transition-colors text-on-surface-variant" data-action="theme" aria-label="Toggle theme">
            <span class="material-symbols-outlined text-[20px]">${state.theme === "dark" ? "light_mode" : "dark_mode"}</span>
          </button>
          <button type="button" class="px-3 py-1.5 font-label-caps text-label-caps bg-primary text-on-primary rounded-lg hover:opacity-80 transition-opacity" data-action="new-thread">New</button>
          <button type="button" class="px-3 py-1.5 font-label-caps text-label-caps border border-outline-variant text-on-surface rounded-lg hover:bg-surface-variant transition-colors" data-action="copy-thread">Copy</button>
        </div>
      </header>
      <main class="flex flex-1 overflow-hidden">
        <!-- Left Column: Chat & Workflow (65%) -->
        <section class="w-[65%] flex flex-col h-full bg-background relative border-r border-outline-variant">
          <div class="flex-1 overflow-y-auto p-margin-desktop space-y-8" data-scroll-region>
            ${
              state.transcript.length
                ? renderTranscript()
                : renderEmptyState()
            }
          </div>
          <!-- Bottom Composer -->
          <form class="p-margin-desktop bg-surface-container-low border-t border-outline-variant" data-composer-form>
            <div class="max-w-4xl mx-auto flex flex-col gap-3">
              <div class="relative inset-terminal rounded-xl border border-outline-variant p-4 focus-within:border-primary transition-all">
                <textarea
                  class="w-full bg-transparent border-none focus:ring-0 font-body-md text-body-md text-on-surface resize-none h-20 placeholder:text-outline outline-none"
                  data-prompt-input
                  placeholder="${escapeAttribute(
                    isCoolingDown()
                      ? `Cooldown active. Input unlocks in ${formatDuration(Math.max(state.rateLimitInfo.resetAt - state.clock, 0))}.`
                      : isBudgetBlocked()
                        ? "Session budget reached. Increase the limit in the right rail to continue."
                        : "Ask Devenv..."
                  )}"
                  ${isCoolingDown() || isBudgetBlocked() ? "disabled" : ""}
                >${escapeHtml(state.prompt)}</textarea>
                <div class="flex justify-between items-center mt-2 pt-2 border-t border-outline-variant/30">
                  <div class="flex items-center gap-2">
                    ${renderToolPicker()}
                  </div>
                  <button
                    type="button"
                    class="px-6 py-2 bg-primary text-on-primary rounded-full font-label-caps text-label-caps font-bold hover:opacity-90 transition-opacity"
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
                ${state.isRunning ? `<div class="mt-2">${renderRunningTicker(currentPendingThinkingContent())}</div>` : ""}
              </div>
            </div>
          </form>
          ${state.toast ? `<div class="toast-banner markdown-body inline-markdown">${renderRichText(state.toast)}</div>` : ""}
        </section>
        <!-- Right Column: Sidebar (35%) -->
        <aside class="w-[35%] flex flex-col h-full bg-surface-container-low border-l border-outline-variant">
          <div class="flex-1 overflow-y-auto p-4 space-y-6" data-rail-region>
            <div class="flex items-center gap-2 mb-2">
              <span class="font-label-caps text-label-caps text-primary">DEVELOPER WORKSPACE</span>
            </div>
            ${renderAccessCard()}
            ${renderSessionsCard()}
            ${renderUsageCard(contextBudget)}
            <!-- Generate Prompt Button -->
            <button type="button" class="w-full py-4 border-2 border-dashed border-outline-variant rounded-xl flex items-center justify-center gap-2 text-on-surface-variant hover:border-primary hover:text-primary transition-all group" data-action="generate-prompt">
              <span class="material-symbols-outlined group-hover:scale-110 transition-transform text-[20px]">auto_awesome</span>
              <span class="font-label-caps text-label-caps uppercase font-bold">Generate Prompt</span>
            </button>
          </div>
          <!-- Sidebar Footer -->
          <footer class="p-4 bg-surface-container-highest border-t border-outline-variant flex justify-between items-center shrink-0">
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full ${state.isRunning ? "bg-primary glowing-pip animate-pulse" : "bg-primary glowing-pip"}"></div>
              <span class="font-label-caps text-[10px] text-on-surface">${state.isRunning ? "Running" : formatBackendLabel(state.activeBackend) + " ready"}</span>
            </div>
            <span class="font-code-sm text-[10px] text-on-surface-variant">${contextBudget.remainingLabel}</span>
          </footer>
        </aside>
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

function renderEmptyState() {
  return `
    <div class="flex flex-col items-center justify-center min-h-[60vh] gap-10 px-12">
      <div class="flex flex-col items-center gap-4">
        <div class="w-12 h-12 rounded-full bg-primary flex items-center justify-center text-on-primary">
          <span class="material-symbols-outlined text-[24px]">neurology</span>
        </div>
        <h1 class="font-headline-lg text-headline-lg text-on-surface text-center">What should we recall?</h1>
        <div class="max-w-lg text-center font-body-lg text-body-lg text-on-surface-variant">Ask Devenv to search memory, inspect prior sessions, or route turns through OpenCode with explicit consent.</div>
      </div>
      <div class="grid grid-cols-1 gap-3 w-full max-w-2xl">
        ${SUGGESTIONS.map(
          (suggestion) =>
            `<button type="button" class="text-left p-4 bg-surface-container border border-outline-variant rounded-lg hover:bg-surface-container-high transition-colors font-body-md text-body-md text-on-surface" data-suggestion="${escapeAttribute(suggestion)}">${escapeHtml(suggestion)}</button>`
        ).join("")}
      </div>
    </div>
  `;
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
    <div class="loading-shell">
      <div class="startup-card">
        <div class="font-label-caps text-label-caps text-on-surface-variant">${escapeHtml(providerLabel)} CHUNKING</div>
        <h1 class="font-headline-sm text-headline-sm text-on-surface" style="margin: 8px 0 12px">Preparing session memory</h1>
        <div class="font-body-md text-body-md text-on-surface-variant markdown-body">${renderRichText(message)}</div>
        <div class="startup-progress-track" style="margin-top: 16px;">
          <div class="startup-progress-fill" style="width:${percent}%;"></div>
        </div>
        <div class="flex gap-4 mt-3 font-body-md text-body-md text-on-surface-variant">
          <strong class="text-on-surface">${escapeHtml(`${percent}%`)}</strong>
          <span>${escapeHtml(total ? `${processed}/${total} sessions` : "Counting sessions")}</span>
          <span>${escapeHtml(`ETA ${eta}`)}</span>
        </div>
      </div>
    </div>
  `;
}

function renderTranscript() {
  return state.transcript
    .map((item) => {
      if (item.role === "user") {
        return `
          <div class="flex flex-col gap-2 max-w-3xl">
            <div class="flex items-center gap-2">
              <div class="w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center">
                <span class="material-symbols-outlined text-[14px] text-on-surface">person</span>
              </div>
              <span class="font-label-caps text-label-caps text-on-surface">You</span>
              <button type="button" class="ml-auto p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant" data-action="copy-message" data-message-id="${escapeAttribute(item.id)}" title="Copy">
                <span class="material-symbols-outlined text-[16px]">content_copy</span>
              </button>
            </div>
            <div class="font-body-lg text-body-lg text-on-surface ml-8">${renderRichText(item.content)}</div>
          </div>
        `;
      }
      if (item.role === "thinking") {
        return renderThinkingDetail(item.content, item.pending);
      }
      if (item.role === "error") {
        return `
          <div class="flex flex-col gap-2 max-w-3xl">
            <div class="flex items-center gap-2">
              <div class="w-6 h-6 rounded-full bg-error flex items-center justify-center">
                <span class="material-symbols-outlined text-[14px] text-on-error">error</span>
              </div>
              <span class="font-label-caps text-label-caps text-error">Error</span>
              <button type="button" class="ml-auto p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant" data-action="copy-message" data-message-id="${escapeAttribute(item.id)}" title="Copy">
                <span class="material-symbols-outlined text-[16px]">content_copy</span>
              </button>
            </div>
            <div class="font-body-lg text-body-lg text-error ml-8">${renderRichText(item.content)}</div>
          </div>
        `;
      }
      return `
        <div class="flex flex-col gap-2 max-w-3xl">
          <div class="flex items-center gap-2">
            <div class="w-6 h-6 rounded-full bg-primary flex items-center justify-center">
              <span class="material-symbols-outlined text-on-primary text-[14px]">smart_toy</span>
            </div>
            <span class="font-label-caps text-label-caps text-primary">Devenv</span>
            <button type="button" class="ml-auto p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant" data-action="copy-message" data-message-id="${escapeAttribute(item.id)}" title="Copy">
              <span class="material-symbols-outlined text-[16px]">content_copy</span>
            </button>
          </div>
          <div class="font-body-lg text-body-lg text-on-surface ml-8 leading-relaxed">${renderRichText(item.content)}</div>
        </div>
      `;
    })
    .join("");
}



function renderAccessCard() {
  const codexAllowed = Boolean(state.accessPolicy.session_access?.codex);
  const opencodeSessionAllowed = Boolean(state.accessPolicy.session_access?.opencode);
  const opencodeBackendAllowed = Boolean(state.accessPolicy.backend_access?.opencode);
  const activeBackendLabel = formatBackendLabel(state.activeBackend);
  return `
    <section class="space-y-3">
      <h3 class="font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2">
        <span class="material-symbols-outlined text-[16px]">vpn_key</span>
        ACCESS &amp; PROVIDERS
      </h3>
      <div class="bg-surface-container rounded-lg border border-outline-variant overflow-hidden">
        <div class="p-3 border-b border-outline-variant/30 flex justify-between items-center">
          <span class="font-body-md text-body-md">Consent</span>
          <span class="text-primary material-symbols-outlined text-[18px]">check_circle</span>
        </div>
        ${renderProviderRow("codex", "Codex", codexAllowed, "session")}
        ${renderProviderRow("opencode", "OpenCode", opencodeSessionAllowed, "session")}
        <div class="p-3 flex justify-between items-center">
          <div class="flex flex-col">
            <span class="font-body-md text-body-md">Backend</span>
            <span class="text-[10px] text-outline uppercase font-bold">${escapeHtml(activeBackendLabel)}</span>
          </div>
          <button type="button" class="px-3 py-1 rounded bg-surface-variant text-on-surface font-label-caps text-[10px] hover:bg-error hover:text-on-error transition-colors" data-action="${opencodeBackendAllowed ? "revoke-backend" : "grant-backend"}" ${state.accessUpdating ? "disabled" : ""}>
            ${opencodeBackendAllowed ? "Revoke" : "Grant"}
          </button>
        </div>
      </div>
      <div class="space-y-3 pt-2">
        <div class="flex flex-col gap-1.5">
          <label class="font-label-caps text-label-caps text-on-surface-variant">PERFORMANCE MODE</label>
          <select class="bg-surface-container-highest border border-outline-variant rounded-lg font-body-md text-body-md p-2 outline-none focus:border-primary" data-performance-select>
            <option value="low" ${state.performanceMode === "low" ? "selected" : ""}>Low</option>
            <option value="medium" ${state.performanceMode === "medium" ? "selected" : ""}>Med</option>
            <option value="high" ${state.performanceMode === "high" ? "selected" : ""}>High</option>
          </select>
        </div>
        <label class="flex items-center gap-3 cursor-pointer">
          <input class="w-4 h-4 rounded border-outline-variant bg-surface-container text-primary focus:ring-0 focus:ring-offset-0" type="checkbox" data-incognito-toggle ${state.privacyMode.incognito ? "checked" : ""} />
          <span class="font-body-md text-body-md">Incognito</span>
        </label>
      </div>
    </section>
  `;
}

function renderProviderRow(provider, label, allowed, type) {
  const grantedClass = allowed ? "text-primary" : "text-outline";
  const grantedText = allowed ? "Granted" : "Not granted";
  const action = allowed ? "revoke" : "grant";
  const actionAttr = type === "session" ? `${action}-session` : `${action}-backend`;
  return `
    <div class="p-3 border-b border-outline-variant/30 flex justify-between items-center">
      <div class="flex flex-col">
        <span class="font-body-md text-body-md">${escapeHtml(label)}</span>
        <span class="text-[10px] ${grantedClass} uppercase font-bold">${grantedText}</span>
      </div>
      <button type="button" class="px-3 py-1 rounded bg-surface-variant text-on-surface font-label-caps text-[10px] hover:bg-error hover:text-on-error transition-colors" data-action="${actionAttr}" data-provider="${escapeAttribute(provider)}" ${state.accessUpdating ? "disabled" : ""}>
        ${allowed ? "Revoke" : "Grant"}
      </button>
    </div>
  `;
}

function renderSessionsCard() {
  const codexAllowed = Boolean(state.accessPolicy.session_access?.codex);
  const opencodeAllowed = Boolean(state.accessPolicy.session_access?.opencode);
  const codexVisible = Boolean(state.visibleSessionProviders?.codex);
  const opencodeVisible = Boolean(state.visibleSessionProviders?.opencode);
  return `
    <section class="space-y-3">
      <div class="flex justify-between items-center">
        <h3 class="font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2">
          <span class="material-symbols-outlined text-[16px]">history</span>
          SESSIONS
        </h3>
        <button type="button" class="p-1 hover:text-primary transition-colors text-on-surface-variant" data-action="refresh-sessions" ${state.sessionLoading ? "disabled" : ""}>
          <span class="material-symbols-outlined text-[18px]">refresh</span>
        </button>
      </div>
      ${renderSessionRow("codex", "Codex History", codexAllowed, codexVisible)}
      ${renderSessionRow("opencode", "OpenCode History", opencodeAllowed, opencodeVisible)}
    </section>
  `;
}

function renderSessionRow(provider, label, allowed, visible) {
  const sessions = state.providerSessions[provider] || [];
  return `
    <div class="bg-surface-container rounded-lg border border-outline-variant overflow-hidden">
      <div class="p-3 flex justify-between items-center">
        <span class="font-body-md text-body-md">${escapeHtml(label)}</span>
        <button type="button" class="px-3 py-1 rounded bg-surface-variant text-on-surface font-label-caps text-[10px] hover:bg-primary hover:text-on-primary transition-colors" data-action="toggle-session-provider" data-provider="${escapeAttribute(provider)}" ${!allowed ? "disabled" : ""}>
          ${visible ? "Hide" : "Show"}
        </button>
      </div>
      ${visible && allowed ? `
        <div class="border-t border-outline-variant/30 p-2 space-y-1 max-h-48 overflow-y-auto">
          ${sessions.length ? sessions.map(session => `
            <button type="button" class="w-full text-left p-2 rounded-lg ${state.selectedProvider === provider && state.selectedSessionId === session.session_id ? "bg-surface-container-highest border border-primary" : "bg-surface-dim border border-transparent"} hover:bg-surface-container-highest transition-colors" data-action="select-session" data-provider="${escapeAttribute(provider)}" data-session-id="${escapeAttribute(session.session_id)}">
              <div class="font-label-caps text-label-caps text-on-surface text-[11px]">${escapeHtml(session.title || "Untitled session")}</div>
              <div class="font-code-sm text-code-sm text-on-surface-variant truncate">${escapeHtml(session.updated_at || session.workspace_path || "")}</div>
            </button>
          `).join("") : `<div class="font-body-md text-body-md text-on-surface-variant p-2">${state.sessionLoading ? "Loading..." : "No sessions"}</div>`}
        </div>
      ` : ""}
    </div>
  `;
}

function renderUsageCard(contextBudget) {
  const budgetState = buildBudgetState();
  const statusLabel = state.isRunning ? "Running" : "Idle";
  const statusColor = state.isRunning ? "bg-primary" : "bg-outline";
  return `
    <section class="space-y-3">
      <h3 class="font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2">
        <span class="material-symbols-outlined text-[16px]">analytics</span>
        USAGE &amp; RUNTIME
      </h3>
      <div class="grid grid-cols-2 gap-3">
        <div class="p-3 bg-surface-container rounded-lg border border-outline-variant">
          <div class="font-label-caps text-label-caps text-outline mb-1 uppercase">Status</div>
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full ${statusColor}"></div>
            <span class="font-body-md text-body-md font-bold uppercase">${escapeHtml(statusLabel)}</span>
          </div>
        </div>
        <div class="p-3 bg-surface-container rounded-lg border border-outline-variant">
          <div class="font-label-caps text-label-caps text-outline mb-1 uppercase">Elapsed</div>
          <div class="font-body-md text-body-md font-bold" data-live-elapsed>${escapeHtml(
            formatDuration(state.isRunning ? Date.now() - state.runStartedAt : state.latestElapsedMs || 0)
          )}</div>
        </div>
        <div class="p-3 bg-surface-container rounded-lg border border-outline-variant">
          <div class="font-label-caps text-label-caps text-outline mb-1 uppercase">Last request</div>
          <div class="font-body-md text-body-md font-bold">${escapeHtml(formatDuration(state.latestElapsedMs || 0))}</div>
        </div>
        <div class="p-3 bg-surface-container rounded-lg border border-outline-variant">
          <div class="font-label-caps text-label-caps text-outline mb-1 uppercase">Session total</div>
          <div class="font-body-md text-body-md font-bold">${escapeHtml(String(state.sessionUsageTotal || 0))} tokens</div>
        </div>
      </div>
      <div class="flex flex-col gap-1.5 pt-2">
        <label class="font-label-caps text-label-caps text-on-surface-variant">TOKEN BUDGET</label>
        <div class="flex gap-2">
          <input class="flex-1 bg-surface-container-highest border border-outline-variant rounded-lg font-code-sm text-code-sm px-3 py-2 outline-none focus:border-primary" data-budget-input type="text" value="${escapeAttribute(state.budgetInput)}" />
          <button type="button" class="px-4 py-2 bg-surface-variant text-on-surface rounded-lg font-label-caps text-label-caps hover:bg-outline-variant transition-colors" data-action="apply-budget">Apply</button>
        </div>
      </div>
    </section>
  `;
}

function renderToolPicker() {
  const availableTools = Array.isArray(state.health?.tools) ? state.health.tools : [];
  const selected = new Set(state.selectedTools);
  const label = selected.size ? `${selected.size} selected` : "All tools";
  return `
    <div class="relative${state.toolPickerOpen ? " open" : ""}">
      <button type="button" class="flex items-center gap-2 px-3 py-1.5 bg-surface-container-highest rounded-lg border border-outline-variant hover:bg-surface-variant transition-colors" data-action="toggle-tool-picker" aria-label="Choose tools">
        <span class="font-label-caps text-label-caps text-primary">TOOLS</span>
        <span class="text-outline">/</span>
        <span class="font-label-caps text-label-caps text-on-surface">${escapeHtml(label)}</span>
      </button>
      ${
        state.toolPickerOpen
          ? `
            <div class="absolute left-0 bottom-full mb-2 z-10 w-72 max-h-80 overflow-auto border border-outline-variant rounded-lg bg-surface-container p-3 shadow-xl">
              <div class="flex items-center justify-between mb-2">
                <strong class="font-label-caps text-label-caps text-on-surface">Choose functions</strong>
                <button type="button" class="font-label-caps text-label-caps text-primary bg-transparent border-0" data-action="clear-tool-selection">Use all</button>
              </div>
              <div class="flex flex-col gap-1.5">
                ${availableTools
                  .map(
                    (toolName) => `
                      <button
                        type="button"
                        class="flex items-center justify-between w-full px-3 py-2 rounded-md border ${selected.has(toolName) ? "border-primary bg-surface" : "border-outline-variant bg-surface-dim"} text-left font-body-md text-body-md text-on-surface hover:bg-surface-container-higher transition-colors"
                        data-action="toggle-tool"
                        data-tool-name="${escapeAttribute(toolName)}"
                      >
                        <span>${escapeHtml(toolName)}</span>
                        <span class="text-primary">${selected.has(toolName) ? "check_circle" : ""}</span>
                      </button>
                    `
                  )
                  .join("")}
              </div>
            </div>
          `
          : ""
      }
    </div>
  `;
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

function renderThinkingDetail(content, pending) {
  const steps = parseThinkingEntries(content);
  const pendingWebSearch = state.pendingRunMode === "web" || steps.some((step) => step.kind === "web_search");
  const headline = pending ? (pendingWebSearch ? "Live web search" : "Live tool trace") : pendingWebSearch ? "Web search trace" : "Tool trace";
  const searchCards = steps.filter((step) => step.kind === "web_search");
  const timelineSteps = steps.filter((step) => step.kind !== "web_search");
  const lastStatus = timelineSteps.length ? timelineSteps[timelineSteps.length - 1].text : "";
  return `
    <div class="ml-8 space-y-4">
      <div class="inset-terminal rounded-lg border border-outline-variant p-4">
        <div class="flex justify-between items-center mb-4">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-primary text-[18px]">terminal</span>
            <span class="font-label-caps text-label-caps text-on-surface uppercase">${escapeHtml(headline)}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="font-label-caps text-label-caps text-on-surface-variant" data-live-elapsed>${escapeHtml(
              formatDuration(state.isRunning ? Date.now() - state.runStartedAt : state.latestElapsedMs || 0)
            )}</span>
            <span class="px-2 py-0.5 rounded bg-secondary-container text-on-secondary-container font-label-caps text-[10px]">${escapeHtml(formatBackendLabel(state.activeBackend))}</span>
          </div>
        </div>
        <div class="space-y-1 font-code-sm text-code-sm text-on-surface-variant">
          ${timelineSteps.map((step, i) => `
            <div class="flex gap-4">
              <span class="text-outline w-4 shrink-0">${i + 1}</span>
              <span>[${escapeHtml(step.label || "TRACE").toUpperCase()}] ${escapeHtml(step.text)}</span>
            </div>
          `).join("")}
        </div>
      </div>
      ${searchCards.length ? `<div class="space-y-2">${searchCards.map(renderThinkingSearchCard).join("")}</div>` : ""}
      <div class="flex items-center gap-3 px-4 py-2 bg-surface-container rounded-full border border-outline-variant w-fit">
        <span class="material-symbols-outlined text-primary text-[16px]${pending ? " animate-pulse" : ""}">bolt</span>
        <span class="font-body-md text-body-md text-on-surface">${escapeHtml(lastStatus || (pending ? "Processing..." : "Completed"))}</span>
      </div>
    </div>
  `;
}

function parseThinkingEntries(content) {
  const entries = String(content || "")
    .split("\n")
    .map((line) => line.replace(/^```(?:text)?/, "").replace(/```$/, "").trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[A-Z_]+\s+/, ""))
    .map(humanizeThinkingLine)
    .filter(Boolean)
    .map((entry) => (typeof entry === "string" ? { kind: "text", text: entry } : entry));
  return normalizeThinkingEntries(entries);
}

function humanizeThinkingLine(line) {
  const lowered = line.toLowerCase();
  if (lowered.startsWith("tool:")) {
    const toolName = line.replace(/^tool:\s*/i, "").trim();
    return {
      kind: "tool_call",
      toolName,
      label: "Tool",
      text: `Calling ${toolName}`,
    };
  }
  if (lowered.startsWith("query:")) {
    return { kind: "web_search", query: line.replace(/^query:\s*/i, "").trim(), results: [], label: "Search" };
  }
  if (lowered.startsWith("result:")) {
    const payload = line.replace(/^result:\s*/i, "").trim();
    const splitIndex = payload.lastIndexOf(" - http");
    if (splitIndex >= 0) {
      return {
        kind: "web_search_result",
        title: payload.slice(0, splitIndex).trim(),
        url: payload.slice(splitIndex + 3).trim(),
      };
    }
    return { kind: "web_search_result", title: payload, url: "" };
  }
  if (lowered.includes("queued prompt")) {
    return null;
  }
  if (lowered.includes("memory context chars")) {
    return { kind: "text", label: "Context", text: "Built the context packet" };
  }
  if (lowered.includes("prior-session")) {
    return null;
  }
  if (lowered.includes("new context")) {
    return null;
  }
  if (lowered.includes("checkpoint blueprint") || lowered.includes("checkpoint")) {
    return { kind: "text", label: "Reasoning", text: "Reasoned through the next step" };
  }
  if (lowered.includes("verification passed")) {
    return { kind: "text", label: "Verify", text: "Verified the response" };
  }
  if (lowered.includes("waiting for runtime response")) {
    return { kind: "text", label: "Runtime", text: "Waiting for the runtime" };
  }
  if (lowered.includes("retrying in")) {
    return { kind: "text", label: "Retry", text: line };
  }
  return { kind: "text", label: "Trace", text: line };
}

function normalizeThinkingEntries(entries) {
  const normalized = [];
  let activeSearch = null;
  for (const entry of entries) {
    if (entry.kind === "web_search") {
      activeSearch = {
        kind: "web_search",
        label: "Search",
        text: `Searching for ${entry.query || "a live source"}`,
        query: entry.query || "",
        results: [],
      };
      normalized.push(activeSearch);
      continue;
    }
    if (entry.kind === "web_search_result") {
      if (activeSearch) {
        activeSearch.results.push({ title: entry.title || "", url: entry.url || "" });
      }
      continue;
    }
    activeSearch = null;
    normalized.push(entry);
  }
  return normalized;
}

function formatThinkingFromResult(result) {
  const lines = [];
  const metadata = result.metadata || {};
  if (metadata.external_context_state === "privacy_blocked") {
    lines.push(createLogEntry("system", "Privacy mode blocked prior memory for this turn"));
  }
  const toolSteps = Array.isArray(result.steps) ? result.steps : [];
  if (!toolSteps.length && Array.isArray(result.stage_traces) && result.stage_traces.length) {
    for (const trace of result.stage_traces.slice(0, 5)) {
      const summary = humanizeStageTraceSummary(trace.summary);
      if (summary) {
        lines.push(createLogEntry("ai", summary));
      }
    }
  }
  for (const step of toolSteps.slice(0, 5)) {
    const description = describeToolStep(step);
    if (description) {
      lines.push(createLogEntry("tool_call", description));
    }
  }
  const webSteps = Array.isArray(result.steps) ? result.steps.filter((step) => step.tool_name === "web_search" && step.success) : [];
  for (const step of webSteps.slice(0, 2)) {
    const query = String(step.arguments?.query || "").trim();
    if (query) {
      lines.push(createLogEntry("web_search", `query: ${query}`));
    }
    const results = Array.isArray(step.data?.results) ? step.data.results.slice(0, 3) : [];
    if (results.length) {
      for (const item of results) {
        const title = String(item?.title || "").trim();
        const url = String(item?.url || "").trim();
        if (title || url) {
          lines.push(createLogEntry("web_search", `result: ${title}${title && url ? " - " : ""}${url}`));
        }
      }
    } else if (step.output) {
      lines.push(createLogEntry("tool", "Searched the web"));
    }
  }
  if (!lines.length) {
    lines.push(createLogEntry("ai", result.final_response ? "Prepared the final answer" : "Checked Devenv context"));
  }
  return formatThinkingBlock(lines);
}

function renderThinkingSearchCard(step) {
  const query = String(step.query || "").trim();
  const results = Array.isArray(step.results) ? step.results : [];
  return `
    <div class="border border-outline-variant rounded-lg bg-terminal p-3">
      <div class="flex items-center gap-2 mb-2">
        <span class="material-symbols-outlined text-primary text-[16px]">public</span>
        <strong class="font-code-sm text-code-sm text-on-surface-variant">${escapeHtml(query || "Web search")}</strong>
      </div>
      ${
        results.length
          ? `<ul class="font-code-sm text-code-sm text-on-surface-variant space-y-1">
              ${results
                .map(
                  (item) => `<li class="flex flex-col"><span>${escapeHtml(item.title || item.url || "Result")}</span>${
                    item.url ? `<a class="text-primary/60 hover:text-primary" href="${escapeAttribute(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.url)}</a>` : ""
                  }</li>`
                )
                .join("")}
            </ul>`
          : `<div class="font-code-sm text-code-sm text-on-surface-variant">Search completed.</div>`
      }
    </div>
  `;
}

function renderRunningTicker(content = "", options = {}) {
  const useGlobe = state.pendingRunMode === "web" || /query:|result:|searching the web/i.test(String(content || ""));
  const frame = currentRunningFrame(content, options);
  const pending = Boolean(options.pending);
  return `
    <span class="inline-flex items-center gap-2 px-4 py-2 bg-surface-container rounded-full border border-outline-variant">
      <span class="material-symbols-outlined text-primary text-[16px]${pending ? " animate-pulse" : ""}">${useGlobe ? "public" : "bolt"}</span>
      <span class="font-body-md text-body-md text-on-surface"${pending ? " data-running-frame" : ""}>${escapeHtml(frame)}</span>
      ${pending ? '<span class="inline-flex gap-1"><span class="w-1 h-1 rounded-full bg-on-surface/25 animate-bounce" style="animation-delay:0s"></span><span class="w-1 h-1 rounded-full bg-on-surface/25 animate-bounce" style="animation-delay:0.18s"></span><span class="w-1 h-1 rounded-full bg-on-surface/25 animate-bounce" style="animation-delay:0.36s"></span></span>' : ""}
    </span>
  `;
}

function currentRunningFrame(content = "", options = {}) {
  const pending = Boolean(options.pending);
  const steps = parseThinkingEntries(content)
    .map((step) => step.text)
    .filter(Boolean);
  if (steps.length) {
    const recentSteps = Array.from(new Set(steps.slice(-4)));
    const index = Math.floor(state.clock / 1600) % recentSteps.length;
    const chosen = recentSteps[index];
    return pending ? streamLine(chosen) : chosen;
  }
  const frames = state.pendingRunMode === "web" ? WEB_RUNNING_STATUS_FRAMES : RUNNING_STATUS_FRAMES;
  const index = Math.floor(state.clock / 1200) % frames.length;
  const chosen = frames[index];
  return pending ? streamLine(chosen) : chosen;
}

function streamLine(text) {
  const value = String(text || "");
  if (!value) {
    return "";
  }
  const tick = Math.max(12, Math.floor((state.clock - state.runStartedAt) / 35));
  if (tick <= value.length) {
    return value.slice(0, tick);
  }
  return value;
}

function describeToolStep(step) {
  if (!step || !step.tool_name) {
    return "";
  }
  const toolName = String(step.tool_name);
  if (toolName === "web_search") {
    return "tool: web_search";
  }
  const summary = summarizeToolArguments(step.arguments || {});
  return summary ? `tool: ${toolName} (${summary})` : `tool: ${toolName}`;
}

function humanizeStageTraceSummary(summary) {
  const text = String(summary || "").trim();
  if (!text) {
    return "";
  }
  const lowered = text.toLowerCase();
  if (lowered.includes("distilled context packet")) {
    return "Prepared the context for the next tool step";
  }
  if (lowered.includes("split oversized checkpoint")) {
    return "Broke the task into smaller steps";
  }
  return text;
}

function summarizeToolArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object") {
    return "";
  }
  const path = typeof argumentsValue.path === "string" ? argumentsValue.path.trim() : "";
  if (path) {
    return path;
  }
  const mode = typeof argumentsValue.mode === "string" ? argumentsValue.mode.trim() : "";
  if (mode) {
    return mode;
  }
  const url = typeof argumentsValue.url === "string" ? argumentsValue.url.trim() : "";
  if (url) {
    return url;
  }
  const keys = Object.keys(argumentsValue).slice(0, 2);
  return keys.join(", ");
}

function runningButtonLabel() {
  const dots = ".".repeat((Math.floor(state.clock / 350) % 3) + 1);
  return `${state.pendingRunMode === "web" ? "Searching web" : "Searching"}${dots}`;
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
  const frame = currentRunningFrame(currentPendingThinkingContent(), { pending: true });
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
  const transcriptScroller = root.querySelector("[data-scroll-region]");
  const railScroller = root.querySelector("[data-rail-region]");
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
  const transcriptScroller = root.querySelector("[data-scroll-region]");
  const railScroller = root.querySelector("[data-rail-region]");
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

function formatBackendLabel(value) {
  if (value === "opencode") {
    return "OpenCode";
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
    return clampAssistantResponse(normalizeTranscriptAnswer(collapseRepeatedBlocks(replay.text)));
  }
  return clampAssistantResponse(normalizeTranscriptAnswer(collapseRepeatedBlocks(text)));
}

function normalizeTranscriptAnswer(content) {
  const text = String(content || "").trim();
  if (!text) {
    return "";
  }
  const normalized = text.replace(/\r\n/g, "\n");
  const qaMatch = normalized.match(/^\s*q\.\s[\s\S]*?\n+a\.\s*([\s\S]+)$/i);
  if (qaMatch && qaMatch[1]) {
    return qaMatch[1].trim();
  }
  const answerOnlyMatch = normalized.match(/^\s*a\.\s*([\s\S]+)$/i);
  if (answerOnlyMatch && answerOnlyMatch[1]) {
    return answerOnlyMatch[1].trim();
  }
  return text;
}

function clampAssistantResponse(content) {
  const text = String(content || "").trim();
  if (!text || text.includes("```")) {
    return text;
  }
  const blocks = text.split(/\n{2,}/).filter(Boolean);
  if (text.length <= 1400 && blocks.length <= 6) {
    return text;
  }
  const shortened = blocks.slice(0, 4).join("\n\n").trim();
  return shortened.length > 1100 ? `${shortened.slice(0, 1100).trimEnd()}…` : `${shortened}\n\n…`;
}

function inferPendingRunMode(prompt) {
  const lowered = String(prompt || "").toLowerCase();
  const webMarkers = ["today", "latest", "current", "currently", "recent", "president", "prime minister", "ceo", "who is"];
  return webMarkers.some((marker) => lowered.includes(marker)) ? "web" : "memory";
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

function collapseRepeatedBlocks(content) {
  const text = String(content || "").trim();
  if (!text) {
    return "";
  }
  const blocks = text
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  if (!blocks.length) {
    return text;
  }
  const deduped = [];
  const seenCanonical = new Set();
  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index];
    if (isAffirmativeOnlyBlock(block) && index + 1 < blocks.length) {
      continue;
    }
    const canonical = canonicalizeResponseBlock(block);
    if (canonical && seenCanonical.has(canonical)) {
      continue;
    }
    if (!deduped.length || deduped[deduped.length - 1] !== block) {
      deduped.push(block);
      if (canonical) {
        seenCanonical.add(canonical);
      }
    }
  }
  return deduped.join("\n\n");
}

function canonicalizeResponseBlock(content) {
  let text = String(content || "").trim().replace(/\s+/g, " ");
  if (!text) {
    return "";
  }
  while (/^yes\.\s+yes\.\s+/i.test(text)) {
    text = text.replace(/^yes\.\s+/i, "").trim();
  }
  const nestedMatch = text.match(/^yes\.\s+yes\.\s+(.+)$/i);
  if (nestedMatch && nestedMatch[1]) {
    text = `Yes. ${nestedMatch[1].trim()}`;
  }
  return text;
}

function isAffirmativeOnlyBlock(content) {
  return /^(yes|yeah|yep)\.?$/i.test(String(content || "").trim());
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
  return "opencode";
}

function persistPreferredBackend(value) {
  try {
    window.localStorage.setItem(STORAGE_BACKEND_KEY, "opencode");
  } catch {}
}

function isValidBackendPreference(value) {
  return value === "opencode";
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
