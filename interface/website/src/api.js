export async function fetchHealth() {
  return request("/api/health");
}

export async function updateModel(model, backend = null) {
  return request("/api/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, backend }),
  });
}

export async function fetchContextSources() {
  return request("/api/context-sources");
}

export async function fetchContextSessions(provider) {
  return request(`/api/context-sources/${encodeURIComponent(provider)}/sessions`);
}

export async function fetchContextSession(provider, sessionId) {
  return request(`/api/context-sources/${encodeURIComponent(provider)}/sessions/${encodeURIComponent(sessionId)}`);
}

export async function prepareContextPrompt(payload) {
  return request("/api/context-builder/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchFiles(path = "") {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return request(`/api/files${query}`);
}

export async function fetchFile(path) {
  const query = `?path=${encodeURIComponent(path)}`;
  return request(`/api/file${query}`);
}

export async function runTurn({
  prompt,
  planningMode = "auto",
  continuePlan = false,
  localOnly = false,
  selectedTools = [],
  sessionBudgetTokens = null,
  backendPreference = "opencode",
}) {
  return request("/api/turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      planning_mode: planningMode,
      continue_plan: continuePlan,
      local_only: localOnly,
      selected_tools: selectedTools,
      session_budget_tokens: sessionBudgetTokens,
      backend_preference: backendPreference,
    }),
  });
}

export async function resetThread() {
  return request("/api/thread/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export async function updateSessionAccess(provider, allowed) {
  return request("/api/session-access", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, allowed }),
  });
}

export async function updateBackendAccess(backend, allowed) {
  return request("/api/backend-access", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend, allowed }),
  });
}

export async function updatePerformance(performanceMode) {
  return request("/api/performance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ performance_mode: performanceMode }),
  });
}

export async function updatePrivacy({ no_memory, incognito }) {
  return request("/api/privacy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ no_memory: Boolean(no_memory), incognito: Boolean(incognito) }),
  });
}

export async function callTool({ toolName, arguments: args }) {
  return request("/api/tool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool_name: toolName, arguments: args }),
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
