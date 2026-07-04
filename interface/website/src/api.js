export async function fetchHealth() {
  return request("/api/health");
}

export async function updateModel(model) {
  return request("/api/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
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

export async function runTurn(prompt, planningMode = "auto", continuePlan = false, localOnly = false, selectedTools = []) {
  return request("/api/turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      planning_mode: planningMode,
      continue_plan: continuePlan,
      local_only: localOnly,
      selected_tools: selectedTools,
    }),
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
