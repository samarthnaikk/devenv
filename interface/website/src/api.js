export async function fetchHealth() {
  return request("/api/health");
}

export async function fetchFiles(path = "") {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return request(`/api/files${query}`);
}

export async function fetchFile(path) {
  const query = `?path=${encodeURIComponent(path)}`;
  return request(`/api/file${query}`);
}

export async function runTurn(prompt, planningMode = "force_direct") {
  return request("/api/turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, planning_mode: planningMode }),
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
