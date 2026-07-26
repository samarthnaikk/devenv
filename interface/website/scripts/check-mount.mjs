import { JSDOM } from "jsdom";

const healthPayload = {
  status: "ok",
  ai_provider: "Local",
  ai_model: "test-model",
  available_models: ["test-model"],
  available_models_by_backend: {},
  selected_models_by_backend: {},
  context_builder_enabled: true,
  context_sources: [],
  opencode_server: { reachable: true },
  performance_mode: "medium",
  privacy: { no_memory: false, incognito: false },
  setup: { ready: true, checked_at: "2026-07-26T00:00:00Z" },
  tool_readiness: {
    generate_prompt: { ready: true },
    web_search: { ready: true },
  },
  mcp_server: {},
  codex_backend: { transport: "responses_mcp" },
  indexing: { active: false, completed: true, total_sessions: 0 },
  access_policy: {
    session_access: { codex: false, opencode: false },
    backend_access: { opencode: false, ollama: false, codex: false },
  },
};

const dom = new JSDOM("<!doctype html><html><body><div id=\"root\"></div></body></html>", {
  url: "http://127.0.0.1:4173/",
});

const fetchCalls = [];

function setGlobal(key, value) {
  Object.defineProperty(globalThis, key, {
    value,
    configurable: true,
    writable: true,
  });
}

setGlobal("window", dom.window);
setGlobal("document", dom.window.document);
Object.defineProperty(dom.window.navigator, "userAgent", {
  value: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
  configurable: true,
});
Object.defineProperty(dom.window.navigator, "clipboard", {
  value: { writeText: async () => {} },
  configurable: true,
});
setGlobal("navigator", dom.window.navigator);
setGlobal("localStorage", dom.window.localStorage);
setGlobal("CustomEvent", dom.window.CustomEvent);
setGlobal("Event", dom.window.Event);
setGlobal("HTMLElement", dom.window.HTMLElement);
setGlobal("Node", dom.window.Node);
setGlobal("MutationObserver", dom.window.MutationObserver);
setGlobal("requestAnimationFrame", (callback) => setTimeout(() => callback(Date.now()), 0));
setGlobal("cancelAnimationFrame", (id) => clearTimeout(id));

const fetchImpl = async (url, options = {}) => {
  fetchCalls.push({ url: String(url), method: options.method || "GET" });
  return {
    ok: true,
    status: 200,
    async json() {
      return healthPayload;
    },
  };
};

setGlobal("fetch", fetchImpl);
dom.window.fetch = fetchImpl;
dom.window.matchMedia = () => ({
  matches: false,
  addEventListener() {},
  removeEventListener() {},
});
dom.window.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
setGlobal("ResizeObserver", dom.window.ResizeObserver);

await import("../src/index.js");
await new Promise((resolve) => setTimeout(resolve, 500));

const root = dom.window.document.getElementById("root");
const markup = root?.innerHTML || "";
const hasHealthFetch = fetchCalls.some((call) => call.url === "/api/health");

if (!hasHealthFetch) {
  throw new Error("Mount check failed: UI never requested /api/health.");
}

if (markup.length < 5000) {
  throw new Error(`Mount check failed: rendered markup was too small (${markup.length} chars).`);
}

const summary = {
  fetchCalls,
  htmlLength: markup.length,
  htmlPreview: markup.slice(0, 320),
};

console.log(JSON.stringify(summary, null, 2));
