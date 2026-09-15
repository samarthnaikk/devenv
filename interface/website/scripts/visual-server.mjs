import { createReadStream } from "node:fs";
import { access, readFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(__dirname, "..");
const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 4173);

const mockHealth = {
  status: "ok",
  ai_provider: "Local",
  ai_model: "visual-test-model",
  available_models: ["visual-test-model"],
  available_models_by_backend: {
    codex: ["gpt-5-codex"],
    opencode: ["visual-test-model"],
  },
  selected_models_by_backend: {
    codex: "gpt-5-codex",
    opencode: "visual-test-model",
  },
  tools: [
    "list_directory",
    "locate_files",
    "read_file",
    "search_text",
    "inspect_symbols",
    "track_symbol",
    "web_search",
    "knowledge_search",
    "generate_prompt",
    "generate_pdf",
  ],
  context_builder_enabled: true,
  context_sources: ["repo", "memory"],
  opencode_server: { reachable: true },
  performance_mode: "medium",
  privacy: { no_memory: false, incognito: false },
  setup: { ready: true, checked_at: "2026-07-27T00:00:00Z" },
  tool_readiness: {
    list_directory: { ready: true, detail: "Directory inspection available." },
    locate_files: { ready: true, detail: "Path lookup available." },
    read_file: { ready: true, detail: "Direct file reads available." },
    search_text: { ready: true, detail: "Repo-wide text search available." },
    inspect_symbols: { ready: true, detail: "Definition inspection available." },
    track_symbol: { ready: true, detail: "Symbol tracing available." },
    generate_prompt: { ready: true },
    generate_pdf: { ready: true },
    web_search: { ready: true },
    knowledge_search: { ready: true, detail: "External reference search available." },
  },
  mcp_server: {},
  codex_backend: { transport: "responses_mcp" },
  indexing: { active: false, completed: true, total_sessions: 2 },
  access_policy: {
    session_access: { codex: true, opencode: true },
    backend_access: { opencode: true, ollama: false, llama_cpp: false, codex: true },
  },
  provider_sessions: {
    codex: [
      { id: "session-1", title: "Homepage layout pass", updated_at: "2026-07-27T09:00:00Z" },
    ],
    opencode: [
      { id: "session-2", title: "Sidebar flow cleanup", updated_at: "2026-07-27T09:05:00Z" },
    ],
  },
};

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "application/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".json", "application/json; charset=utf-8"],
]);

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function sendText(res, statusCode, body) {
  res.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
  });
  res.end(body);
}

async function serveStatic(res, urlPath) {
  const cleanPath = decodeURIComponent(urlPath.split("?")[0]);
  const relativePath = cleanPath === "/" ? "index.html" : cleanPath.replace(/^\/+/, "");
  const filePath = path.resolve(websiteDir, relativePath);

  if (!filePath.startsWith(websiteDir)) {
    sendText(res, 403, "Forbidden");
    return;
  }

  try {
    await access(filePath);
  } catch {
    sendText(res, 404, "Not found");
    return;
  }

  const ext = path.extname(filePath);
  const contentType = contentTypes.get(ext) || "application/octet-stream";
  const stat = await readFile(filePath);
  res.writeHead(200, {
    "Content-Type": contentType,
    "Content-Length": stat.byteLength,
    "Cache-Control": "no-store",
  });
  createReadStream(filePath).pipe(res);
}

const server = http.createServer(async (req, res) => {
  try {
    if (!req.url) {
      sendText(res, 400, "Missing URL");
      return;
    }

    if (req.url.startsWith("/api/health")) {
      sendJson(res, 200, mockHealth);
      return;
    }

    if (req.url.startsWith("/api/")) {
      sendJson(res, 200, { ok: true });
      return;
    }

    await serveStatic(res, req.url);
  } catch (error) {
    sendText(res, 500, error instanceof Error ? error.message : "Unexpected server error");
  }
});

server.listen(port, host, () => {
  console.log(`visual server listening on http://${host}:${port}`);
});
