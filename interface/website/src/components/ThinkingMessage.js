import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatDuration, formatBackendLabel } from "../utils/format.js";

export function ThinkingMessage({ message }) {
  const { state } = useApp();
  const steps = parseThinkingEntries(message.content);
  const pendingKnowledgeSearch = state.pendingRunMode === "knowledge" || steps.some((step) => step.kind === "knowledge_search");
  const pendingWebSearch = state.pendingRunMode === "web" || steps.some((step) => step.kind === "web_search");
  const headline = message.pending
    ? pendingKnowledgeSearch ? "Live knowledge search" : pendingWebSearch ? "Live web search" : "Live tool trace"
    : pendingKnowledgeSearch ? "Knowledge search trace" : pendingWebSearch ? "Web search trace" : "Tool trace";
  const searchCards = steps.filter((step) => step.kind === "web_search" || step.kind === "knowledge_search");
  const timelineSteps = steps.filter((step) => step.kind !== "web_search" && step.kind !== "knowledge_search");
  const lastStatus = timelineSteps.length ? timelineSteps[timelineSteps.length - 1].text : "";
  const summary = summarizeSearchCards(searchCards);
  const elapsed = state.isRunning ? formatDuration(Date.now() - state.runStartedAt) : formatDuration(state.latestElapsedMs || 0);

  return React.createElement(
    "div",
    { className: "ml-8 space-y-4" },
    React.createElement(
      "div",
      { className: "inset-terminal rounded-lg border border-outline-variant p-4" },
      React.createElement(
        "div",
        { className: "flex justify-between items-center mb-4" },
          React.createElement(
            "div",
            { className: "flex items-center gap-2" },
            React.createElement("span", { className: "material-symbols-outlined text-primary text-[18px]" }, "terminal"),
            React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface uppercase" }, headline)
        ),
        React.createElement(
          "div",
          { className: "flex items-center gap-2" },
          React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface-variant" }, elapsed),
          React.createElement(
            "span",
            { className: "px-2 py-0.5 rounded bg-secondary-container text-on-secondary-container font-label-caps text-[10px]" },
            formatBackendLabel(state.activeBackend)
          )
        )
      ),
      summary
        ? React.createElement(
            "div",
            { className: "mb-4 flex flex-wrap gap-2" },
            summary.map((item, index) =>
              React.createElement(
                "div",
                {
                  key: `${item.label}-${index}`,
                  className: "rounded-full border border-outline-variant bg-surface-container px-3 py-1 text-[11px] uppercase tracking-[0.12em] text-on-surface-variant",
                },
                `${item.label}: ${item.value}`
              )
            )
          )
        : null,
      React.createElement(
        "div",
        { className: "space-y-1 font-code-sm text-code-sm text-on-surface-variant" },
        timelineSteps.map((step, i) =>
          React.createElement(
            "div",
            { key: i, className: "flex gap-4" },
            React.createElement("span", { className: "text-outline w-4 shrink-0" }, i + 1),
            React.createElement("span", null, `[${(step.label || "TRACE").toUpperCase()}] ${step.text}`)
          )
        )
      ),
      searchCards.length
        ? React.createElement(
            "div",
            { className: "space-y-2 mt-3" },
            searchCards.map((step, i) => step.kind === "knowledge_search" ? renderKnowledgeCard(step, i) : renderSearchCard(step, i))
          )
        : null
    ),
    React.createElement(
      "div",
      { className: "flex items-center gap-3 px-4 py-2 bg-surface-container rounded-full border border-outline-variant w-fit" },
      React.createElement(
        "span",
        { className: `material-symbols-outlined text-primary text-[16px]${message.pending ? " animate-pulse" : ""}` },
        "bolt"
      ),
      React.createElement("span", { className: "font-body-md text-body-md text-on-surface" }, lastStatus || (message.pending ? "Processing..." : "Completed"))
    )
  );
}

function renderSearchCard(step, key) {
  const query = String(step.query || "").trim();
  const results = Array.isArray(step.results) ? step.results : [];
  return React.createElement(
    "div",
    { key, className: "border border-outline-variant rounded-xl bg-terminal p-3" },
    React.createElement(
      "div",
      { className: "mb-2 flex items-start justify-between gap-3" },
      React.createElement(
        "div",
        { className: "min-w-0" },
        React.createElement(
          "div",
          { className: "mb-1 flex items-center gap-2" },
          React.createElement("span", { className: "material-symbols-outlined text-primary text-[16px]" }, "public"),
          React.createElement("strong", { className: "font-code-sm text-code-sm text-on-surface" }, "Web search")
        ),
        React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant break-words" }, query || "Web search")
      ),
      React.createElement(
        "span",
        { className: "shrink-0 rounded-full bg-surface-container-highest px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-on-surface-variant" },
        `${results.length} result${results.length === 1 ? "" : "s"}`
      )
    ),
    results.length
      ? React.createElement(
          "div",
          { className: "space-y-2" },
          results.map((item, ri) =>
            React.createElement(
              "div",
              { key: ri, className: "rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2" },
              React.createElement("div", { className: "font-body-md text-body-md text-on-surface" }, item.title || item.url || "Result"),
              item.url
                ? React.createElement(
                    "a",
                    { className: "mt-1 block break-all font-code-sm text-code-sm text-primary/70 hover:text-primary", href: item.url, target: "_blank", rel: "noreferrer" },
                    item.url
                  )
                : null
            )
          )
        )
      : React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant" }, "Search completed.")
  );
}

function summarizeSearchCards(cards) {
  const validCards = Array.isArray(cards) ? cards : [];
  if (!validCards.length) return [];
  const resultCount = validCards.reduce((total, card) => total + (Array.isArray(card.results) ? card.results.length : 0), 0);
  const sourceCount = new Set(
    validCards
      .map((card) => card.kind === "knowledge_search" ? String(card.source || "general").toLowerCase() : "web")
      .filter(Boolean)
  ).size;
  const query = validCards.find((card) => String(card.query || "").trim())?.query || "";
  const summary = [
    { label: "Results", value: String(resultCount) },
    { label: "Sources", value: String(sourceCount) },
  ];
  if (query) {
    summary.unshift({ label: "Focus", value: query.length > 52 ? `${query.slice(0, 49)}...` : query });
  }
  return summary;
}

function renderKnowledgeCard(step, key) {
  return React.createElement(KnowledgeSearchCard, { key, step });
}

function KnowledgeSearchCard({ step }) {
  const [open, setOpen] = React.useState(true);
  const results = Array.isArray(step.results) ? step.results : [];
  const sourceMeta = getKnowledgeSourceMeta(step.source);
  return React.createElement(
    "div",
    { className: "border border-outline-variant rounded-xl bg-terminal p-3" },
    React.createElement(
      "button",
      {
        type: "button",
        className: "w-full flex items-center justify-between gap-3 text-left",
        onClick: () => setOpen((value) => !value),
      },
      React.createElement(
        "div",
        { className: "flex items-center gap-3 min-w-0" },
        React.createElement("span", { className: "material-symbols-outlined text-primary text-[16px]" }, "public"),
        React.createElement("span", { className: "material-symbols-outlined text-primary text-[18px]" }, sourceMeta.icon),
        React.createElement(
          "div",
          { className: "min-w-0" },
          React.createElement("div", { className: "font-label-caps text-label-caps text-on-surface uppercase" }, sourceMeta.label),
          React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant truncate" }, step.query || "Source query not available")
        )
      ),
      React.createElement(
        "div",
        { className: "flex items-center gap-2 shrink-0" },
        React.createElement("span", { className: "px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" }, `${results.length} result${results.length === 1 ? "" : "s"}`),
        React.createElement("span", { className: "material-symbols-outlined text-on-surface-variant text-[18px]" }, open ? "expand_less" : "expand_more")
      )
    ),
    open
      ? React.createElement(
          "div",
          { className: "mt-3 space-y-2" },
          results.length
            ? results.map((item, index) =>
                React.createElement(
                  "div",
                  { key: `${step.source}-${index}`, className: "rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2" },
                  React.createElement("div", { className: "font-body-md text-body-md text-on-surface" }, item.title || item.url || "Result"),
                  item.url
                    ? React.createElement("a", { className: "block mt-1 font-code-sm text-code-sm text-primary/70 hover:text-primary break-all", href: item.url, target: "_blank", rel: "noreferrer" }, item.url)
                    : null
                )
              )
            : React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant" }, "No results yet for this source.")
        )
      : null
  );
}

function parseThinkingEntries(content) {
  const entries = String(content || "")
    .split("\n")
    .map((line) => line.replace(/^```(?:text)?/, "").replace(/```$/, "").trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^([A-Z_]+)\s+(.*)$/);
      return {
        source: match ? match[1] : "",
        body: match ? match[2] : line,
      };
    })
    .map(({ source, body }) => humanizeThinkingLine(body, source))
    .filter(Boolean)
    .map((entry) => (typeof entry === "string" ? { kind: "text", text: entry } : entry));
  return normalizeThinkingEntries(entries);
}

function humanizeThinkingLine(line, sourceTag = "") {
  const lowered = line.toLowerCase();
  if (lowered.startsWith("tool:")) {
    const toolName = line.replace(/^tool:\s*/i, "").trim();
    return { kind: "tool_call", toolName, label: "Tool", text: `Calling ${toolName}` };
  }
  if (lowered.startsWith("query:")) {
    const query = line.replace(/^query:\s*/i, "").trim();
    return sourceTag === "KNOWLEDGE_SEARCH"
      ? { kind: "knowledge_search_query", query, results: [], label: "Search" }
      : { kind: "web_search", query, results: [], label: "Search" };
  }
  if (lowered.startsWith("source:")) {
    const parsed = safeParseJson(line.replace(/^source:\s*/i, "").trim());
    if (parsed && typeof parsed.source === "string") {
      return { kind: "knowledge_source", source: parsed.source, query: String(parsed.query || "").trim() };
    }
  }
  if (lowered.startsWith("result:")) {
    const payload = line.replace(/^result:\s*/i, "").trim();
    if (payload.startsWith("{")) {
      const parsed = safeParseJson(payload);
      if (parsed && typeof parsed.source === "string") {
        return {
          kind: "knowledge_search_result",
          source: parsed.source,
          title: String(parsed.title || ""),
          url: String(parsed.url || ""),
          query: String(parsed.query || ""),
        };
      }
    }
    const splitIndex = payload.lastIndexOf(" - http");
    if (splitIndex >= 0) {
      return { kind: "web_search_result", title: payload.slice(0, splitIndex).trim(), url: payload.slice(splitIndex + 3).trim() };
    }
    return { kind: "web_search_result", title: payload, url: "" };
  }
  if (lowered.includes("queued prompt")) return null;
  if (lowered.includes("memory context chars")) return { kind: "text", label: "Context", text: "Built the context packet" };
  if (lowered.includes("prior-session")) return null;
  if (lowered.includes("new context")) return null;
  if (lowered.includes("checkpoint blueprint") || lowered.includes("checkpoint")) return { kind: "text", label: "Reasoning", text: "Reasoned through the next step" };
  if (lowered.includes("verification passed")) return { kind: "text", label: "Verify", text: "Verified the response" };
  if (lowered.includes("waiting for runtime response")) return { kind: "text", label: "Runtime", text: "Waiting for the runtime" };
  if (lowered.includes("retrying in")) return { kind: "text", label: "Retry", text: line };
  if (lowered.startsWith("query:") && /github|youtube|reddit|stackoverflow|quora|documentation/i.test(line)) return { kind: "text", label: "Search", text: line.replace(/^query:\s*/i, "") };
  return { kind: "text", label: "Trace", text: line };
}

function normalizeThinkingEntries(entries) {
  const normalized = [];
  let activeSearch = null;
  let activeKnowledge = null;
  let pendingKnowledgeQuery = "";
  for (const entry of entries) {
    if (entry.kind === "web_search") {
      activeKnowledge = null;
      pendingKnowledgeQuery = "";
      activeSearch = { kind: "web_search", label: "Search", text: `Searching for ${entry.query || "a live source"}`, query: entry.query || "", results: [] };
      normalized.push(activeSearch);
      continue;
    }
    if (entry.kind === "web_search_result") {
      if (activeSearch) activeSearch.results.push({ title: entry.title || "", url: entry.url || "" });
      continue;
    }
    if (entry.kind === "knowledge_search_query") {
      activeSearch = null;
      activeKnowledge = null;
      pendingKnowledgeQuery = entry.query || "";
      continue;
    }
    if (entry.kind === "knowledge_source") {
      activeSearch = null;
      activeKnowledge = { kind: "knowledge_search", source: entry.source || "general", query: entry.query || pendingKnowledgeQuery || "", results: [] };
      normalized.push(activeKnowledge);
      continue;
    }
    if (entry.kind === "knowledge_search_result") {
      if (!activeKnowledge || activeKnowledge.source !== entry.source) {
        activeKnowledge = { kind: "knowledge_search", source: entry.source || "general", query: entry.query || pendingKnowledgeQuery || "", results: [] };
        normalized.push(activeKnowledge);
      }
      activeKnowledge.results.push({ title: entry.title || "", url: entry.url || "" });
      continue;
    }
    activeSearch = null;
    activeKnowledge = null;
    pendingKnowledgeQuery = "";
    normalized.push(entry);
  }
  return normalized;
}

function safeParseJson(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function getKnowledgeSourceMeta(source) {
  switch (String(source || "").toLowerCase()) {
    case "github":
      return { label: "GitHub", icon: "deployed_code" };
    case "youtube":
      return { label: "YouTube", icon: "smart_display" };
    case "reddit":
      return { label: "Reddit", icon: "forum" };
    case "stackoverflow":
      return { label: "Stack Overflow", icon: "code_blocks" };
    case "documentation":
      return { label: "Documentation", icon: "description" };
    case "quora":
      return { label: "Quora", icon: "contact_support" };
    default:
      return { label: "Web", icon: "travel_explore" };
  }
}
