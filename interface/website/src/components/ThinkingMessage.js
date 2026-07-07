import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatDuration, formatBackendLabel } from "../utils/format.js";

export function ThinkingMessage({ message }) {
  const { state } = useApp();
  const steps = parseThinkingEntries(message.content);
  const pendingWebSearch = state.pendingRunMode === "web" || steps.some((step) => step.kind === "web_search");
  const headline = message.pending
    ? pendingWebSearch ? "Live web search" : "Live tool trace"
    : pendingWebSearch ? "Web search trace" : "Tool trace";
  const searchCards = steps.filter((step) => step.kind === "web_search");
  const timelineSteps = steps.filter((step) => step.kind !== "web_search");
  const lastStatus = timelineSteps.length ? timelineSteps[timelineSteps.length - 1].text : "";
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
            searchCards.map((step, i) => renderSearchCard(step, i))
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
    { key, className: "border border-outline-variant rounded-lg bg-terminal p-3" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2 mb-2" },
      React.createElement("span", { className: "material-symbols-outlined text-primary text-[16px]" }, "public"),
      React.createElement("strong", { className: "font-code-sm text-code-sm text-on-surface-variant" }, query || "Web search")
    ),
    results.length
      ? React.createElement(
          "ul",
          { className: "font-code-sm text-code-sm text-on-surface-variant space-y-1" },
          results.map((item, ri) =>
            React.createElement(
              "li",
              { key: ri, className: "flex flex-col" },
              React.createElement("span", null, item.title || item.url || "Result"),
              item.url
                ? React.createElement(
                    "a",
                    { className: "text-primary/60 hover:text-primary", href: item.url, target: "_blank", rel: "noreferrer" },
                    item.url
                  )
                : null
            )
          )
        )
      : React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant" }, "Search completed.")
  );
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
    return { kind: "tool_call", toolName, label: "Tool", text: `Calling ${toolName}` };
  }
  if (lowered.startsWith("query:")) {
    return { kind: "web_search", query: line.replace(/^query:\s*/i, "").trim(), results: [], label: "Search" };
  }
  if (lowered.startsWith("result:")) {
    const payload = line.replace(/^result:\s*/i, "").trim();
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
  return { kind: "text", label: "Trace", text: line };
}

function normalizeThinkingEntries(entries) {
  const normalized = [];
  let activeSearch = null;
  for (const entry of entries) {
    if (entry.kind === "web_search") {
      activeSearch = { kind: "web_search", label: "Search", text: `Searching for ${entry.query || "a live source"}`, query: entry.query || "", results: [] };
      normalized.push(activeSearch);
      continue;
    }
    if (entry.kind === "web_search_result") {
      if (activeSearch) activeSearch.results.push({ title: entry.title || "", url: entry.url || "" });
      continue;
    }
    activeSearch = null;
    normalized.push(entry);
  }
  return normalized;
}
