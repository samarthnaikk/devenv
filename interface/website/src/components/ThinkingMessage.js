import React from "react";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatDuration, formatBackendLabel } from "../utils/format.js";
import { ThinkingOrb, stateForThinkingStep } from "./ThinkingOrb.js";
import { BeamFrame, MetalSurface, MotionDeck, MotionReveal, MotionShimmerText, MotionStack, MotionStage, MotionSwap } from "./MotionPrimitives.js";

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
  const activeStep = timelineSteps[timelineSteps.length - 1] || searchCards[searchCards.length - 1] || { label: headline, text: headline };
  const orbState = searchCards.length || state.pendingRunMode === "web" || state.pendingRunMode === "knowledge"
    ? "searching"
    : stateForThinkingStep(activeStep, message.pending);
  const statusWord = {
    working: "Working",
    searching: "Searching",
    solving: "Thinking",
    listening: "Listening",
    composing: "Composing",
    shaping: "Shaping",
  }[orbState];
  const groupedTimeline = groupTimelineSteps(timelineSteps);

  return React.createElement(
    MotionStage,
    { className: "thinking-shell ml-8 space-y-4", delay: 70 },
    React.createElement(
      BeamFrame,
      { active: message.pending, tone: orbState === "searching" ? "ocean" : "mono", className: "thinking-card inset-terminal rounded-[24px] border border-outline-variant p-4" },
      React.createElement("span", { className: "thinking-card-ribbon", "aria-hidden": "true" }),
      React.createElement(
        "div",
        { className: "thinking-card-head flex justify-between items-center mb-4" },
        React.createElement(
          "div",
          { className: "flex items-center gap-3 min-w-0" },
          React.createElement(ThinkingOrb, { state: orbState, size: 64, paused: !message.pending, label: `${headline}: ${orbState}` }),
          React.createElement("span", { className: "material-symbols-outlined text-primary text-[18px]" }, "terminal"),
          React.createElement(
            "div",
            { className: "min-w-0 flex flex-col" },
            React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface uppercase" }, headline),
            React.createElement(
              MotionShimmerText,
              { className: "thinking-headline-detail", active: message.pending },
              message.pending ? "Inspecting, routing, and shaping the next response" : "Trace retained for inspection"
            )
          )
        ),
        React.createElement(
          "div",
          { className: "thinking-head-pills flex items-center gap-2" },
          React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface-variant" }, elapsed),
          React.createElement(
            "span",
            { className: "thinking-pill px-2 py-0.5 rounded bg-secondary-container text-on-secondary-container font-label-caps text-[10px]" },
            formatBackendLabel(state.activeBackend)
          )
        )
      ),
      React.createElement(
        MotionDeck,
        { className: "thinking-metrics mb-4" },
        metricPill("Trace", `${timelineSteps.length} step${timelineSteps.length === 1 ? "" : "s"}`),
        metricPill(searchCards.length ? "Search" : "Mode", searchCards.length ? `${searchCards.length} live source${searchCards.length === 1 ? "" : "s"}` : headline.replace(/ trace$/i, "")),
        metricPill("Status", message.pending ? statusWord : "Completed")
      ),
      summary
        ? React.createElement(
            MotionStack,
            { className: "thinking-summary-stack mb-4" },
            React.createElement(
              MotionDeck,
              { className: "thinking-summary-deck" },
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
          )
        : null,
      React.createElement(
        "div",
        { className: "thinking-timeline space-y-3 font-code-sm text-code-sm text-on-surface-variant" },
        groupedTimeline.map((group, groupIndex) =>
          React.createElement(
            MotionReveal,
            { key: `${group.label}-${groupIndex}`, delay: groupIndex * 55, className: "thinking-trace-row" },
            React.createElement(
              "section",
              { className: "thinking-trace-group" },
              React.createElement(
                "div",
                { className: "thinking-trace-group-head" },
                React.createElement("span", { className: "thinking-trace-group-label" }, group.label),
                React.createElement("span", { className: "thinking-trace-group-count" }, `${group.steps.length} event${group.steps.length === 1 ? "" : "s"}`)
              ),
              React.createElement(
                "div",
                { className: "thinking-trace-group-body" },
                group.steps.map(({ step, absoluteIndex }, localIndex) =>
                  React.createElement(
                    MotionReveal,
                    { key: `${group.label}-${absoluteIndex}`, delay: localIndex * 28 },
                    React.createElement(
                      "div",
                      { className: "thinking-step-row flex gap-3 items-start" },
                      React.createElement("span", { className: "thinking-step-rail", "aria-hidden": "true" }),
                      React.createElement(ThinkingOrb, {
                        state: stateForThinkingStep(step, message.pending && absoluteIndex === timelineSteps.length - 1),
                        size: 20,
                        paused: !(message.pending && absoluteIndex === timelineSteps.length - 1),
                        label: `${step.text}: ${stateForThinkingStep(step, message.pending && absoluteIndex === timelineSteps.length - 1)}`,
                      }),
                      React.createElement("span", { className: "thinking-step-index" }, absoluteIndex + 1),
                      React.createElement(
                        "div",
                        { className: "thinking-step-copy" },
                        React.createElement(
                          "div",
                          { className: "thinking-step-meta" },
                          React.createElement("span", { className: "thinking-step-tag" }, step.label || "Trace"),
                          React.createElement("span", { className: "thinking-step-state" }, describeStepState(step, message.pending && absoluteIndex === timelineSteps.length - 1))
                        ),
                        React.createElement("div", { className: "thinking-step-text" }, step.text)
                      )
                    )
                  )
                )
              )
            )
          )
        )
      ),
      searchCards.length
        ? React.createElement(
            "div",
            { className: "space-y-2 mt-3 thinking-results-zone" },
            searchCards.map((step, i) =>
              React.createElement(
                MotionStack,
                { key: `${step.kind}-${i}`, className: "thinking-result-stack" },
                step.kind === "knowledge_search" ? renderKnowledgeCard(step, i) : renderSearchCard(step, i)
              )
            )
          )
        : null
    ),
    React.createElement(
      MetalSurface,
      { className: "thinking-status-pill flex items-center gap-3 px-4 py-2 bg-surface-container rounded-full border border-outline-variant w-fit" },
      React.createElement("span", { className: "material-symbols-outlined text-primary text-[16px]" }, "bolt"),
      React.createElement(
        "span",
        { className: `font-body-md text-body-md text-on-surface process-status${message.pending ? " is-live" : ""}` },
        message.pending ? React.createElement(MotionSwap, null, React.createElement("span", { className: "process-status-word" }, statusWord), React.createElement("span", { className: "process-status-dots", "aria-hidden": "true" }, "...")) : (lastStatus || "Completed")
      )
    )
  );
}

function metricPill(label, value) {
  return React.createElement(
    "div",
    { className: "thinking-metric-pill" },
    React.createElement("span", { className: "thinking-metric-label" }, label),
    React.createElement("strong", { className: "thinking-metric-value" }, value)
  );
}

function renderSearchCard(step, key) {
  const query = String(step.query || "").trim();
  const results = Array.isArray(step.results) ? step.results : [];
  return React.createElement(
    MotionReveal,
    { key, className: "thinking-search-wrap" },
    React.createElement(
      "div",
      { className: "thinking-search-panel border border-outline-variant rounded-xl bg-terminal p-3" },
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
          { className: "space-y-2 thinking-result-list" },
          results.map((item, ri) =>
            React.createElement(
              "div",
              { key: ri, className: "thinking-result-card rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2" },
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
    )
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
    MotionReveal,
    { className: "thinking-search-wrap" },
    React.createElement(
      "div",
      { className: "thinking-search-panel border border-outline-variant rounded-xl bg-terminal p-3" },
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
                  { key: `${step.source}-${index}`, className: "thinking-result-card rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2" },
                  React.createElement("div", { className: "font-body-md text-body-md text-on-surface" }, item.title || item.url || "Result"),
                  item.url
                    ? React.createElement("a", { className: "block mt-1 font-code-sm text-code-sm text-primary/70 hover:text-primary break-all", href: item.url, target: "_blank", rel: "noreferrer" }, item.url)
                    : null
                )
              )
            : React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant" }, "No results yet for this source.")
        )
      : null
    )
  );
}

function groupTimelineSteps(steps) {
  const groups = [];
  steps.forEach((step, absoluteIndex) => {
    const label = String(step.label || "Trace").trim() || "Trace";
    const lastGroup = groups[groups.length - 1];
    if (lastGroup && lastGroup.label === label) {
      lastGroup.steps.push({ step, absoluteIndex });
      return;
    }
    groups.push({ label, steps: [{ step, absoluteIndex }] });
  });
  return groups;
}

function describeStepState(step, isLive) {
  if (isLive) return "live";
  const state = stateForThinkingStep(step, false);
  if (state === "searching") return "search";
  if (state === "working") return "run";
  if (state === "listening") return "input";
  if (state === "composing") return "compose";
  if (state === "shaping") return "shape";
  return "done";
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
  if (lowered.includes("memory context chars")) return { kind: "text", label: "Context", text: "Measured the available session context" };
  if (lowered.includes("prior-session")) return null;
  if (lowered.includes("new context")) return null;
  if (lowered.includes("mapped the request into")) return { kind: "text", label: "Plan", text: line };
  if (lowered.includes("execution plan")) return { kind: "text", label: "Plan", text: line };
  if (lowered.includes("distilled context packet")) return { kind: "text", label: "Context", text: line };
  if (lowered.includes("grounded context packet")) return { kind: "text", label: "Context", text: line };
  if (lowered.includes("executed the active checkpoint")) return { kind: "text", label: "Run", text: line };
  if (lowered.includes("recorded runtime output")) return { kind: "text", label: "Trace", text: line };
  if (lowered.includes("verification confirmed")) return { kind: "text", label: "Verify", text: line };
  if (lowered.includes("verified the runtime result")) return { kind: "text", label: "Verify", text: line };
  if (lowered.includes("verification flagged an issue")) return { kind: "text", label: "Verify", text: line };
  if (lowered.includes("focused on:")) return { kind: "text", label: "Focus", text: line };
  if (lowered.includes("answer destination:")) return { kind: "text", label: "Output", text: line };
  if (lowered.includes("touched files recorded:")) return { kind: "text", label: "Files", text: line };
  if (lowered.includes("planned checkpoints:")) return { kind: "text", label: "Plan", text: line };
  if (lowered.includes("checkpoint blueprint") || lowered.includes("checkpoint")) return { kind: "text", label: "Plan", text: "Refined the current execution checkpoint" };
  if (lowered.includes("verification passed")) return { kind: "text", label: "Verify", text: "Verified the runtime result" };
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
