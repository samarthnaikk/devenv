import React from "react";
import { useApp } from "../context/AppContext.js";
import { BeamFrame, MotionDeck, MotionReveal, MotionShimmerText } from "./MotionPrimitives.js";

const TOOL_META = {
  list_directory: { icon: "folder_open", label: "Files", hint: "Map folders and top-level structure", category: "workspace" },
  locate_files: { icon: "find_in_page", label: "Locate", hint: "Find likely files before reading", category: "workspace" },
  read_file: { icon: "description", label: "Read", hint: "Open exact files and inspect content", category: "workspace" },
  search_text: { icon: "match_case", label: "Search", hint: "Search the repo for strings and usages", category: "workspace" },
  inspect_symbols: { icon: "route", label: "Symbols", hint: "Inspect definitions, exports, and structure", category: "workspace" },
  track_symbol: { icon: "conversion_path", label: "Trace", hint: "Follow a symbol through the codebase", category: "workspace" },
  generate_pdf: { icon: "picture_as_pdf", label: "PDF", hint: "Generate polished PDFs", category: "artifacts" },
  generate_prompt: { icon: "auto_awesome", label: "Prompt", hint: "Prepare a strong prompt", category: "artifacts" },
  knowledge_search: { icon: "hub", label: "Knowledge", hint: "Pull repos and references", category: "research" },
  web_search: { icon: "language", label: "Web", hint: "Search live sources", category: "research" },
};

const TOOL_CATEGORY_META = {
  workspace: {
    label: "Workspace",
    detail: "Stay grounded in the current repo before planning or answering.",
  },
  research: {
    label: "Live Research",
    detail: "Pull current web facts or external references when memory is not enough.",
  },
  artifacts: {
    label: "Artifacts",
    detail: "Generate polished outputs instead of only answering in chat.",
  },
};

const USER_VISIBLE_TOOLS = new Set(Object.keys(TOOL_META));

export function ToolPicker() {
  const { state, dispatch } = useApp();
  const availableTools = (Array.isArray(state.health?.tools) ? state.health.tools : []).filter((toolName) => USER_VISIBLE_TOOLS.has(toolName));
  const toolReadiness = state.health?.tool_readiness || {};
  const visibleSelectedTools = state.selectedTools.filter((toolName) => USER_VISIBLE_TOOLS.has(toolName));
  const selected = new Set(visibleSelectedTools);
  const [droppingTools, setDroppingTools] = React.useState([]);
  const selectedToolsKey = state.selectedTools.join("|");
  const visibleSelectedToolsKey = visibleSelectedTools.join("|");

  React.useEffect(() => {
    if (selectedToolsKey === visibleSelectedToolsKey) return;
    dispatch({ type: "SET_SELECTED_TOOLS", payload: visibleSelectedTools });
  }, [dispatch, selectedToolsKey, visibleSelectedTools, visibleSelectedToolsKey]);

  React.useEffect(() => {
    if (!droppingTools.length) return undefined;
    const timer = window.setTimeout(() => {
      setDroppingTools([]);
    }, 720);
    return () => window.clearTimeout(timer);
  }, [droppingTools]);

  const toggleToolPicker = () => {
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: !state.toolPickerOpen });
  };

  const clearToolSelection = () => {
    dispatch({ type: "SET_SELECTED_TOOLS", payload: [] });
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
  };

  const toggleTool = (toolName) => {
    const next = new Set(visibleSelectedTools);
    if (next.has(toolName)) {
      next.delete(toolName);
    } else {
      next.add(toolName);
      setDroppingTools((current) => Array.from(new Set([...current, toolName])));
    }
    dispatch({ type: "SET_SELECTED_TOOLS", payload: Array.from(next).sort() });
  };

  const selectedTools = Array.from(selected);
  const routeSummary = summarizeRoute(visibleSelectedTools, state.planMode);
  const groupedTools = groupToolsByCategory(availableTools);
  const selectedToolChips = selectedTools.map((toolName) => {
    const meta = describeTool(toolName);
    return React.createElement(
      "button",
      {
        key: `selected-${toolName}`,
        type: "button",
        className: `tool-token inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-primary/40 bg-surface text-on-surface hover:bg-surface-container-high transition-colors${droppingTools.includes(toolName) ? " tool-token-drop" : ""}`,
        onClick: () => toggleTool(toolName),
        title: `Remove ${toolName}`,
        "aria-label": `Remove ${toolName}`,
      },
      React.createElement("span", { className: "material-symbols-outlined text-[15px] text-primary" }, meta.icon),
      React.createElement("span", { className: "font-label-caps text-[10px] uppercase tracking-[0.08em]" }, meta.label),
      React.createElement("span", { className: "material-symbols-outlined text-[13px] text-on-surface-variant" }, "close")
    );
  });

  return React.createElement(
    "div",
    { className: `tool-picker relative flex flex-col gap-2${state.toolPickerOpen ? " open" : ""}` },
    React.createElement(
      "div",
      { className: "tool-picker-trigger-row" },
      React.createElement(
        BeamFrame,
        { active: state.toolPickerOpen, tone: "mono", className: "tool-picker-trigger-shell rounded-xl" },
        React.createElement(
          "button",
          {
            type: "button",
            className: "tool-picker-trigger flex items-center gap-2 px-3 py-1.5 bg-surface-container-highest rounded-xl border border-outline-variant hover:bg-surface-variant transition-colors",
            onClick: toggleToolPicker,
            "aria-label": selected.size ? `Choose tools, ${selected.size} selected` : "Choose tools",
          },
          React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "TOOLS"),
          React.createElement(
            MotionShimmerText,
            { className: "tool-picker-trigger-copy", active: state.toolPickerOpen },
            routeSummary.trigger
          ),
          React.createElement("span", { className: "material-symbols-outlined text-[16px] text-on-surface-variant ml-auto" }, state.toolPickerOpen ? "expand_less" : "expand_more")
        )
      )
    ),
    selectedTools.length
      ? React.createElement(
          MotionReveal,
          null,
          React.createElement(
            "div",
            { className: "tool-picker-selected-row" },
            React.createElement("span", { className: "tool-picker-selected-label" }, "Active tools"),
            React.createElement("div", { className: "tool-picker-selected-list" }, selectedToolChips)
          )
        )
      : null,
    state.toolPickerOpen
      ? React.createElement(
          "div",
          { className: "tool-picker-panel absolute left-0 bottom-full mb-3 z-10 w-[33rem] max-w-[calc(100vw-2rem)] overflow-hidden" },
          React.createElement("div", { className: "tool-picker-panel-glow", "aria-hidden": "true" }),
          React.createElement(
            "div",
            { className: "tool-picker-panel-inner" },
            React.createElement(
              React.Fragment,
              null,
              React.createElement(
                "div",
                { className: "tool-picker-panel-header" },
                React.createElement(
                  "div",
                  { className: "tool-picker-panel-copy" },
                  React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, "Route this turn"),
                  React.createElement("span", { className: "text-[11px] leading-5 text-on-surface-variant" }, routeSummary.panel)
                ),
                React.createElement(
                  "button",
                  {
                    type: "button",
                    className: "font-label-caps text-label-caps text-primary bg-transparent border border-outline-variant rounded-full px-3 py-1 shrink-0 hover:bg-surface-container-high transition-colors",
                    onClick: clearToolSelection,
                  },
                  "Clear"
                )
              ),
              React.createElement(
                "div",
                { className: "tool-picker-route-note" },
                React.createElement("span", { className: "tool-picker-route-label" }, state.planMode ? "Plan mode" : "Auto route"),
                React.createElement("span", { className: "tool-picker-route-copy" }, state.planMode ? "The runtime will inspect the repo and return a flowchart only. Live route cards stay selected for normal turns after you exit plan mode." : "Leave the tray empty to let Devenv choose between memory, live search, and tool-assisted execution.")
              ),
              React.createElement(
                MotionDeck,
                { className: "tool-picker-sections" },
                groupedTools.map(([category, toolNames]) =>
                  React.createElement(
                    "section",
                    { key: category, className: "tool-picker-section" },
                    React.createElement(
                      "div",
                      { className: "tool-picker-section-copy" },
                      React.createElement("span", { className: "tool-picker-section-label" }, TOOL_CATEGORY_META[category]?.label || "Tools"),
                      React.createElement("span", { className: "tool-picker-section-detail" }, TOOL_CATEGORY_META[category]?.detail || "Route the runtime through these tools.")
                    ),
                    React.createElement(
                      "div",
                      { className: "tool-picker-grid" },
                      toolNames.map((toolName) => {
                        const meta = describeTool(toolName, toolReadiness[toolName]);
                        return React.createElement(
                          "button",
                          {
                            key: toolName,
                            type: "button",
                            className: `tool-picker-tile${selected.has(toolName) ? " is-selected" : ""}`,
                            onClick: () => toggleTool(toolName),
                          },
                          React.createElement(
                            "span",
                            { className: "tool-picker-tile-icon" },
                            React.createElement("span", { className: "material-symbols-outlined text-[18px] text-primary" }, meta.icon)
                          ),
                          React.createElement("span", { className: "tool-picker-tile-label" }, meta.label),
                          React.createElement("span", { className: "tool-picker-tile-hint" }, meta.hint),
                          selected.has(toolName)
                            ? React.createElement(
                                "span",
                                { className: "tool-picker-tile-check" },
                                React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "south")
                              )
                            : null
                        );
                      })
                    )
                  )
                )
              )
            )
          )
        )
      : null
  );
}

function groupToolsByCategory(toolNames) {
  const orderedCategories = ["workspace", "research", "artifacts"];
  return orderedCategories
    .map((category) => [
      category,
      toolNames.filter((toolName) => (TOOL_META[toolName]?.category || "workspace") === category),
    ])
    .filter(([, items]) => items.length);
}

function describeTool(toolName, readiness = {}) {
  const meta = TOOL_META[toolName] || {};
  const fallbackLabel = String(toolName || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
  return {
    icon: meta.icon || "build",
    label: meta.label || fallbackLabel || "Tool",
    hint: (typeof readiness.detail === "string" && readiness.detail.trim()) || meta.hint || "General workspace action",
  };
}

function summarizeRoute(selectedTools, planMode) {
  if (planMode) {
    return {
      trigger: "Plan flow active",
      panel: "Plan mode uses repo inspection and blueprint generation first. Route cards remain available for direct turns after planning.",
    };
  }
  if (!selectedTools.length) {
    return {
      trigger: "Auto route",
      panel: "Pick a route card to bias the runtime toward web, knowledge, prompt, or PDF work. Leave everything clear for automatic routing.",
    };
  }
  const workspaceRoutes = selectedTools.filter((toolName) => ["list_directory", "locate_files", "read_file", "search_text", "inspect_symbols", "track_symbol"].includes(toolName));
  if (workspaceRoutes.length && workspaceRoutes.length === selectedTools.length) {
    return {
      trigger: workspaceRoutes.length === 1 ? `${describeTool(workspaceRoutes[0]).label} route` : "Workspace route",
      panel: workspaceRoutes.length === 1
        ? `This turn is constrained to ${describeTool(workspaceRoutes[0]).label.toLowerCase()} inspection inside the repo.`
        : "This turn is constrained to repo inspection tools, which helps Devenv stay grounded in the current codebase before answering or planning.",
    };
  }
  if (selectedTools.length === 1) {
    const meta = describeTool(selectedTools[0]);
    return {
      trigger: `${meta.label} route`,
      panel: `This turn is biased toward ${meta.label.toLowerCase()} behavior. You can stack more route cards if the request needs multiple surfaces.`,
    };
  }
  return {
    trigger: `${selectedTools.length} routes active`,
    panel: "Multiple route cards are active, so the runtime will constrain itself to those selected surfaces where possible.",
  };
}
