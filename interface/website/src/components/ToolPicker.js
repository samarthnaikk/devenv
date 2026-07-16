import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";

const TOOL_META = {
  generate_pdf: { icon: "picture_as_pdf", label: "PDF", hint: "Generate polished PDFs" },
  generate_prompt: { icon: "auto_awesome", label: "Prompt", hint: "Prepare a strong prompt" },
  knowledge_search: { icon: "hub", label: "Knowledge", hint: "Pull repos and references" },
  web_search: { icon: "language", label: "Web", hint: "Search live sources" },
};

const USER_VISIBLE_TOOLS = new Set(Object.keys(TOOL_META));

export function ToolPicker() {
  const { state, dispatch } = useApp();
  const availableTools = (Array.isArray(state.health?.tools) ? state.health.tools : []).filter((toolName) => USER_VISIBLE_TOOLS.has(toolName));
  const visibleSelectedTools = state.selectedTools.filter((toolName) => USER_VISIBLE_TOOLS.has(toolName));
  const selected = new Set(visibleSelectedTools);
  const [droppingTools, setDroppingTools] = React.useState([]);
  const label = selected.size ? `${selected.size} selected` : "All tools";
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
        "button",
        {
          type: "button",
          className: "tool-picker-trigger flex items-center gap-2 px-3 py-1.5 bg-surface-container-highest rounded-lg border border-outline-variant hover:bg-surface-variant transition-colors",
          onClick: toggleToolPicker,
          "aria-label": "Choose tools",
        },
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "TOOLS"),
        React.createElement("span", { className: "text-outline" }, "/"),
        React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface" }, label)
      )
    ),
    selectedTools.length
      ? React.createElement(
          "div",
          { className: "tool-picker-selected-row" },
          React.createElement("span", { className: "tool-picker-selected-label" }, "Active tools"),
          React.createElement("div", { className: "tool-picker-selected-list" }, selectedToolChips)
        )
      : null,
    state.toolPickerOpen
      ? React.createElement(
          "div",
          { className: "tool-picker-panel absolute left-0 bottom-full mb-3 z-10 w-[40rem] max-w-[calc(100vw-2rem)] overflow-hidden" },
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
                  React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, "Choose functions"),
                  React.createElement("span", { className: "text-[11px] leading-5 text-on-surface-variant" }, "Pick tool cards from the tray. Selected ones drop into your active tool row.")
                ),
                React.createElement(
                  "button",
                  {
                    type: "button",
                    className: "font-label-caps text-label-caps text-primary bg-transparent border border-outline-variant rounded-full px-3 py-1 shrink-0 hover:bg-surface-container-high transition-colors",
                    onClick: clearToolSelection,
                  },
                  "Use all"
                )
              ),
              React.createElement(
                "div",
                { className: "tool-picker-grid" },
                availableTools.map((toolName) => {
                  const meta = describeTool(toolName);
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
      : null
  );
}

function describeTool(toolName) {
  const meta = TOOL_META[toolName] || {};
  const fallbackLabel = String(toolName || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
  return {
    icon: meta.icon || "build",
    label: meta.label || fallbackLabel || "Tool",
    hint: meta.hint || "General workspace action",
  };
}
