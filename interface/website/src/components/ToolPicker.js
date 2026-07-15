import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { escapeHtml } from "../utils/format.js";

const TOOL_META = {
  audit_changes: { icon: "history", label: "Audit" },
  edit_file: { icon: "edit_note", label: "Edit" },
  generate_prompt: { icon: "auto_awesome", label: "Prompt" },
  inspect_symbols: { icon: "account_tree", label: "Symbols" },
  inspect_trace: { icon: "timeline", label: "Trace" },
  list_directory: { icon: "folder", label: "Folders" },
  locate_files: { icon: "find_in_page", label: "Files" },
  manage_memory: { icon: "memory", label: "Memory" },
  peek_lines: { icon: "subject", label: "Peek" },
  read_file: { icon: "description", label: "Read" },
  remove_file: { icon: "delete", label: "Delete" },
  run_diagnostics: { icon: "health_and_safety", label: "Checks" },
  run_shell: { icon: "terminal", label: "Shell" },
  search_text: { icon: "search", label: "Search" },
  track_symbol: { icon: "conversion_path", label: "Track" },
  web_search: { icon: "language", label: "Web" },
  write_file: { icon: "note_add", label: "Write" },
};

export function ToolPicker() {
  const { state, dispatch } = useApp();
  const availableTools = Array.isArray(state.health?.tools) ? state.health.tools : [];
  const selected = new Set(state.selectedTools);
  const label = selected.size ? `${selected.size} selected` : "All tools";

  const toggleToolPicker = () => {
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: !state.toolPickerOpen });
  };

  const clearToolSelection = () => {
    dispatch({ type: "SET_SELECTED_TOOLS", payload: [] });
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
  };

  const toggleTool = (toolName) => {
    const next = new Set(state.selectedTools);
    if (next.has(toolName)) {
      next.delete(toolName);
    } else {
      next.add(toolName);
    }
    dispatch({ type: "SET_SELECTED_TOOLS", payload: Array.from(next).sort() });
  };

  const selectedTools = Array.from(selected);

  return React.createElement(
    "div",
    { className: `relative flex flex-col gap-2${state.toolPickerOpen ? " open" : ""}` },
    React.createElement(
      "div",
      { className: "flex items-center gap-2 flex-wrap" },
      React.createElement(
        "button",
        {
          type: "button",
          className: "flex items-center gap-2 px-3 py-1.5 bg-surface-container-highest rounded-lg border border-outline-variant hover:bg-surface-variant transition-colors",
          onClick: toggleToolPicker,
          "aria-label": "Choose tools",
        },
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "TOOLS"),
        React.createElement("span", { className: "text-outline" }, "/"),
        React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface" }, label)
      ),
      selectedTools.length
        ? selectedTools.map((toolName) => {
            const meta = describeTool(toolName);
            return React.createElement(
              "button",
              {
                key: `selected-${toolName}`,
                type: "button",
                className: "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-primary/40 bg-surface text-on-surface hover:bg-surface-container-high transition-colors",
                onClick: () => toggleTool(toolName),
                title: `Remove ${toolName}`,
                "aria-label": `Remove ${toolName}`,
              },
              React.createElement("span", { className: "material-symbols-outlined text-[15px] text-primary" }, meta.icon),
              React.createElement("span", { className: "font-label-caps text-[10px] uppercase tracking-[0.08em]" }, meta.label),
              React.createElement("span", { className: "material-symbols-outlined text-[13px] text-on-surface-variant" }, "close")
            );
          })
        : null
    ),
    state.toolPickerOpen
      ? React.createElement(
          "div",
          { className: "absolute left-0 bottom-full mb-2 z-10 w-80 max-h-80 overflow-auto border border-outline-variant rounded-lg bg-surface-container p-3 shadow-xl" },
          React.createElement(
            "div",
            { className: "flex items-center justify-between mb-2" },
            React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, "Choose functions"),
            React.createElement(
              "button",
              {
                type: "button",
                className: "font-label-caps text-label-caps text-primary bg-transparent border-0",
                onClick: clearToolSelection,
              },
              "Use all"
            )
          ),
          React.createElement(
            "div",
            { className: "flex flex-col gap-1.5" },
            availableTools.map((toolName) => {
              const meta = describeTool(toolName);
              return React.createElement(
                "button",
                {
                  key: toolName,
                  type: "button",
                  className: `flex items-center justify-between w-full px-3 py-2 rounded-md border text-left font-body-md text-body-md text-on-surface hover:bg-surface-container-higher transition-colors ${selected.has(toolName) ? "border-primary bg-surface" : "border-outline-variant bg-surface-dim"}`,
                  onClick: () => toggleTool(toolName),
                },
                React.createElement(
                  "span",
                  { className: "flex items-center gap-2 min-w-0" },
                  React.createElement("span", { className: "material-symbols-outlined text-[18px] text-primary shrink-0" }, meta.icon),
                  React.createElement(
                    "span",
                    { className: "flex flex-col min-w-0" },
                    React.createElement("span", { className: "font-body-md text-body-md text-on-surface truncate" }, meta.label),
                    React.createElement("span", { className: "text-[10px] text-on-surface-variant font-code-sm truncate" }, escapeHtml(toolName))
                  )
                ),
                selected.has(toolName)
                  ? React.createElement("span", { className: "material-symbols-outlined text-[18px] text-primary" }, "check_circle")
                  : null
              );
            })
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
  };
}
