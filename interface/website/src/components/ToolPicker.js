import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { escapeHtml } from "../utils/format.js";

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

  return React.createElement(
    "div",
    { className: `relative${state.toolPickerOpen ? " open" : ""}` },
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
    state.toolPickerOpen
      ? React.createElement(
          "div",
          { className: "absolute left-0 bottom-full mb-2 z-10 w-72 max-h-80 overflow-auto border border-outline-variant rounded-lg bg-surface-container p-3 shadow-xl" },
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
            availableTools.map((toolName) =>
              React.createElement(
                "button",
                {
                  key: toolName,
                  type: "button",
                  className: `flex items-center justify-between w-full px-3 py-2 rounded-md border text-left font-body-md text-body-md text-on-surface hover:bg-surface-container-higher transition-colors ${selected.has(toolName) ? "border-primary bg-surface" : "border-outline-variant bg-surface-dim"}`,
                  onClick: () => toggleTool(toolName),
                },
                React.createElement("span", null, escapeHtml(toolName)),
                React.createElement("span", { className: "text-primary" }, selected.has(toolName) ? "check_circle" : "")
              )
            )
          )
        )
      : null
  );
}
