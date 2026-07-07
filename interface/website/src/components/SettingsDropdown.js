import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, escapeAttribute, formatBackendLabel } from "../utils/format.js";

export function SettingsDropdown() {
  const { state, dispatch } = useApp();
  const models = state.healthMeta.availableModels.length
    ? state.healthMeta.availableModels
    : [state.healthMeta.model || "opencode/claude-sonnet-4"];
  const currentModel = state.healthMeta.model || models[0];

  const handleModelChange = async (event) => {
    const model = event.target.value;
    if (!model) return;
    try {
      const { updateModel } = await import("../api.js");
      const result = await updateModel(model);
      dispatch({ type: "SET_HEALTH_META", payload: { provider: result.ai_provider || state.healthMeta.provider, model: result.ai_model || model, availableModels: result.available_models || state.healthMeta.availableModels } });
      const { showToast } = await import("./Header.js");
      showToast(dispatch, "Model switched to " + model.split("/").pop());
    } catch (err) {
      const { showToast } = await import("./Header.js");
      showToast(dispatch, "Failed to switch model: " + err.message);
    }
  };

  const closeSettings = () => {
    dispatch({ type: "SET_SHOW_SETTINGS", payload: false });
  };

  return React.createElement(
    "div",
    { className: "relative z-40" },
    React.createElement(
      "div",
      {
        className: "absolute right-4 top-0 w-72 bg-surface-container-high border border-outline-variant rounded-xl shadow-2xl p-4 space-y-4",
        "data-settings-panel": true,
      },
      React.createElement(
        "div",
        { className: "flex items-center justify-between" },
        React.createElement("h3", { className: "font-label-caps text-label-caps text-on-surface-variant" }, "Settings"),
        React.createElement(
          "button",
          {
            type: "button",
            className: "p-1 rounded hover:bg-surface-variant transition-colors text-on-surface-variant",
            onClick: closeSettings,
            "aria-label": "Close settings",
          },
          React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "close")
        )
      ),
      React.createElement(
        "div",
        { className: "space-y-1.5" },
        React.createElement("label", { className: "font-label-caps text-[11px] text-on-surface-variant block" }, "Model"),
        React.createElement(
          "select",
          {
            className: "w-full bg-surface-container-highest border border-outline-variant rounded-lg font-body-md text-body-md text-on-surface p-2 outline-none focus:border-primary",
            value: currentModel,
            onChange: handleModelChange,
          },
          models.map((m) =>
            React.createElement("option", { key: m, value: m }, m)
          )
        )
      ),
      React.createElement(
        "div",
        { className: "pt-2 border-t border-outline-variant/30" },
        React.createElement("div", { className: "font-label-caps text-[11px] text-on-surface-variant" }, "Backend"),
        React.createElement("div", { className: "font-body-md text-body-md text-on-surface mt-0.5" }, formatBackendLabel(state.activeBackend))
      )
    )
  );
}
