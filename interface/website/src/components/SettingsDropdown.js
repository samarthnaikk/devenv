import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, escapeAttribute, formatBackendLabel } from "../utils/format.js";

export function SettingsDropdown() {
  const { state, dispatch } = useApp();
  const preferredBackend = state.preferredBackend || "opencode";
  const backendModelMap = state.healthMeta.selectedModelsByBackend || {};
  const backendCatalog = state.healthMeta.availableModelsByBackend || {};
  const models = backendCatalog[preferredBackend]?.length
    ? backendCatalog[preferredBackend]
    : state.healthMeta.availableModels.length
      ? state.healthMeta.availableModels
      : [backendModelMap[preferredBackend] || state.healthMeta.model || "opencode/claude-sonnet-4"];
  const currentModel = backendModelMap[preferredBackend] || state.healthMeta.model || models[0];
  const backendStatus = state.backends?.[preferredBackend] || null;
  const backendDetail = backendStatus?.detail || "";
  const isBackendAvailable = backendStatus ? backendStatus.available !== false : true;

  const handleModelChange = async (event) => {
    const model = event.target.value;
    if (!model) return;
    try {
      const { updateModel } = await import("../api.js");
      const result = await updateModel(model, preferredBackend);
      dispatch({
        type: "SET_HEALTH_META",
        payload: {
          provider: result.ai_provider || state.healthMeta.provider,
          model: result.ai_model || model,
          availableModels: result.available_models || state.healthMeta.availableModels,
          availableModelsByBackend: result.available_models_by_backend || state.healthMeta.availableModelsByBackend,
          selectedModelsByBackend: result.selected_models_by_backend || state.healthMeta.selectedModelsByBackend,
        },
      });
      const { showToast } = await import("./Header.js");
      showToast(dispatch, `${formatBackendLabel(preferredBackend)} model switched to ${model.split("/").pop()}`);
    } catch (err) {
      const { showToast } = await import("./Header.js");
      showToast(dispatch, "Failed to switch model: " + err.message);
    }
  };

  const handleBackendChange = (event) => {
    dispatch({ type: "SET_PREFERRED_BACKEND", payload: event.target.value || "opencode" });
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
        React.createElement("label", { className: "font-label-caps text-[11px] text-on-surface-variant block" }, "Preferred Backend"),
        React.createElement(
          "select",
          {
            className: "w-full bg-surface-container-highest border border-outline-variant rounded-lg font-body-md text-body-md text-on-surface p-2 outline-none focus:border-primary",
            value: preferredBackend,
            onChange: handleBackendChange,
          },
          ["opencode", "ollama", "codex"].map((backend) =>
            React.createElement("option", { key: backend, value: backend }, formatBackendLabel(backend))
          )
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
              disabled: !isBackendAvailable,
            },
          models.map((m) =>
            React.createElement("option", { key: m, value: m }, m)
          )
        ),
        React.createElement(
          "div",
          { className: `text-[11px] ${isBackendAvailable ? "text-on-surface-variant" : "text-error"}` },
          isBackendAvailable
            ? backendDetail || `Choose a ${formatBackendLabel(preferredBackend)} model.`
            : backendDetail || `${formatBackendLabel(preferredBackend)} is unavailable.`
        )
      ),
      React.createElement(
        "div",
        { className: "pt-2 border-t border-outline-variant/30" },
        React.createElement("div", { className: "font-label-caps text-[11px] text-on-surface-variant" }, "Backend"),
        React.createElement("div", { className: "font-body-md text-body-md text-on-surface mt-0.5" }, `${formatBackendLabel(state.activeBackend)} active`)
      )
    )
  );
}
