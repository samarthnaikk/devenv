import React from "react";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, escapeAttribute, formatBackendLabel } from "../utils/format.js";
import { persistPreferredBackend, persistPreferredModels, persistTheme } from "../utils/storage.js";
import { BeamFrame, MotionBadge, MotionDeck, MotionReveal, MotionShimmerText, MotionTilt } from "./MotionPrimitives.js";

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
  const routeSummary = state.planMode
    ? "Plan mode keeps the next turn in blueprint-first flow."
    : !state.selectedTools.length
      ? "Auto route lets Devenv choose memory, tools, or live search."
      : `${state.selectedTools.length} route${state.selectedTools.length === 1 ? "" : "s"} will constrain the next turn.`;

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
      persistPreferredModels(result.selected_models_by_backend || { ...state.healthMeta.selectedModelsByBackend, [preferredBackend]: model });
      const { showToast } = await import("./Header.js");
      showToast(dispatch, `${formatBackendLabel(preferredBackend)} model switched to ${model.split("/").pop()}`);
    } catch (err) {
      const { showToast } = await import("./Header.js");
      showToast(dispatch, "Failed to switch model: " + err.message);
    }
  };

  const handleBackendChange = (event) => {
    const next = event.target.value || "opencode";
    dispatch({ type: "SET_PREFERRED_BACKEND", payload: next });
    persistPreferredBackend(next);
  };

  const closeSettings = () => {
    dispatch({ type: "SET_SHOW_SETTINGS", payload: false });
  };

  const toggleTheme = () => {
    const nextTheme = state.theme === "dark" ? "light" : "dark";
    dispatch({ type: "SET_THEME", payload: nextTheme });
    persistTheme(nextTheme);
  };

  return React.createElement(
    "div",
    { className: "relative z-40" },
    React.createElement(
      MotionReveal,
      null,
      React.createElement(
        BeamFrame,
        {
          active: false,
          tone: "mono",
          className: "settings-panel-shell absolute right-4 top-0 w-72 rounded-[22px] p-4 space-y-4",
          "data-settings-panel": true,
        },
        React.createElement("span", { className: "settings-panel-ribbon", "aria-hidden": "true" }),
        React.createElement("span", { className: "settings-panel-orbit settings-panel-orbit-one", "aria-hidden": "true" }),
        React.createElement("span", { className: "settings-panel-orbit settings-panel-orbit-two", "aria-hidden": "true" }),
        React.createElement(
          "div",
          { className: "flex items-center justify-between" },
          React.createElement(
            "div",
            { className: "flex flex-col gap-1" },
            React.createElement("h3", { className: "font-label-caps text-label-caps text-on-surface-variant" }, "Settings"),
            React.createElement(
              MotionShimmerText,
              { className: "settings-panel-copy", active: state.isRunning },
              "Preferred backend and model routing"
            )
          ),
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
          { className: "settings-panel-section space-y-1.5" },
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
          MotionDeck,
          { className: "settings-preview-deck" },
          React.createElement(
            MotionTilt,
            null,
            React.createElement(
              "div",
              { className: "settings-preview-card" },
              React.createElement("span", { className: "settings-preview-kicker" }, "Route"),
              React.createElement("strong", { className: "settings-preview-title" }, state.planMode ? "Blueprint first" : "Adaptive runtime"),
              React.createElement("p", { className: "settings-preview-copy" }, routeSummary)
            )
          ),
          React.createElement(
            MotionTilt,
            null,
            React.createElement(
              "div",
              { className: "settings-preview-card settings-preview-card-theme" },
              React.createElement("span", { className: "settings-preview-kicker" }, "Theme"),
              React.createElement("strong", { className: "settings-preview-title" }, state.theme === "light" ? "Light shell active" : "Dark shell active"),
              React.createElement(
                "div",
                { className: "settings-preview-swatches", "aria-hidden": "true" },
                React.createElement("span", { className: "settings-preview-swatch settings-preview-swatch-ocean" }),
                React.createElement("span", { className: "settings-preview-swatch settings-preview-swatch-mint" }),
                React.createElement("span", { className: "settings-preview-swatch settings-preview-swatch-paper" })
              )
            )
          )
        ),
        React.createElement(
          MotionDeck,
          { className: "settings-status-grid" },
          React.createElement(MotionTilt, null, statusChip("Preferred", formatBackendLabel(preferredBackend))),
          React.createElement(MotionTilt, null, statusChip("Active", formatBackendLabel(state.activeBackend || preferredBackend))),
          React.createElement(MotionTilt, null, statusChip("Model", compactModelName(currentModel)))
        ),
        React.createElement(
          "div",
          { className: "settings-panel-section space-y-3" },
          React.createElement(
            "div",
            { className: "flex items-center justify-between gap-3" },
            React.createElement(
              "div",
              { className: "flex flex-col gap-1 min-w-0" },
              React.createElement("span", { className: "font-label-caps text-[11px] text-on-surface-variant block" }, "Theme"),
              React.createElement(
                MotionShimmerText,
                { className: "settings-theme-copy", active: state.theme === "light" },
                state.theme === "light" ? "Light shell with glass, beam, and paper highlights" : "Dark shell with calmer contrast and the same motion system"
              )
            ),
            React.createElement(
              "button",
              {
                type: "button",
                className: `theme-toggle${state.theme === "light" ? " is-light" : " is-dark"}`,
                onClick: toggleTheme,
                "aria-label": `Switch to ${state.theme === "light" ? "dark" : "light"} theme`,
              },
              React.createElement(
                "span",
                { className: "theme-toggle-track" },
                React.createElement("span", { className: "material-symbols-outlined theme-toggle-icon theme-toggle-icon-light" }, "light_mode"),
                React.createElement("span", { className: "material-symbols-outlined theme-toggle-icon theme-toggle-icon-dark" }, "dark_mode"),
                React.createElement("span", { className: "theme-toggle-thumb" })
              )
            )
          ),
          React.createElement(
            "div",
            { className: "settings-theme-row" },
            themeChip("Light", "Default shell", state.theme === "light"),
            themeChip("Dark", "Optional contrast", state.theme === "dark")
          )
        ),
        React.createElement(
          "div",
          { className: "settings-panel-section space-y-1.5" },
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
          { className: "settings-panel-section pt-2 border-t border-outline-variant/30" },
          React.createElement("div", { className: "font-label-caps text-[11px] text-on-surface-variant" }, "Backend"),
          React.createElement("div", { className: "font-body-md text-body-md text-on-surface mt-0.5" }, `${formatBackendLabel(state.activeBackend)} active`),
          React.createElement("div", { className: "settings-panel-footnote" }, routeSummary)
        )
      )
    )
  );
}

function statusChip(label, value) {
  return React.createElement(
    "div",
    { className: "settings-status-chip" },
    React.createElement("span", { className: "settings-status-label" }, label),
    React.createElement("strong", { className: "settings-status-value" }, value)
  );
}

function themeChip(label, detail, active) {
  return React.createElement(
    MotionBadge,
    { className: `settings-theme-chip${active ? " is-selected" : ""}`, active },
    React.createElement("strong", null, label),
    React.createElement("span", null, detail)
  );
}

function compactModelName(value) {
  const text = String(value || "").trim();
  if (!text) return "Auto";
  const last = text.split("/").pop() || text;
  return last.length > 18 ? `${last.slice(0, 15)}...` : last;
}
