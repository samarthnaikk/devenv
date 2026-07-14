import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, formatBackendLabel } from "../utils/format.js";
import { persistAccess } from "../utils/storage.js";
import { showToast } from "./Header.js";

export function AccessCard() {
  const { state, dispatch } = useApp();
  const codexAllowed = Boolean(state.accessPolicy.session_access?.codex);
  const opencodeSessionAllowed = Boolean(state.accessPolicy.session_access?.opencode);
  const opencodeBackendAllowed = Boolean(state.accessPolicy.backend_access?.opencode);
  const ollamaBackendAllowed = Boolean(state.accessPolicy.backend_access?.ollama);
  const codexBackendAllowed = Boolean(state.accessPolicy.backend_access?.codex);
  const activeBackendLabel = formatBackendLabel(state.activeBackend);

  const updateSessionAccess = async (provider, allowed) => {
    dispatch({ type: "SET_ACCESS_UPDATING", payload: true });
    try {
      const { updateSessionAccess: apiUpdate } = await import("../api.js");
      const payload = await apiUpdate(provider, allowed);
      dispatch({ type: "SET_ACCESS_POLICY", payload });
      persistAccess(payload);
      if (!allowed) {
        dispatch({ type: "SET_PROVIDER_SESSIONS", payload: { provider, sessions: [] } });
        dispatch({ type: "SET_VISIBLE_SESSION_PROVIDERS", payload: { provider, visible: false } });
        if (state.selectedProvider === provider) {
          dispatch({ type: "SET_SELECTED_SESSION_ID", payload: "" });
        }
      }
      await refreshHealth(dispatch);
      showToast(dispatch, `${provider} session access ${allowed ? "granted" : "revoked"}`);
    } finally {
      dispatch({ type: "SET_ACCESS_UPDATING", payload: false });
    }
  };

  const updateBackendAccess = async (backend, allowed) => {
    dispatch({ type: "SET_ACCESS_UPDATING", payload: true });
    try {
      const { updateBackendAccess: apiUpdate } = await import("../api.js");
      const payload = await apiUpdate(backend, allowed);
      dispatch({ type: "SET_ACCESS_POLICY", payload });
      persistAccess(payload);
      await refreshHealth(dispatch);
      showToast(dispatch, `${formatBackendLabel(backend)} backend ${allowed ? "enabled" : "disabled"}`);
    } finally {
      dispatch({ type: "SET_ACCESS_UPDATING", payload: false });
    }
  };

  const handlePerformanceChange = async (event) => {
    const value = event.target.value || "medium";
    try {
      const { updatePerformance } = await import("../api.js");
      const payload = await updatePerformance(value);
      dispatch({ type: "SET_PERFORMANCE_MODE", payload: payload.performance_mode || value });
      await refreshHealth(dispatch, { silent: true });
      showToast(dispatch, `Performance set to ${payload.performance_mode || value}`);
    } catch {
      showToast(dispatch, "Failed to update performance mode");
    }
  };

  const handleIncognitoToggle = async (event) => {
    const incognito = Boolean(event.target.checked);
    try {
      const { updatePrivacy } = await import("../api.js");
      const payload = await updatePrivacy({ no_memory: incognito, incognito });
      dispatch({ type: "SET_PRIVACY_MODE", payload: payload.privacy || { no_memory: incognito, incognito } });
      await refreshHealth(dispatch, { silent: true });
      showToast(dispatch, incognito ? "Incognito mode enabled" : "Incognito mode disabled");
    } catch {
      showToast(dispatch, "Failed to update privacy mode");
    }
  };

  const handlePlanToggle = (event) => {
    const enabled = Boolean(event.target.checked);
    dispatch({ type: "SET_PLAN_MODE", payload: enabled });
    showToast(dispatch, enabled ? "Plan mode enabled — Devenv will generate a plan only" : "Plan mode disabled");
  };

  return React.createElement(
    "section",
    { className: "space-y-3" },
    React.createElement(
      "h3",
      { className: "font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2" },
      React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "vpn_key"),
      "ACCESS & PROVIDERS"
    ),
    React.createElement(
      "div",
      { className: "bg-surface-container rounded-lg border border-outline-variant overflow-hidden" },
      React.createElement(
        "div",
        { className: "p-3 border-b border-outline-variant/30 flex justify-between items-center" },
        React.createElement("span", { className: "font-body-md text-body-md" }, "Consent"),
        React.createElement("span", { className: "text-primary material-symbols-outlined text-[18px]" }, "check_circle")
      ),
      renderProviderRow("codex", "Codex", codexAllowed, "session", state, updateSessionAccess),
      renderProviderRow("opencode", "OpenCode", opencodeSessionAllowed, "session", state, updateSessionAccess),
      renderBackendRow("opencode", opencodeBackendAllowed, activeBackendLabel, state, updateBackendAccess),
      renderBackendRow("ollama", ollamaBackendAllowed, activeBackendLabel, state, updateBackendAccess),
      renderBackendRow("codex", codexBackendAllowed, activeBackendLabel, state, updateBackendAccess)
    ),
    React.createElement(
      "div",
      { className: "space-y-3 pt-2" },
      React.createElement(
        "div",
        { className: "flex flex-col gap-1.5" },
        React.createElement("label", { className: "font-label-caps text-label-caps text-on-surface-variant" }, "PERFORMANCE MODE"),
        React.createElement(
          "select",
          {
            className: "bg-surface-container-highest border border-outline-variant rounded-lg font-body-md text-body-md p-2 outline-none focus:border-primary",
            value: state.performanceMode,
            onChange: handlePerformanceChange,
          },
          React.createElement("option", { value: "low" }, "Low"),
          React.createElement("option", { value: "medium" }, "Med"),
          React.createElement("option", { value: "high" }, "High")
        )
      ),
      React.createElement(
        "label",
        { className: "flex items-center gap-3 cursor-pointer" },
        React.createElement("input", {
          className: "w-4 h-4 rounded border-outline-variant bg-surface-container text-primary focus:ring-0 focus:ring-offset-0",
          type: "checkbox",
          checked: state.privacyMode.incognito,
          onChange: handleIncognitoToggle,
        }),
        React.createElement("span", { className: "font-body-md text-body-md" }, "Incognito")
      ),
      React.createElement(
        "label",
        { className: "flex items-center gap-3 cursor-pointer" },
        React.createElement("input", {
          className: "w-4 h-4 rounded border-outline-variant bg-surface-container text-primary focus:ring-0 focus:ring-offset-0",
          type: "checkbox",
          checked: state.planMode,
          onChange: handlePlanToggle,
        }),
        React.createElement(
          "div",
          { className: "flex flex-col" },
          React.createElement("span", { className: "font-body-md text-body-md" }, "Plan mode"),
          React.createElement("span", { className: "text-[10px] text-on-surface-variant" }, "Generate a flowchart plan only")
        )
      )
    )
  );
}

function renderProviderRow(provider, label, allowed, type, state, updateSessionAccess) {
  const actionAttr = allowed ? "revoke" : "grant";
  return React.createElement(
    "div",
    { key: provider, className: "p-3 border-b border-outline-variant/30 flex justify-between items-center" },
    React.createElement(
      "div",
      { className: "flex flex-col" },
      React.createElement("span", { className: "font-body-md text-body-md" }, escapeHtml(label)),
      React.createElement("span", { className: `text-[10px] uppercase font-bold ${allowed ? "text-primary" : "text-outline"}` }, allowed ? "Granted" : "Not granted")
    ),
    React.createElement(
      "button",
      {
        type: "button",
        className: `px-3 py-1 rounded font-label-caps text-[10px] transition-colors ${allowed ? "bg-surface-variant text-on-surface hover:bg-error hover:text-on-error" : "bg-primary text-on-primary hover:opacity-80"}`,
        onClick: () => updateSessionAccess(provider, actionAttr === "grant"),
        disabled: state.accessUpdating,
      },
      allowed ? "Revoke" : "Grant"
    )
  );
}

function renderBackendRow(backend, allowed, activeBackendLabel, state, updateBackendAccess) {
  const label = formatBackendLabel(backend);
  const isActive = activeBackendLabel === label;
  return React.createElement(
    "div",
    { className: "p-3 border-b border-outline-variant/30 flex justify-between items-center" },
    React.createElement(
      "div",
      { className: "flex flex-col" },
      React.createElement("span", { className: "font-body-md text-body-md" }, label),
      React.createElement(
        "span",
        { className: "text-[10px] text-outline uppercase font-bold" },
        isActive ? "Active" : allowed ? "Enabled" : "Disabled"
      )
    ),
    React.createElement(
      "button",
      {
        type: "button",
        className: `px-3 py-1 rounded font-label-caps text-[10px] transition-colors ${allowed ? "bg-surface-variant text-on-surface hover:bg-error hover:text-on-error" : "bg-primary text-on-primary hover:opacity-80"}`,
        onClick: () => updateBackendAccess(backend, !allowed),
        disabled: state.accessUpdating,
      },
      allowed ? "Revoke" : "Grant"
    )
  );
}

async function refreshHealth(dispatch, options = {}) {
  try {
    const { fetchHealth } = await import("../api.js");
    const healthPayload = await fetchHealth();
    dispatch({ type: "SET_HEALTH", payload: healthPayload });
    dispatch({
      type: "SET_HEALTH_META",
      payload: {
        provider: healthPayload.ai_provider || "",
        model: healthPayload.ai_model || "",
        availableModels: healthPayload.available_models || [],
        availableModelsByBackend: healthPayload.available_models_by_backend || {},
        selectedModelsByBackend: healthPayload.selected_models_by_backend || {},
      },
    });
    dispatch({ type: "SET_ACCESS_POLICY", payload: healthPayload.access_policy || { session_access: {}, backend_access: { opencode: false, ollama: false, codex: false } } });
    dispatch({ type: "SET_BACKENDS", payload: healthPayload.ai_backends || {} });
    dispatch({ type: "SET_ACTIVE_BACKEND", payload: healthPayload.active_backend || "opencode" });
    dispatch({ type: "SET_PREFERRED_BACKEND", payload: healthPayload.preferred_backend || "opencode" });
    dispatch({ type: "SET_PERFORMANCE_MODE", payload: healthPayload.performance_mode || "medium" });
    dispatch({ type: "SET_PRIVACY_MODE", payload: healthPayload.privacy || { no_memory: false, incognito: false } });
  } catch (err) {
    if (!options.silent) {
      showToast(dispatch, "Health check failed");
    }
  }
}
