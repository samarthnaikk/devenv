import React from "https://esm.sh/react@18";
import { AppProvider, useApp } from "./context/AppContext.js";
import { Header } from "./components/Header.js";
import { SettingsDropdown } from "./components/SettingsDropdown.js";
import { ChatColumn } from "./components/ChatColumn.js";
import { Sidebar } from "./components/Sidebar.js";
import { Toast } from "./components/Toast.js";
import { fetchHealth } from "./api.js";

function AppInner() {
  const { state, dispatch } = useApp();
  const clockRef = React.useRef(null);
  const healthRef = React.useRef(false);

  React.useEffect(() => {
    clockRef.current = window.setInterval(() => {
      dispatch({ type: "SET_CLOCK", payload: Date.now() });
    }, 1000);
    return () => window.clearInterval(clockRef.current);
  }, []);

  React.useEffect(() => {
    if (!state.clock) return;
    dispatch({
      type: "SET_USAGE_WINDOW",
      payload: state.usageWindow.filter((entry) => state.clock - entry.timestamp < 60000),
    });
    dispatch({
      type: "SET_RATE_LIMIT_INFO",
      payload: state.rateLimitInfo && state.rateLimitInfo.resetAt > state.clock ? state.rateLimitInfo : null,
    });
  }, [state.clock]);

  const pollingRef = React.useRef(null);

  React.useEffect(() => {
    if (healthRef.current) return;
    healthRef.current = true;
    fetchHealth()
      .then((payload) => {
        dispatch({ type: "SET_HEALTH", payload });
        applyHealthPayload(dispatch, payload);
      })
      .catch((error) => {
        dispatch({ type: "SET_BOOT_ERROR", payload: error.message });
      });
  }, []);

  React.useEffect(() => {
    const indexing = state.health?.indexing;
    if (!indexing?.active) {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }
    if (pollingRef.current) return;
    pollingRef.current = window.setInterval(() => {
      fetchHealth()
        .then((payload) => {
          dispatch({ type: "SET_HEALTH", payload });
          applyHealthPayload(dispatch, payload);
        })
        .catch(() => {});
    }, 2000);
    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [state.health?.indexing?.active]);

  React.useEffect(() => {
    const handler = (e) => {
      if (e.detail?.suggestion) {
        dispatch({ type: "SET_PROMPT", payload: e.detail.suggestion });
      }
    };
    window.addEventListener("opencode-suggestion", handler);
    return () => window.removeEventListener("opencode-suggestion", handler);
  }, []);

  React.useEffect(() => {
    if (state.showSettings) {
      const handler = (e) => {
        if (!e.target.closest("[data-settings-panel]") && !e.target.closest('[data-action="toggle-settings"]')) {
          dispatch({ type: "SET_SHOW_SETTINGS", payload: false });
        }
      };
      window.addEventListener("click", handler);
      return () => window.removeEventListener("click", handler);
    }
  }, [state.showSettings]);

  if (state.bootError) {
    return React.createElement("div", { className: "loading-shell" }, `Failed to load interface: ${state.bootError}`);
  }

  if (!state.health) {
    return React.createElement("div", { className: "loading-shell" }, "Booting Devenv web interface...");
  }

  const indexing = state.health.indexing || null;
  if (indexing?.active) {
    return React.createElement(StartupShell, { indexing, state, dispatch });
  }

  return React.createElement(
    "div",
    { className: "flex flex-col h-screen overflow-hidden bg-background" },
    React.createElement(Header, null),
    state.showSettings ? React.createElement(SettingsDropdown, null) : null,
    React.createElement(
      "main",
      { className: "flex flex-1 overflow-hidden" },
      React.createElement(ChatColumn, null),
      React.createElement(Sidebar, null)
    ),
    React.createElement(Toast, null)
  );
}

function StartupShell({ indexing }) {
  const percent = Math.max(0, Math.min(100, Number(indexing?.percent || 0)));
  const processed = Number(indexing?.processed_sessions || 0);
  const total = Number(indexing?.total_sessions || 0);
  const message = indexing?.message || "Preparing Devenv memory retrieval\u2026";
  const providerLabel = (indexing?.providers || []).length ? String(indexing.providers.join(" + ")).toUpperCase() : "LOCAL";

  return React.createElement(
    "div",
    { className: "loading-shell" },
    React.createElement(
      "div",
      { className: "startup-card" },
      React.createElement("div", { className: "font-label-caps text-label-caps text-on-surface-variant" }, providerLabel, " CHUNKING"),
      React.createElement("h1", { className: "font-headline-sm text-headline-sm text-on-surface", style: { margin: "8px 0 12px" } }, "Preparing session memory"),
      React.createElement("div", { className: "font-body-md text-body-md text-on-surface-variant" }, message),
      React.createElement(
        "div",
        { className: "startup-progress-track", style: { marginTop: "16px" } },
        React.createElement("div", { className: "startup-progress-fill", style: { width: `${percent}%` } })
      ),
      React.createElement(
        "div",
        { className: "flex gap-4 mt-3 font-body-md text-body-md text-on-surface-variant" },
        React.createElement("strong", { className: "text-on-surface" }, `${percent}%`),
        React.createElement("span", null, total ? `${processed}/${total} sessions` : "Counting sessions"),
        React.createElement("span", null, "ETA " + (indexing?.eta_seconds != null ? formatDuration(Number(indexing.eta_seconds) * 1000) : "Estimating\u2026"))
      )
    )
  );
}

function applyHealthPayload(dispatch, payload) {
  dispatch({
    type: "SET_HEALTH_META",
    payload: {
      provider: payload.ai_provider || "",
      model: payload.ai_model || "",
      availableModels: payload.available_models || [],
    },
  });
  dispatch({ type: "SET_ACCESS_POLICY", payload: payload.access_policy || { session_access: {}, backend_access: {} } });
  dispatch({ type: "SET_BACKENDS", payload: payload.ai_backends || {} });
  dispatch({ type: "SET_ACTIVE_BACKEND", payload: payload.active_backend || "opencode" });
  dispatch({ type: "SET_PERFORMANCE_MODE", payload: payload.performance_mode || "medium" });
  dispatch({ type: "SET_PRIVACY_MODE", payload: payload.privacy || { no_memory: false, incognito: false } });
}

function formatDuration(ms) {
  const totalSeconds = Math.max(Math.ceil(ms / 1000), 0);
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60);
  return minutes ? `${minutes}m ${String(seconds).padStart(2, "0")}s` : `${seconds}s`;
}

export function App() {
  return React.createElement(
    AppProvider,
    null,
    React.createElement(AppInner, null)
  );
}
