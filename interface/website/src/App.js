import React from "https://esm.sh/react@18";
import { AppProvider, useApp } from "./context/AppContext.js";
import { Header } from "./components/Header.js";
import { SettingsDropdown } from "./components/SettingsDropdown.js";
import { ChatColumn } from "./components/ChatColumn.js";
import { Sidebar } from "./components/Sidebar.js";
import { Toast } from "./components/Toast.js";
import { fetchHealth, updateSessionAccess as apiUpdateSessionAccess, updateBackendAccess as apiUpdateBackendAccess } from "./api.js";
import { persistAccess } from "./utils/storage.js";

function AppInner() {
  const { state, dispatch } = useApp();
  const clockRef = React.useRef(null);
  const healthRef = React.useRef(false);
  const pollingRef = React.useRef(null);
  const [setupDone, setSetupDone] = React.useState(false);
  const setupStartedRef = React.useRef(false);

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
    const hasAccess = Object.values(state.accessPolicy?.session_access || {}).some(Boolean);
    const needsPoll = hasAccess && indexing && !indexing.completed;
    if (!needsPoll) {
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
  }, [state.health?.indexing?.active, state.health?.indexing?.completed, state.accessPolicy?.session_access]);

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
  const hasAccess = Object.values(state.accessPolicy?.session_access || {}).some(Boolean);
  const needsSetup = !setupDone && indexing && !indexing.completed && (!hasAccess || indexing.active || Number(indexing.total_sessions || 0) > 0);
  if (needsSetup) {
    setupStartedRef.current = true;
  }
  const showSetup = needsSetup || (setupStartedRef.current && !setupDone);

  if (showSetup) {
    return React.createElement(ConsentScreen, {
      dispatch,
      accessPolicy: state.accessPolicy,
      indexing,
      onFinish: () => setSetupDone(true),
    });
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

function ConsentScreen({ dispatch, accessPolicy, indexing, onFinish }) {
  const codexGranted = Boolean(accessPolicy.session_access?.codex);
  const opencodeGranted = Boolean(accessPolicy.session_access?.opencode);

  const [phase, setPhase] = React.useState(() => {
    if (codexGranted && opencodeGranted) {
      return indexing?.active ? "indexing_opencode" : "all_done";
    }
    if (codexGranted) {
      return indexing?.active ? "indexing_codex" : "codex_done";
    }
    return "idle";
  });
  const [logs, setLogs] = React.useState([]);
  const logContainerRef = React.useRef(null);
  const lastPercentRef = React.useRef(0);
  const prevActiveRef = React.useRef(indexing?.active);
  const initRef = React.useRef(false);

  const addLog = React.useCallback((message, type) => {
    setLogs((prev) => [...prev, { id: Date.now() + (prev.length + 1), message, type: type || "info", ts: new Date().toLocaleTimeString() }]);
  }, []);

  React.useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    if (codexGranted && indexing?.active) {
      addLog("Access granted for Codex");
      addLog(`Chunking started: ${indexing.total_sessions || "?"} sessions`);
    }
  }, []);

  React.useEffect(() => {
    const prev = prevActiveRef.current;
    prevActiveRef.current = indexing?.active;
    const pct = Math.floor(Number(indexing?.percent) || 0);
    const proc = Number(indexing?.processed_sessions) || 0;
    const total = Number(indexing?.total_sessions) || 0;

    if ((phase === "indexing_codex" || phase === "indexing_opencode") && indexing?.active && !prev) {
      addLog("Access granted for " + (phase === "indexing_codex" ? "Codex" : "OpenCode"));
      addLog(`Chunking started: ${total || "?"} sessions`);
      lastPercentRef.current = pct;
    }

    if (indexing?.active && (phase === "indexing_codex" || phase === "indexing_opencode")) {
      if (pct > lastPercentRef.current) {
        addLog(`${proc}/${total} sessions chunked (${pct}%)`, "progress");
        lastPercentRef.current = pct;
      }
    }

    if (!indexing?.active && prev && (phase === "indexing_codex" || phase === "indexing_opencode")) {
      addLog("Chunking complete!", "done");
      const next = phase === "indexing_codex" ? "codex_done" : "all_done";
      window.setTimeout(() => setPhase(next), 600);
    }
  }, [indexing?.active, indexing?.percent, indexing?.processed_sessions]);

  React.useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleGrant = async (provider) => {
    setPhase(provider === "codex" ? "indexing_codex" : "indexing_opencode");
    if (provider === "opencode") {
      lastPercentRef.current = 0;
    }
    try {
      const payload = await apiUpdateSessionAccess(provider, true);
      dispatch({ type: "SET_ACCESS_POLICY", payload });
      persistAccess(payload);
      try {
        const bp = await apiUpdateBackendAccess("opencode", true);
        dispatch({ type: "SET_ACCESS_POLICY", payload: bp });
        persistAccess(bp);
      } catch {}
    } catch {
      addLog(`Failed to grant ${provider} access`, "error");
      setPhase("idle");
    }
  };

  const progress = {
    percent: Math.max(0, Math.min(100, Number(indexing?.percent || 0))),
    processed: Number(indexing?.processed_sessions || 0),
    total: Number(indexing?.total_sessions || 0),
    message: indexing?.message || "",
  };

  const isChunking = phase === "indexing_codex" || phase === "indexing_opencode";
  const activeProvider = phase === "idle" ? "\u2014" : phase === "indexing_codex" || phase === "codex_done" ? "Codex" : "OpenCode";
  const codexDone = phase === "codex_done" || phase === "all_done";
  const opencodeDone = phase === "all_done";

  return React.createElement(
    "div",
    { className: "loading-shell" },
    React.createElement(
      "div",
      { className: "startup-card", style: { maxWidth: "560px" } },
      React.createElement(
        "div",
        { className: "flex items-center justify-between mb-5" },
        React.createElement(
          "div",
          { className: "flex items-center gap-3" },
          React.createElement(
            "div",
            { className: "w-10 h-10 rounded-full bg-primary flex items-center justify-center" },
            React.createElement("span", { className: "material-symbols-outlined text-on-primary text-[20px]" }, "vpn_key")
          ),
          React.createElement(
            "div",
            null,
            React.createElement("h1", { className: "font-headline-sm text-headline-sm text-on-surface", style: { margin: 0 } }, "Set up memory access"),
            React.createElement("p", { className: "font-body-md text-body-md text-on-surface-variant", style: { margin: "2px 0 0" } }, "Grant access and chunk prior sessions")
          )
        )
      ),
      React.createElement(
        "div",
        { className: "grid grid-cols-2 gap-4" },
        React.createElement(
          "div",
          { className: "space-y-3" },
          React.createElement(
            "div",
            { className: "font-label-caps text-label-caps text-outline mb-2" },
            "PROVIDERS"
          ),
          setupRow("codex", "Codex", codexGranted, codexDone, isChunking && phase === "indexing_codex", handleGrant),
          setupRow("opencode", "OpenCode", opencodeGranted, opencodeDone, isChunking && phase === "indexing_opencode", handleGrant)
        ),
        React.createElement(
          "div",
          { className: "space-y-3" },
          React.createElement(
            "div",
            { className: "font-label-caps text-label-caps text-outline mb-2" },
            "PROGRESS"
          ),
          React.createElement(
            "div",
            { className: "bg-surface-container rounded-lg border border-outline-variant p-3", style: { minHeight: "180px", maxHeight: "220px", overflowY: "auto", display: "flex", flexDirection: "column" } },
            isChunking || codexDone || opencodeDone
              ? React.createElement(
                  React.Fragment,
                  null,
                  React.createElement(
                    "div",
                    { className: "font-label-caps text-label-caps text-on-surface mb-2" },
                    activeProvider,
                    isChunking ? " chunking" : " chunked"
                  ),
                  isChunking
                    ? React.createElement(
                        React.Fragment,
                        null,
                        React.createElement(
                          "div",
                          { className: "startup-progress-track", style: { marginBottom: "8px", height: "6px" } },
                          React.createElement("div", {
                            className: "startup-progress-fill",
                            style: { width: `${progress.percent}%`, height: "6px", transition: "width 0.8s ease" },
                          })
                        ),
                        React.createElement(
                          "div",
                          { className: "flex gap-3 font-code-sm text-code-sm text-on-surface-variant mb-3" },
                          React.createElement("span", { className: "text-on-surface font-bold" }, `${progress.percent}%`),
                          React.createElement("span", null, progress.total ? `${progress.processed}/${progress.total} sessions` : "Counting sessions"),
                          React.createElement("span", null, indexing?.eta_seconds != null ? `ETA ${formatDuration(Number(indexing.eta_seconds) * 1000)}` : "Estimating\u2026")
                        )
                      )
                    : React.createElement(
                        "div",
                        { className: "flex items-center gap-2 mb-3 text-primary font-label-caps text-label-caps" },
                        React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "check_circle"),
                        "Done"
                      ),
                  React.createElement(
                    "div",
                    { ref: logContainerRef, className: "space-y-1 flex-1", style: { overflowY: "auto" } },
                    logs.map((log) =>
                      React.createElement(
                        "div",
                        {
                          key: log.id,
                          className: "consent-log-row",
                          style: {
                            fontSize: "11px",
                            lineHeight: "1.5",
                            color: log.type === "done" ? "#4fdbc8" : log.type === "error" ? "#ffb4ab" : log.type === "progress" ? "#859490" : "#bbcac6",
                            padding: "2px 0",
                            display: "flex",
                            gap: "8px",
                          },
                        },
                        React.createElement("span", { style: { color: "#3c4947", flexShrink: 0, fontFamily: "JetBrains Mono, monospace" } }, log.ts),
                        log.message
                      )
                    )
                  )
                )
              : React.createElement(
                  "div",
                  { className: "flex items-center justify-center h-full text-on-surface-variant font-body-md text-body-md" },
                  "Grant a provider to begin"
                )
          )
        )
      ),
      phase === "all_done"
        ? React.createElement(
            "button",
            {
              type: "button",
              className: "mt-5 w-full py-3 bg-primary text-on-primary rounded-xl font-label-caps text-label-caps font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2",
              onClick: onFinish,
            },
            "Finish Setup",
            React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "arrow_forward")
          )
        : null
    )
  );
}

function setupRow(provider, label, granted, done, isActive, handleGrant) {
  const isButtonDisabled = granted || done || isActive;
  return React.createElement(
    "div",
    {
      key: provider,
      className: "flex items-center justify-between p-3 bg-surface-container rounded-lg border " + (done || granted ? "border-primary/40" : "border-outline-variant"),
    },
    React.createElement(
      "div",
      { className: "flex items-center gap-3" },
      React.createElement(
        "span",
        { className: "material-symbols-outlined text-[18px] " + (done ? "text-primary" : "text-outline") },
        done ? "check_circle" : "radio_button_unchecked"
      ),
      React.createElement(
        "div",
        null,
        React.createElement("div", { className: "font-body-md text-body-md text-on-surface" }, label),
        React.createElement("div", { className: "font-code-sm text-code-sm " + (done ? "text-primary" : "text-outline") }, done || granted ? "Granted" : "Not granted")
      )
    ),
    React.createElement(
      "button",
      {
        type: "button",
        className: "px-4 py-1.5 rounded-lg font-label-caps text-label-caps transition-colors " + (isButtonDisabled ? "bg-surface-variant text-on-surface-variant cursor-default" : "bg-primary text-on-primary hover:opacity-80"),
        onClick: () => !isButtonDisabled && handleGrant(provider),
        disabled: isButtonDisabled,
      },
      done ? "Done" : granted ? "Granted" : isActive ? "Processing\u2026" : "Grant"
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
      availableModelsByBackend: payload.available_models_by_backend || {},
      selectedModelsByBackend: payload.selected_models_by_backend || {},
    },
  });
  dispatch({ type: "SET_ACCESS_POLICY", payload: payload.access_policy || { session_access: {}, backend_access: { opencode: false, ollama: false, codex: false } } });
  dispatch({ type: "SET_BACKENDS", payload: payload.ai_backends || {} });
  dispatch({ type: "SET_ACTIVE_BACKEND", payload: payload.active_backend || "opencode" });
  dispatch({ type: "SET_PREFERRED_BACKEND", payload: payload.preferred_backend || "opencode" });
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
