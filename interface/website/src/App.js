import React from "react";
import { AppProvider, useApp } from "./context/AppContext.js";
import { Header } from "./components/Header.js";
import { SettingsDropdown } from "./components/SettingsDropdown.js";
import { ChatColumn } from "./components/ChatColumn.js";
import { Sidebar } from "./components/Sidebar.js";
import { Toast } from "./components/Toast.js";
import { BeamFrame, MetalSurface, MotionDeck, MotionReveal, MotionShimmerText, MotionStage } from "./components/MotionPrimitives.js";
import { fetchHealth, updateSessionAccess as apiUpdateSessionAccess, updateBackendAccess as apiUpdateBackendAccess } from "./api.js";
import { loadPreferredBackend, persistAccess, persistPreferredModels, persistSetupState } from "./utils/storage.js";

function AppInner() {
  const { state, dispatch } = useApp();
  const clockRef = React.useRef(null);
  const healthRef = React.useRef(false);
  const pollingRef = React.useRef(null);
  const [setupDone, setSetupDone] = React.useState(state.setupComplete);
  const setupStartedRef = React.useRef(false);
  const restoreRef = React.useRef(false);

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
    if (!state.health || restoreRef.current) return;
    restoreRef.current = true;
    const restore = async () => {
      const persisted = state.persistedAccess || { session_access: {}, backend_access: {} };
      try {
        if (persisted.session_access?.codex && !state.accessPolicy.session_access?.codex) {
          const payload = await apiUpdateSessionAccess("codex", true);
          dispatch({ type: "SET_ACCESS_POLICY", payload });
        }
        if (persisted.session_access?.opencode && !state.accessPolicy.session_access?.opencode) {
          const payload = await apiUpdateSessionAccess("opencode", true);
          dispatch({ type: "SET_ACCESS_POLICY", payload });
        }
        for (const backend of ["opencode", "ollama", "codex"]) {
          if (persisted.backend_access?.[backend] && !state.accessPolicy.backend_access?.[backend]) {
            const payload = await apiUpdateBackendAccess(backend, true);
            dispatch({ type: "SET_ACCESS_POLICY", payload });
          }
        }
        const payload = await fetchHealth();
        dispatch({ type: "SET_HEALTH", payload });
        applyHealthPayload(dispatch, payload);
        persistAccess(payload.access_policy || persisted);
        persistPreferredModels(payload.selected_models_by_backend || {});
        if (Object.values(payload.access_policy?.session_access || {}).some(Boolean)) {
          dispatch({ type: "SET_SETUP_COMPLETE", payload: true });
          setSetupDone(true);
          persistSetupState(true);
        }
      } catch {
        restoreRef.current = false;
      }
    };
    restore();
  }, [state.health]);

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
        if (Array.isArray(e.detail.selectedTools)) {
          dispatch({ type: "SET_SELECTED_TOOLS", payload: e.detail.selectedTools });
        }
        if (typeof e.detail.planMode === "boolean") {
          dispatch({ type: "SET_PLAN_MODE", payload: e.detail.planMode });
        }
        dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
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
    return React.createElement(BootStateScreen, {
      icon: "error",
      eyebrow: "Startup interrupted",
      title: "Devenv hit a boot failure",
      body: state.bootError,
      tone: "ember",
      detail: "The web shell loaded, but the health handshake did not complete.",
    });
  }

  if (!state.health) {
    return React.createElement(BootStateScreen, {
      icon: "deployed_code",
      eyebrow: "Preparing workspace",
      title: "Booting Devenv web interface",
      body: "Restoring your local shell, checking runtime health, and bringing the workspace online.",
      tone: "ocean",
      loading: true,
    });
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
      onFinish: () => {
        setSetupDone(true);
        dispatch({ type: "SET_SETUP_COMPLETE", payload: true });
        persistSetupState(true);
      },
    });
  }

  return React.createElement(
    "div",
    { className: "app-shell flex flex-col h-screen overflow-hidden bg-background" },
    React.createElement("div", { className: "app-shell-noise", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-orbit app-shell-orbit-one", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-orbit app-shell-orbit-two", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-veil app-shell-veil-top", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-veil app-shell-veil-bottom", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-aura app-shell-aura-one", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-aura app-shell-aura-two", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-aura app-shell-aura-three", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-beam app-shell-beam-top", "aria-hidden": "true" }),
    React.createElement("div", { className: "app-shell-beam app-shell-beam-bottom", "aria-hidden": "true" }),
    React.createElement(Header, null),
    state.showSettings ? React.createElement(SettingsDropdown, null) : null,
    React.createElement(
      "main",
      { className: "app-main flex flex-1 overflow-hidden" },
      React.createElement(
        MotionStage,
        { axis: "y", className: "app-main-chat flex-1 min-w-0" },
        React.createElement(
          BeamFrame,
          { active: state.isRunning, tone: "ocean", className: "app-main-chat-shell" },
          React.createElement("div", { className: "app-main-chat-shell-orbit", "aria-hidden": "true" }),
          React.createElement("div", { className: "app-main-chat-shell-orbit app-main-chat-shell-orbit-two", "aria-hidden": "true" }),
          React.createElement(ChatColumn, null)
        )
      ),
      React.createElement(
        MotionStage,
        { axis: "x", delay: 110, className: "app-main-sidebar-stage" },
        React.createElement(Sidebar, null)
      )
    ),
    React.createElement(Toast, null)
  );
}

function ConsentScreen({ dispatch, accessPolicy, indexing, onFinish }) {
  const codexGranted = Boolean(accessPolicy.session_access?.codex);
  const opencodeGranted = Boolean(accessPolicy.session_access?.opencode);
  const anyGranted = codexGranted || opencodeGranted;

  const [phase, setPhase] = React.useState(() => {
    if (codexGranted && opencodeGranted) {
      return indexing?.active ? "indexing_opencode" : "all_done";
    }
    if (opencodeGranted) {
      return indexing?.active ? "indexing_opencode" : "opencode_done";
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
    } else if (opencodeGranted && indexing?.active) {
      addLog("Access granted for OpenCode");
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
  const opencodeDone = phase === "opencode_done" || phase === "all_done";

  return React.createElement(
    MotionStage,
    { className: "loading-shell loading-shell-setup", delay: 40 },
    React.createElement(
      BeamFrame,
      { tone: "ocean", className: "startup-frame" },
      React.createElement(
        MetalSurface,
        { className: "startup-card", style: { maxWidth: "760px" } },
      React.createElement(
        "div",
        { className: "flex flex-col gap-4 mb-5" },
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
            React.createElement("h1", { className: "font-headline-sm text-headline-sm text-on-surface", style: { margin: 0 } }, "Set up remembered access"),
            React.createElement("p", { className: "font-body-md text-body-md text-on-surface-variant", style: { margin: "2px 0 0" } }, "Grant provider access once, let Devenv restore it next time, and keep your session history ready.")
          )
        ),
        React.createElement(
          "div",
          { className: "grid md:grid-cols-3 gap-3" },
          startupFact("Remembered", anyGranted ? "Your last access choices will be restored automatically." : "Once granted, access can be restored on the next launch."),
          startupFact("Chunking", "Devenv indexes prior sessions in the background after access is granted."),
          startupFact("Local-first", "You can still prefer Ollama and keep web lookups or PDFs scoped per task.")
        ),
        anyGranted
          ? React.createElement(
              "div",
              { className: "startup-saved-banner" },
              React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "bookmark_added"),
              React.createElement(
                "div",
                { className: "flex flex-col gap-1 min-w-0" },
                React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, "Saved access found"),
                React.createElement(
                  "span",
                  { className: "text-[12px] leading-5 text-on-surface-variant" },
                  codexGranted && opencodeGranted
                    ? "Codex and OpenCode are already authorized for session grounding."
                    : codexGranted
                      ? "Codex is already authorized for session grounding."
                      : "OpenCode is already authorized for session grounding."
                )
              )
            )
          : null
      ),
      React.createElement(
        MotionDeck,
        { className: "startup-status-grid mb-5" },
        startupStatusChip("Provider", activeProvider),
        startupStatusChip("Phase", isChunking ? "Indexing" : anyGranted ? "Ready" : "Awaiting grant"),
        startupStatusChip("Progress", isChunking ? `${progress.percent}%` : anyGranted ? "Saved" : "0%")
      ),
      React.createElement(
        "div",
        { className: "startup-section-intro" },
        React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, "What happens here"),
        React.createElement(
          "p",
          { className: "text-[12px] leading-5 text-on-surface-variant", style: { margin: "6px 0 0" } },
          "Provider access unlocks prior-session retrieval. Once granted, Devenv can restore that access on the next launch and continue indexing in the background."
        )
      ),
      React.createElement(
        "div",
        { className: "grid md:grid-cols-[0.95fr,1.05fr] gap-4" },
        React.createElement(
          "div",
          { className: "space-y-3" },
          React.createElement(
            "div",
            { className: "font-label-caps text-label-caps text-outline mb-2" },
              "PROVIDERS"
            ),
          React.createElement(
            "div",
            { className: "bg-surface-container-low rounded-lg border border-outline-variant p-3 text-[12px] text-on-surface-variant" },
            anyGranted
              ? "A saved provider grant was already detected. You can finish setup immediately or grant the remaining provider."
              : "Grant the providers you want Devenv to reuse for past-session grounding."
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
                  "Grant a provider to begin indexing and restore session memory."
                )
          )
        )
      ),
      phase === "all_done" || anyGranted
        ? React.createElement(
            "button",
            {
              type: "button",
              className: "mt-5 w-full py-3 bg-primary text-on-primary rounded-xl font-label-caps text-label-caps font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2",
              onClick: onFinish,
            },
            anyGranted && phase !== "all_done" ? "Continue with saved access" : "Finish setup and continue",
            React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "arrow_forward")
          )
        : null
      )
    )
  );
}

function startupFact(label, body) {
  return React.createElement(
    "div",
    { className: "rounded-lg border border-outline-variant bg-surface-container-low p-3" },
    React.createElement("div", { className: "font-label-caps text-label-caps text-primary mb-1" }, label),
    React.createElement("div", { className: "text-[12px] leading-5 text-on-surface-variant" }, body)
  );
}

function startupStatusChip(label, value) {
  return React.createElement(
    "div",
    { className: "startup-status-chip" },
    React.createElement("span", { className: "startup-status-label" }, label),
    React.createElement("strong", { className: "startup-status-value" }, value)
  );
}

function setupRow(provider, label, granted, done, isActive, handleGrant) {
  const isButtonDisabled = granted || done || isActive;
  return React.createElement(
    "div",
    {
      key: provider,
      className: "startup-provider-row flex items-center justify-between p-3 bg-surface-container rounded-lg border " + (done || granted ? "border-primary/40" : "border-outline-variant"),
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
  const persistedPreferredBackend = loadPreferredBackend();
  const preferredBackend = persistedPreferredBackend || payload.preferred_backend || "opencode";
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
  dispatch({ type: "SET_PREFERRED_BACKEND", payload: preferredBackend });
  dispatch({ type: "SET_PERFORMANCE_MODE", payload: payload.performance_mode || "medium" });
  dispatch({ type: "SET_PRIVACY_MODE", payload: payload.privacy || { no_memory: false, incognito: false } });
}

function formatDuration(ms) {
  const totalSeconds = Math.max(Math.ceil(ms / 1000), 0);
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60);
  return minutes ? `${minutes}m ${String(seconds).padStart(2, "0")}s` : `${seconds}s`;
}

function BootStateScreen({ icon, eyebrow, title, body, detail, tone = "ocean", loading = false }) {
  return React.createElement(
    "div",
    { className: "loading-shell" },
    React.createElement(
      MotionReveal,
      { className: "loading-shell-panel", delay: 40 },
      React.createElement(
        BeamFrame,
        { tone, className: "startup-frame" },
        React.createElement(
          MetalSurface,
          { className: "startup-card startup-card-compact" },
          React.createElement(
            "div",
            { className: "startup-hero" },
            React.createElement(
              "div",
              { className: "startup-icon-shell" },
              React.createElement("span", { className: "material-symbols-outlined text-[22px]" }, icon)
            ),
            React.createElement(
              "div",
              { className: "startup-hero-copy" },
              React.createElement("div", { className: "startup-kicker" }, eyebrow),
              React.createElement("h1", { className: "startup-title font-headline-sm text-headline-sm text-on-surface" }, title),
              React.createElement(
                loading ? MotionShimmerText : "p",
                loading ? { className: "startup-copy is-booting" } : { className: "startup-copy" },
                body
              )
            )
          ),
          React.createElement(
            "div",
            { className: "startup-section-intro" },
            React.createElement("strong", { className: "font-label-caps text-label-caps text-on-surface" }, loading ? "Live boot status" : "Failure details"),
            React.createElement(
              "p",
              { className: "text-[12px] leading-5 text-on-surface-variant", style: { margin: "6px 0 0" } },
              detail || "Devenv is validating the local runtime before rendering the main workspace."
            )
          ),
          React.createElement(
            "div",
            { className: "startup-facts-grid" },
            startupFact("Runtime", loading ? "Health checks and provider wiring are running now." : "Retry after the backend health endpoint is reachable."),
            startupFact("UI shell", "The local light interface is mounted before chat history is restored."),
            startupFact("Local-first", "Ollama remains available once the runtime handshake succeeds.")
          ),
          React.createElement(
            MotionDeck,
            { className: "startup-status-grid mt-4" },
            startupStatusChip("Surface", loading ? "Booting" : "Error"),
            startupStatusChip("Memory", loading ? "Restoring" : "Paused"),
            startupStatusChip("Backend", loading ? "Checking" : "Retry needed")
          )
        )
      )
    )
  );
}

export function App() {
  return React.createElement(
    AppProvider,
    null,
    React.createElement(AppInner, null)
  );
}
