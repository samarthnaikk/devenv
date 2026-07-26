import React from "react";
import { useApp } from "../context/AppContext.js";
import { escapeHtml, escapeAttribute } from "../utils/format.js";
import { showToast } from "./Header.js";
import { BeamFrame, MotionDeck, MotionReveal, MotionShimmerText } from "./MotionPrimitives.js";

export function SessionsCard() {
  const { state, dispatch } = useApp();
  const codexAllowed = Boolean(state.accessPolicy.session_access?.codex);
  const opencodeAllowed = Boolean(state.accessPolicy.session_access?.opencode);
  const codexVisible = Boolean(state.visibleSessionProviders?.codex);
  const opencodeVisible = Boolean(state.visibleSessionProviders?.opencode);

  const toggleProviderVisibility = (provider) => {
    const nextVisible = !state.visibleSessionProviders[provider];
    dispatch({ type: "SET_VISIBLE_SESSION_PROVIDERS", payload: { provider, visible: nextVisible } });
    if (!nextVisible && state.selectedProvider === provider) {
      dispatch({ type: "SET_SELECTED_SESSION_ID", payload: "" });
    }
    if (nextVisible && state.accessPolicy.session_access?.[provider] && !state.providerSessions[provider].length) {
      refreshProviderSessions(provider, state, dispatch);
    }
  };

  const selectSession = (provider, sessionId) => {
    dispatch({ type: "SET_SELECTED_PROVIDER", payload: provider });
    dispatch({ type: "SET_SELECTED_SESSION_ID", payload: sessionId });
    refreshSelectedSession(provider, sessionId, state, dispatch);
  };

  const refreshAllSessions = async () => {
    const providers = ["codex", "opencode"].filter(
      (p) => state.visibleSessionProviders[p] && state.accessPolicy.session_access?.[p]
    );
    for (const provider of providers) {
      await refreshProviderSessions(provider, state, dispatch);
    }
    showToast(dispatch, providers.length ? "Refreshed open session lists" : "Open a provider list to load its sessions");
  };

  return React.createElement(
    "section",
    { className: "workspace-card-stack space-y-3" },
    React.createElement(
      MotionReveal,
      null,
      React.createElement(
        "div",
        { className: "workspace-card-title-row flex justify-between items-center" },
        React.createElement(
          "div",
          { className: "min-w-0" },
          React.createElement(
            "h3",
            { className: "font-label-caps text-label-caps text-on-surface-variant flex items-center gap-2" },
            React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "history"),
            "SESSIONS"
          ),
          React.createElement(
            MotionShimmerText,
            { className: "workspace-card-title-meta", active: state.sessionLoading },
            state.sessionLoading ? "Refreshing visible providers" : "Codex and OpenCode thread history"
          )
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "p-1 hover:text-primary transition-colors text-on-surface-variant",
            onClick: refreshAllSessions,
            disabled: state.sessionLoading,
          },
          React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, "refresh")
        )
      )
    ),
    React.createElement(
      "div",
      { className: "workspace-summary-rail workspace-summary-rail-secondary" },
      React.createElement("span", { className: "workspace-summary-pill" }, `${(state.providerSessions?.codex || []).length + (state.providerSessions?.opencode || []).length} loaded`),
      React.createElement("span", { className: "workspace-summary-pill" }, state.selectedProvider ? escapeHtml(state.selectedProvider) : "No selection"),
      React.createElement("span", { className: "workspace-summary-pill" }, codexVisible || opencodeVisible ? "Rail open" : "Rail idle"),
      React.createElement("span", { className: "workspace-summary-copy" }, "Open provider histories, inspect threads, and keep retrieval context visible.")
    ),
    React.createElement(
      BeamFrame,
      { active: state.sessionLoading, tone: "mono", className: "workspace-card-shell rounded-[22px] overflow-hidden" },
      React.createElement("span", { className: "workspace-card-ribbon workspace-card-ribbon-secondary", "aria-hidden": "true" }),
      React.createElement(
        MotionDeck,
        { className: "workspace-provider-grid p-3 border-b border-outline-variant/30" },
        providerStat("Codex", codexAllowed, (state.providerSessions?.codex || []).length, codexVisible),
        providerStat("OpenCode", opencodeAllowed, (state.providerSessions?.opencode || []).length, opencodeVisible)
      ),
      renderSessionRow("codex", "Codex History", codexAllowed, codexVisible, state, dispatch, toggleProviderVisibility, selectSession),
      renderSessionRow("opencode", "OpenCode History", opencodeAllowed, opencodeVisible, state, dispatch, toggleProviderVisibility, selectSession)
    )
  );
}

function renderSessionRow(provider, label, allowed, visible, state, dispatch, toggleProviderVisibility, selectSession) {
  const sessions = state.providerSessions[provider] || [];
  return React.createElement(
    "div",
    { key: provider, className: "workspace-card-section workspace-session-section bg-surface-container rounded-lg border border-outline-variant overflow-hidden" },
    React.createElement(
      "div",
      { className: "p-3 flex justify-between items-center gap-3" },
      React.createElement(
        "div",
        { className: "workspace-session-headline" },
        React.createElement("span", { className: "font-body-md text-body-md" }, escapeHtml(label)),
        React.createElement("span", { className: "workspace-session-meta" }, allowed ? visible ? "Open in rail" : "Available to inspect" : "Grant access to inspect")
      ),
      React.createElement(
        "button",
        {
          type: "button",
          className: "px-3 py-1 rounded bg-surface-variant text-on-surface font-label-caps text-[10px] hover:bg-primary hover:text-on-primary transition-colors",
          onClick: () => toggleProviderVisibility(provider),
          disabled: !allowed,
        },
        visible ? "Hide" : "Show"
      )
    ),
    visible && allowed
      ? React.createElement(
          "div",
          { className: "border-t border-outline-variant/30 p-2 space-y-1 max-h-48 overflow-y-auto workspace-session-list" },
          sessions.length
            ? sessions.map((session) =>
                React.createElement(
                  "button",
                  {
                    key: session.session_id,
                    type: "button",
                    className: `w-full text-left p-2 rounded-lg workspace-session-card ${state.selectedProvider === provider && state.selectedSessionId === session.session_id ? "is-selected bg-surface-container-highest border border-primary" : "bg-surface-dim border border-transparent"} hover:bg-surface-container-highest transition-colors`,
                    onClick: () => selectSession(provider, session.session_id),
                  },
                  React.createElement("div", { className: "font-label-caps text-label-caps text-on-surface text-[11px]" }, escapeHtml(session.title || "Untitled session")),
                  React.createElement("div", { className: "font-code-sm text-code-sm text-on-surface-variant truncate" }, escapeHtml(session.updated_at || session.workspace_path || ""))
                )
              )
            : React.createElement("div", { className: "font-body-md text-body-md text-on-surface-variant p-2" }, state.sessionLoading ? "Loading..." : "No sessions")
        )
      : null
  );
}

function providerStat(label, allowed, count, visible) {
  return React.createElement(
    "div",
    { className: "workspace-provider-stat" },
    React.createElement("span", { className: "workspace-provider-label" }, label),
    React.createElement("strong", { className: "workspace-provider-value" }, allowed ? `${count}` : "Off"),
    React.createElement("span", { className: "workspace-provider-detail" }, allowed ? visible ? "Visible now" : "Ready to open" : "Needs consent")
  );
}

async function refreshProviderSessions(provider, state, dispatch) {
  if (!state.accessPolicy.session_access?.[provider]) {
    dispatch({ type: "SET_PROVIDER_SESSIONS", payload: { provider, sessions: [] } });
    return;
  }
  dispatch({ type: "SET_SESSION_LOADING", payload: true });
  try {
    const { fetchContextSessions } = await import("../api.js");
    const payload = await fetchContextSessions(provider);
    dispatch({ type: "SET_PROVIDER_SESSIONS", payload: { provider, sessions: payload.sessions || [] } });
    if (!state.selectedSessionId && payload.sessions?.[0]) {
      dispatch({ type: "SET_SELECTED_PROVIDER", payload: provider });
      dispatch({ type: "SET_SELECTED_SESSION_ID", payload: payload.sessions[0].session_id });
      refreshSelectedSession(provider, payload.sessions[0].session_id, state, dispatch);
    }
  } finally {
    dispatch({ type: "SET_SESSION_LOADING", payload: false });
  }
}

async function refreshSelectedSession(provider, sessionId, state, dispatch) {
  if (!provider || !sessionId || !state.accessPolicy.session_access?.[provider]) return;
  try {
    const { fetchContextSession } = await import("../api.js");
    const payload = await fetchContextSession(provider, sessionId);
    dispatch({ type: "SET_SESSION_DETAILS", payload: { key: `${provider}:${sessionId}`, details: payload } });
  } catch {
    // silently fail
  }
}
