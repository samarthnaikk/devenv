import React from "https://esm.sh/react@18.2.0";
import { useApp } from "../context/AppContext.js";

export function Header() {
  const { state, dispatch } = useApp();

  const toggleSettings = () => {
    dispatch({ type: "SET_SHOW_SETTINGS", payload: !state.showSettings });
  };

  const newThread = async () => {
    try {
      const { resetThread } = await import("../api.js");
      await resetThread();
    } catch {
      // backend reset is best-effort
    }
    dispatch({ type: "SET_PROMPT", payload: "" });
    dispatch({ type: "SET_TRANSCRIPT", payload: [] });
    dispatch({ type: "SET_TOOL_PICKER_OPEN", payload: false });
    dispatch({ type: "SET_SELECTED_TOOLS", payload: [] });
    dispatch({ type: "SET_PLAN_BLUEPRINT", payload: null });
    dispatch({ type: "SET_SESSION_USAGE_TOTAL", payload: 0 });
    dispatch({ type: "SET_LATEST_TURN_TOKENS", payload: 0 });
    dispatch({ type: "SET_LATEST_ELAPSED_MS", payload: 0 });
    dispatch({
      type: "SET_RETRIEVAL_STATUS",
      payload: { mode: "new_context", label: "New context", detail: "No prior Devenv session has been reused yet." },
    });
    showToast(dispatch, "Started a new retrieval thread");
  };

  const copyThread = async () => {
    if (!state.transcript.length) {
      showToast(dispatch, "Nothing to copy yet");
      return;
    }
    const transcriptText = state.transcript
      .map((entry) => {
        const role = entry.role === "user" ? "You" : "Devenv";
        return `${role}\n${String(entry.content || "").trim()}`;
      })
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(transcriptText);
      showToast(dispatch, "Thread copied");
    } catch {
      showToast(dispatch, "Clipboard access failed");
    }
  };

  return React.createElement(
    "header",
    { className: "flex justify-between items-center h-14 px-margin-desktop w-full z-50 bg-surface border-b border-outline-variant shrink-0" },
    React.createElement(
      "div",
      { className: "flex items-center gap-4" },
      React.createElement("span", { className: "font-headline-md text-headline-md font-bold text-on-surface" }, "Devenv")
    ),
    React.createElement(
      "div",
      { className: "flex items-center gap-3" },
      React.createElement(
        "button",
        {
          type: "button",
          "data-action": "toggle-settings",
          className: `p-2 rounded-lg hover:bg-surface-variant transition-colors text-on-surface-variant ${state.showSettings ? "bg-surface-variant text-primary" : ""}`,
          onClick: toggleSettings,
          "aria-label": "Settings",
        },
        React.createElement("span", { className: "material-symbols-outlined text-[20px]" }, "settings")
      ),
      React.createElement(
        "button",
        {
          type: "button",
          className: "px-3 py-1.5 font-label-caps text-label-caps bg-primary text-on-primary rounded-lg hover:opacity-80 transition-opacity",
          onClick: newThread,
        },
        "New"
      ),
      React.createElement(
        "button",
        {
          type: "button",
          className: "px-3 py-1.5 font-label-caps text-label-caps border border-outline-variant text-on-surface rounded-lg hover:bg-surface-variant transition-colors",
          onClick: copyThread,
        },
        "Copy"
      )
    )
  );
}

let toastTimeoutId = null;

export function showToast(dispatch, message) {
  dispatch({ type: "SET_TOAST", payload: message });
  if (toastTimeoutId) window.clearTimeout(toastTimeoutId);
  toastTimeoutId = window.setTimeout(() => {
    dispatch({ type: "SET_TOAST", payload: "" });
  }, 1600);
}
