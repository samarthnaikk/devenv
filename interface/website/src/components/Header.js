import React from "react";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel } from "../utils/format.js";
import { BeamFrame, MetalSurface, MotionBadge, MotionReveal, MotionShimmerText, MotionSwap } from "./MotionPrimitives.js";

export function Header() {
  const { state, dispatch } = useApp();
  const activeBackend = formatBackendLabel(state.activeBackend || state.preferredBackend || "opencode");
  const routeLabel = summarizeHeaderRoute(state.selectedTools, state.planMode);
  const statusLabel = state.isRunning ? "Live" : state.planMode ? "Plan" : "Direct";

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
    { className: "app-header flex justify-between items-center px-margin-desktop w-full z-50 shrink-0" },
    React.createElement(
      MotionReveal,
      { className: "min-w-0" },
      React.createElement(
        BeamFrame,
        { active: state.isRunning, tone: "ocean", className: "app-header-brand rounded-2xl px-3 py-2" },
        React.createElement(
          "div",
          { className: "flex items-center gap-4" },
          React.createElement(
            "div",
            { className: "app-header-mark" },
            React.createElement(
              MotionSwap,
              { className: "items-center justify-center" },
              React.createElement("span", { className: "material-symbols-outlined text-[18px]" }, state.isRunning ? "bolt" : "auto_awesome")
            )
          ),
          React.createElement(
            "div",
            { className: "flex flex-col gap-1 min-w-0" },
            React.createElement("span", { className: "font-headline-md text-headline-md font-bold text-on-surface" }, "Devenv"),
            React.createElement(
              "div",
              { className: "app-header-pills" },
              React.createElement(
                MotionBadge,
                { className: "app-header-pill app-header-pill-live", active: state.isRunning },
                state.isRunning ? "Live turn" : "Shell ready"
              ),
              React.createElement("span", { className: "app-header-pill" }, activeBackend),
              React.createElement("span", { className: "app-header-pill" }, `${statusLabel} mode`),
              React.createElement("span", { className: "app-header-pill" }, routeLabel)
            ),
            React.createElement(
              "div",
              { className: "app-header-statusline text-on-surface-variant" },
              React.createElement(
                MotionShimmerText,
                { active: state.isRunning, className: "app-header-statuscopy" },
                state.isRunning
                  ? `Running through ${activeBackend}`
                  : state.planMode
                    ? "Planning with grounded files and read-only tools first"
                    : describeHeaderStatus(state.selectedTools)
              )
            )
          )
        )
      )
    ),
    React.createElement(
      MotionReveal,
      { delay: 90 },
      React.createElement(
        MetalSurface,
        { className: "app-header-actions flex items-center gap-2 rounded-2xl px-2 py-2" },
        React.createElement(
          "button",
          {
            type: "button",
            "data-action": "toggle-settings",
            className: `app-header-button icon-button p-2 rounded-2xl text-on-surface-variant ${state.showSettings ? "is-active text-primary" : ""}`,
            onClick: toggleSettings,
            "aria-label": "Settings",
          },
          React.createElement("span", { className: "material-symbols-outlined text-[20px]" }, "settings")
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "app-header-button solid-button px-3.5 py-2 font-label-caps text-label-caps rounded-2xl",
            onClick: newThread,
          },
          "New"
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "app-header-button ghost-button px-3.5 py-2 font-label-caps text-label-caps rounded-2xl",
            onClick: copyThread,
          },
          "Copy"
        )
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

function summarizeHeaderRoute(selectedTools, planMode) {
  if (planMode) return "Repo plan";
  const tools = Array.isArray(selectedTools) ? selectedTools : [];
  if (!tools.length) return "Auto route";
  if (tools.includes("track_symbol")) return "Trace route";
  if (tools.includes("inspect_symbols")) return "Symbols route";
  if (tools.includes("search_text")) return "Search route";
  if (tools.includes("read_file")) return "Read route";
  if (tools.includes("locate_files")) return "Locate route";
  if (tools.includes("list_directory")) return "Files route";
  if (tools.includes("knowledge_search")) return "Knowledge route";
  if (tools.includes("web_search")) return "Web route";
  if (tools.includes("generate_pdf")) return "PDF route";
  if (tools.includes("generate_prompt")) return "Prompt route";
  return `${tools.length} routes`;
}

function describeHeaderStatus(selectedTools) {
  const tools = Array.isArray(selectedTools) ? selectedTools : [];
  if (!tools.length) return "Light shell, direct answers, and tools only when the task actually needs them";
  if (tools.includes("track_symbol")) return "Following one symbol through the codebase before answering";
  if (tools.includes("inspect_symbols")) return "Inspecting definitions, exports, and structure first";
  if (tools.includes("search_text")) return "Scanning the repo for strings and usage sites first";
  if (tools.includes("read_file")) return "Opening exact files before answering or planning";
  if (tools.includes("locate_files")) return "Finding the right files before deeper inspection";
  if (tools.includes("list_directory")) return "Mapping folders and workspace structure first";
  if (tools.includes("knowledge_search")) return "Pulling external references, repos, docs, and threads";
  if (tools.includes("web_search")) return "Biasing toward current web results and live facts";
  if (tools.includes("generate_pdf")) return "Preparing a polished PDF artifact instead of only chat output";
  if (tools.includes("generate_prompt")) return "Preparing a stronger prompt output for the task";
  return "Constraining the runtime to the selected surfaces";
}
