import React from "https://esm.sh/react@18";
import { loadTheme, loadPersistedAccess } from "../utils/storage.js";

const READ_ONLY_TOOLS = ["list_directory", "read_file", "glob", "grep", "inspect_symbols", "search_symbols"];

const initialState = {
  health: null,
  prompt: "",
  transcript: [],
  isRunning: false,
  bootError: "",
  healthMeta: { provider: "", model: "", availableModels: [], availableModelsByBackend: {}, selectedModelsByBackend: {} },
  usageWindow: [],
  rateLimitInfo: null,
  clock: Date.now(),
  theme: loadTheme(),
  toast: "",
  retrievalStatus: {
    mode: "new_context",
    label: "New context",
    detail: "No prior Devenv session has been reused yet.",
  },
  accessPolicy: { session_access: { codex: false, opencode: false }, backend_access: { opencode: false } },
  persistedAccess: loadPersistedAccess(),
  backends: {},
  activeBackend: "opencode",
  preferredBackend: "opencode",
  selectedProvider: "codex",
  visibleSessionProviders: { codex: false, opencode: false },
  providerSessions: { codex: [], opencode: [] },
  selectedSessionId: "",
  sessionDetails: {},
  sessionLoading: false,
  accessUpdating: false,
  performanceMode: "medium",
  privacyMode: { no_memory: false, incognito: false },
  sessionBudgetTokens: 25000,
  budgetInput: "25000",
  sessionUsageTotal: 0,
  latestTurnTokens: 0,
  latestElapsedMs: 0,
  runStartedAt: 0,
  healthRefreshPending: false,
  pendingRunMode: "memory",
  selectedTools: [],
  toolPickerOpen: false,
  planMode: false,
  planBlueprint: null,
  showSettings: false,
};

function appReducer(state, action) {
  switch (action.type) {
    case "SET_HEALTH":
      return { ...state, health: action.payload };
    case "SET_BOOT_ERROR":
      return { ...state, bootError: action.payload };
    case "SET_HEALTH_META":
      return { ...state, healthMeta: action.payload };
    case "SET_PROMPT":
      return { ...state, prompt: action.payload };
    case "SET_TRANSCRIPT":
      return { ...state, transcript: action.payload };
    case "APPEND_TRANSCRIPT":
      return { ...state, transcript: [...state.transcript, action.payload] };
    case "UPDATE_TRANSCRIPT_ENTRY":
      return {
        ...state,
        transcript: state.transcript.map((entry) =>
          entry.id === action.payload.id ? { ...entry, ...action.payload.updates } : entry
        ),
      };
    case "SET_IS_RUNNING":
      return { ...state, isRunning: action.payload };
    case "SET_RUN_STARTED_AT":
      return { ...state, runStartedAt: action.payload };
    case "SET_CLOCK":
      return { ...state, clock: action.payload };
    case "SET_USAGE_WINDOW":
      return { ...state, usageWindow: action.payload };
    case "SET_RATE_LIMIT_INFO":
      return { ...state, rateLimitInfo: action.payload };
    case "SET_TOAST":
      return { ...state, toast: action.payload };
    case "SET_THEME":
      return { ...state, theme: action.payload };
    case "SET_ACCESS_POLICY":
      return { ...state, accessPolicy: action.payload };
    case "SET_ACCESS_UPDATING":
      return { ...state, accessUpdating: action.payload };
    case "SET_BACKENDS":
      return { ...state, backends: action.payload };
    case "SET_ACTIVE_BACKEND":
      return { ...state, activeBackend: action.payload };
    case "SET_PREFERRED_BACKEND":
      return { ...state, preferredBackend: action.payload };
    case "SET_SELECTED_PROVIDER":
      return { ...state, selectedProvider: action.payload };
    case "SET_VISIBLE_SESSION_PROVIDERS":
      return { ...state, visibleSessionProviders: { ...state.visibleSessionProviders, [action.payload.provider]: action.payload.visible } };
    case "SET_PROVIDER_SESSIONS":
      return { ...state, providerSessions: { ...state.providerSessions, [action.payload.provider]: action.payload.sessions } };
    case "SET_SELECTED_SESSION_ID":
      return { ...state, selectedSessionId: action.payload };
    case "SET_SESSION_DETAILS":
      return { ...state, sessionDetails: { ...state.sessionDetails, [action.payload.key]: action.payload.details } };
    case "SET_SESSION_LOADING":
      return { ...state, sessionLoading: action.payload };
    case "SET_PERFORMANCE_MODE":
      return { ...state, performanceMode: action.payload };
    case "SET_PRIVACY_MODE":
      return { ...state, privacyMode: { ...state.privacyMode, ...action.payload } };
    case "SET_BUDGET_TOKENS":
      return { ...state, sessionBudgetTokens: action.payload };
    case "SET_BUDGET_INPUT":
      return { ...state, budgetInput: action.payload };
    case "SET_SESSION_USAGE_TOTAL":
      return { ...state, sessionUsageTotal: action.payload };
    case "SET_LATEST_TURN_TOKENS":
      return { ...state, latestTurnTokens: action.payload };
    case "SET_LATEST_ELAPSED_MS":
      return { ...state, latestElapsedMs: action.payload };
    case "SET_HEALTH_REFRESH_PENDING":
      return { ...state, healthRefreshPending: action.payload };
    case "SET_PENDING_RUN_MODE":
      return { ...state, pendingRunMode: action.payload };
    case "SET_SELECTED_TOOLS":
      return { ...state, selectedTools: action.payload };
    case "SET_TOOL_PICKER_OPEN":
      return { ...state, toolPickerOpen: action.payload };
    case "SET_PLAN_MODE":
      return { ...state, planMode: action.payload, planBlueprint: action.payload ? state.planBlueprint : null };
    case "SET_PLAN_BLUEPRINT":
      return { ...state, planBlueprint: action.payload };
    case "SET_SHOW_SETTINGS":
      return { ...state, showSettings: action.payload };
    case "SET_RETRIEVAL_STATUS":
      return { ...state, retrievalStatus: action.payload };
    default:
      return state;
  }
}

const AppContext = React.createContext(null);

export function AppProvider({ children }) {
  const [state, dispatch] = React.useReducer(appReducer, initialState);

  React.useEffect(() => {
    document.body.dataset.theme = state.theme;
  }, [state.theme]);

  const value = React.useMemo(() => ({ state, dispatch }), [state]);

  return React.createElement(AppContext.Provider, { value }, children);
}

export function useApp() {
  const context = React.useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within AppProvider");
  }
  return context;
}

export { initialState, READ_ONLY_TOOLS };
