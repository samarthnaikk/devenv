const STORAGE_THEME_KEY = "devenv-ui-theme";
const STORAGE_ACCESS_KEY = "devenv-ui-access";
const STORAGE_BACKEND_KEY = "devenv-ui-backend";
const STORAGE_MODELS_KEY = "devenv-ui-models";
const STORAGE_SETUP_KEY = "devenv-ui-setup";

export function loadTheme() {
  return "dark";
}

export function persistTheme(theme) {
  try {
    window.localStorage.setItem(STORAGE_THEME_KEY, "dark");
  } catch {}
}

export function loadPersistedAccess() {
  try {
    const raw = window.localStorage.getItem(STORAGE_ACCESS_KEY);
    if (!raw) return { session_access: {}, backend_access: {} };
    const payload = JSON.parse(raw);
    return typeof payload === "object" && payload ? payload : { session_access: {}, backend_access: {} };
  } catch {
    return { session_access: {}, backend_access: {} };
  }
}

export function persistAccess(accessPolicy) {
  try {
    window.localStorage.setItem(STORAGE_ACCESS_KEY, JSON.stringify(accessPolicy));
  } catch {}
}

export function persistPreferredBackend(value) {
  try {
    window.localStorage.setItem(STORAGE_BACKEND_KEY, value);
  } catch {}
}

export function loadPreferredBackend() {
  try {
    return window.localStorage.getItem(STORAGE_BACKEND_KEY) || "opencode";
  } catch {
    return "opencode";
  }
}

export function persistPreferredModels(value) {
  try {
    window.localStorage.setItem(STORAGE_MODELS_KEY, JSON.stringify(value || {}));
  } catch {}
}

export function loadPreferredModels() {
  try {
    const raw = window.localStorage.getItem(STORAGE_MODELS_KEY);
    if (!raw) return {};
    const payload = JSON.parse(raw);
    return typeof payload === "object" && payload ? payload : {};
  } catch {
    return {};
  }
}

export function persistSetupState(value) {
  try {
    window.localStorage.setItem(STORAGE_SETUP_KEY, value ? "done" : "pending");
  } catch {}
}

export function loadSetupState() {
  try {
    return window.localStorage.getItem(STORAGE_SETUP_KEY) === "done";
  } catch {
    return false;
  }
}
