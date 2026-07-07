const STORAGE_THEME_KEY = "devenv-ui-theme";
const STORAGE_ACCESS_KEY = "devenv-ui-access";
const STORAGE_BACKEND_KEY = "devenv-ui-backend";

export function loadTheme() {
  const forcedTheme = new URLSearchParams(window.location.search).get("theme");
  if (forcedTheme === "dark" || forcedTheme === "light") {
    return forcedTheme;
  }
  try {
    return window.localStorage.getItem(STORAGE_THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function persistTheme(theme) {
  try {
    window.localStorage.setItem(STORAGE_THEME_KEY, theme);
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
