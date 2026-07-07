import React from "https://esm.sh/react@18";
import { useApp } from "../context/AppContext.js";
import { formatBackendLabel } from "../utils/format.js";

export function Toast() {
  const { state } = useApp();
  if (!state.toast) return null;

  return React.createElement(
    "div",
    { className: "toast-banner markdown-body inline-markdown" },
    state.toast
  );
}
