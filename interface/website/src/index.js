import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.js";

try {
  const storedTheme = window.localStorage.getItem("devenv-ui-theme");
  document.body.dataset.theme = storedTheme === "dark" ? "dark" : "light";
} catch {
  document.body.dataset.theme = "light";
}

const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);
root.render(React.createElement(React.StrictMode, null, React.createElement(App, null)));
