import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.js";

const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);
root.render(React.createElement(React.StrictMode, null, React.createElement(App, null)));
