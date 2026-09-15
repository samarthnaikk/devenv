import React from "react";
import { Transcript } from "./Transcript.js";
import { Composer } from "./Composer.js";

export function ChatColumn() {
  return React.createElement(
    "section",
    { className: "chat-column flex-1 min-w-0 flex flex-col bg-background relative" },
    React.createElement("div", { className: "chat-column-grid", "aria-hidden": "true" }),
    React.createElement(Transcript, null),
    React.createElement(Composer, null)
  );
}
