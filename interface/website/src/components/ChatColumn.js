import React from "https://esm.sh/react@18.2.0";
import { Transcript } from "./Transcript.js";
import { Composer } from "./Composer.js";

export function ChatColumn() {
  return React.createElement(
    "section",
    { className: "flex-1 min-w-0 flex flex-col h-full bg-background relative" },
    React.createElement(Transcript, null),
    React.createElement(Composer, null)
  );
}
