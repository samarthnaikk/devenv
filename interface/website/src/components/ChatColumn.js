import React from "https://esm.sh/react@18";
import { Transcript } from "./Transcript.js";
import { Composer } from "./Composer.js";

export function ChatColumn() {
  return React.createElement(
    "section",
    { className: "w-[65%] flex flex-col h-full bg-background relative border-r border-outline-variant" },
    React.createElement(Transcript, null),
    React.createElement(Composer, null)
  );
}
