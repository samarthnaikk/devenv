import React from "https://esm.sh/react@18.2.0";
import { renderMarkdown } from "../lib/markdown.js";

export function AssistantMessage({ message }) {
  return React.createElement(
    "div",
    { className: "flex flex-col gap-2 max-w-3xl" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2" },
      React.createElement(
        "div",
        { className: "w-6 h-6 rounded-full bg-primary flex items-center justify-center" },
        React.createElement("span", { className: "material-symbols-outlined text-on-primary text-[14px]" }, "smart_toy")
      ),
      React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "Devenv"),
      React.createElement(
        "button",
        {
          type: "button",
          className: "ml-auto p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant",
          "data-action": "copy-message",
          "data-message-id": message.id,
          title: "Copy",
        },
        React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "content_copy")
      )
    ),
    React.createElement(
      "div",
      {
        className: "font-body-lg text-body-lg text-on-surface ml-8 leading-relaxed markdown-body assistant-markdown",
        dangerouslySetInnerHTML: { __html: renderMarkdown(String(message.content || "")) },
      }
    )
  );
}
