import React from "react";
import { renderMarkdown } from "../lib/markdown.js";
import { BeamFrame, MotionReveal } from "./MotionPrimitives.js";

export function AssistantMessage({ message, onCopy, onReply }) {
  return React.createElement(
    MotionReveal,
    { className: "message-stack flex flex-col gap-2 max-w-3xl" },
    React.createElement(
      BeamFrame,
      { active: false, tone: "ocean", className: "message-card assistant-message-card rounded-[24px] p-4" },
      React.createElement(
        "div",
        { className: "flex items-center gap-2" },
        React.createElement(
          "div",
          { className: "message-avatar assistant-avatar w-7 h-7 rounded-full flex items-center justify-center" },
          React.createElement("span", { className: "material-symbols-outlined text-on-primary text-[14px]" }, "auto_awesome")
        ),
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "Devenv"),
        React.createElement("div", { className: "ml-auto flex items-center gap-1 message-actions" },
        React.createElement(
          "button",
          {
            type: "button",
            className: "message-action p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant",
            onClick: onReply,
            title: "Reply",
          },
          React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "reply")
        ),
        React.createElement(
          "button",
          {
            type: "button",
            className: "message-action p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant",
            onClick: onCopy,
            title: "Copy",
          },
          React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "content_copy")
        )
        )
      ),
      message.replyTo
        ? React.createElement(
            "div",
            { className: "message-reply mt-3 rounded-2xl border border-outline-variant/70 bg-surface-container px-3 py-2 text-[12px] text-on-surface-variant" },
            React.createElement("div", { className: "mb-1 font-label-caps text-label-caps text-primary" }, `Replying to ${message.replyTo.author}`),
            React.createElement("div", null, message.replyTo.excerpt)
          )
        : null,
      React.createElement(
        "div",
        {
          className: "font-body-lg text-body-lg text-on-surface leading-relaxed markdown-body assistant-markdown message-body",
          dangerouslySetInnerHTML: { __html: renderMarkdown(String(message.content || "")) },
        }
      )
    )
  );
}
