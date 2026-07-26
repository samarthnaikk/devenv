import React from "react";
import { escapeHtml } from "../utils/format.js";
import { BeamFrame, MetalSurface, MotionBadge, MotionReveal } from "./MotionPrimitives.js";

export function ErrorMessage({ message, onCopy, onReply }) {
  const diagnostics = message.diagnostics || {};
  const railPills = [
    diagnostics.sourceLabel || "Needs attention",
    diagnostics.routeLabel,
    diagnostics.backendLabel,
  ].filter(Boolean).slice(0, 3);

  return React.createElement(
    MotionReveal,
    { className: "error-message-shell max-w-3xl", delay: 110 },
    React.createElement(
      BeamFrame,
      { tone: "ember", className: "error-message-beam" },
      React.createElement(
        MetalSurface,
        { className: "error-message-panel" },
        React.createElement(
          "div",
          { className: "flex items-center gap-2" },
          React.createElement(
            "div",
            { className: "error-message-icon" },
            React.createElement("span", { className: "material-symbols-outlined text-[14px] text-on-error" }, "error")
          ),
          React.createElement(MotionBadge, { className: "error-message-pill", active: true }, diagnostics.badgeLabel || "Error"),
          React.createElement("div", { className: "ml-auto flex items-center gap-1" },
            React.createElement(
              "button",
              {
                type: "button",
                className: "p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant",
                onClick: onReply,
                title: "Reply",
              },
              React.createElement("span", { className: "material-symbols-outlined text-[16px]" }, "reply")
            ),
            React.createElement(
              "button",
              {
                type: "button",
                className: "p-1 rounded hover:bg-surface-container transition-colors text-on-surface-variant",
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
              { className: "error-message-reply" },
              React.createElement("div", { className: "mb-1 font-label-caps text-label-caps text-error" }, `Replying to ${message.replyTo.author}`),
              React.createElement("div", null, message.replyTo.excerpt)
            )
          : null,
        React.createElement(
          "div",
          { className: "error-message-rail" },
          ...railPills.map((pill) => React.createElement("span", { key: pill, className: "error-message-rail-pill" }, pill)),
          React.createElement("span", { className: "error-message-rail-copy" }, diagnostics.detail || "This response surfaced an execution or validation issue instead of a normal answer.")
        ),
        React.createElement(
          "div",
          {
            className: "font-body-lg text-body-lg text-error error-message-copy",
            dangerouslySetInnerHTML: { __html: renderRichText(message.content) },
          }
        )
      )
    )
  );
}

function renderRichText(content) {
  const text = String(content || "");
  if (text.includes("```")) {
    return text
      .split(/```/)
      .map((chunk, index) => (index % 2 ? `<pre><code>${escapeHtml(chunk.replace(/^\w+\n/, ""))}</code></pre>` : renderParagraphs(chunk)))
      .join("");
  }
  return renderParagraphs(text);
}

function renderParagraphs(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (/^#{1,3}\s+/.test(trimmed)) {
        return trimmed
          .split("\n")
          .map((line) => {
            const match = line.match(/^(#{1,3})\s+(.*)$/);
            if (!match) return `<p>${renderInlineMarkdown(line)}</p>`;
            return `<h${match[1].length}>${renderInlineMarkdown(match[2])}</h${match[1].length}>`;
          })
          .join("");
      }
      if (trimmed.startsWith("- ")) {
        return `<ul>${trimmed.split("\n").map((line) => `<li>${renderInlineMarkdown(line.replace(/^- /, ""))}</li>`).join("")}</ul>`;
      }
      if (/^\d+\.\s/.test(trimmed)) {
        return `<ol>${trimmed.split("\n").map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\.\s/, ""))}</li>`).join("")}</ol>`;
      }
      return `<p>${trimmed.split("\n").map((line) => renderInlineMarkdown(line)).join("<br />")}</p>`;
    })
    .join("");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text || "")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^\*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}
