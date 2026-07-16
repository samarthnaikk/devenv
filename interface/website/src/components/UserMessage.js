import React from "https://esm.sh/react@18.2.0";
import { escapeHtml } from "../utils/format.js";

export function UserMessage({ message, onCopy, onReply }) {
  return React.createElement(
    "div",
    { className: "flex flex-col gap-2 max-w-3xl" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2" },
      React.createElement(
        "div",
        { className: "w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center" },
        React.createElement(
          "span",
          { className: "material-symbols-outlined text-[14px] text-on-surface" },
          "person"
        )
      ),
      React.createElement("span", { className: "font-label-caps text-label-caps text-on-surface" }, "You"),
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
          { className: "ml-8 rounded-lg border border-outline-variant/70 bg-surface-container px-3 py-2 text-[12px] text-on-surface-variant" },
          React.createElement("div", { className: "mb-1 font-label-caps text-label-caps text-primary" }, `Replying to ${message.replyTo.author}`),
          React.createElement("div", null, message.replyTo.excerpt)
        )
      : null,
    React.createElement(
      "div",
      {
        className: "font-body-lg text-body-lg text-on-surface ml-8",
        dangerouslySetInnerHTML: { __html: renderRichText(message.content) },
      }
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
      if (trimmed.startsWith("> ")) {
        return `<blockquote>${trimmed.split("\n").map((line) => renderInlineMarkdown(line.replace(/^>\s?/, ""))).join("<br />")}</blockquote>`;
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
