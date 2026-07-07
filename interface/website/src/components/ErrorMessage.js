import React from "https://esm.sh/react@18";
import { escapeHtml, escapeAttribute } from "../utils/format.js";

export function ErrorMessage({ message }) {
  return React.createElement(
    "div",
    { className: "flex flex-col gap-2 max-w-3xl" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2" },
      React.createElement(
        "div",
        { className: "w-6 h-6 rounded-full bg-error flex items-center justify-center" },
        React.createElement("span", { className: "material-symbols-outlined text-[14px] text-on-error" }, "error")
      ),
      React.createElement("span", { className: "font-label-caps text-label-caps text-error" }, "Error"),
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
        className: "font-body-lg text-body-lg text-error ml-8",
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
