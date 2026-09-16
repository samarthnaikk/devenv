import React from "react";
import { renderMarkdown } from "../lib/markdown.js";
import { BeamFrame, MetalSurface, MotionBadge, MotionReveal, MotionShimmerText, MotionStage } from "./MotionPrimitives.js";

export function AssistantMessage({ message, onCopy, onReply }) {
  const diagnostics = message.diagnostics || {};
  const statusChips = [
    diagnostics.sourceLabel,
    diagnostics.routeLabel,
    diagnostics.backendLabel,
    diagnostics.toolLabel,
    diagnostics.retrievalLabel,
  ].filter(Boolean);
  const evidenceItems = Array.isArray(diagnostics.evidenceItems) ? diagnostics.evidenceItems : [];

  return React.createElement(
    MotionStage,
    { axis: "x", className: "message-stack flex flex-col gap-2 max-w-3xl message-stack-assistant" },
    React.createElement(
      BeamFrame,
      { active: false, tone: "ocean", className: "message-card assistant-message-card rounded-[24px] p-4" },
      React.createElement("span", { className: "message-card-orbit", "aria-hidden": "true" }),
      React.createElement("span", { className: "message-card-orbit message-card-orbit-secondary", "aria-hidden": "true" }),
      React.createElement("span", { className: "message-card-ribbon message-card-ribbon-assistant", "aria-hidden": "true" }),
      React.createElement("span", { className: "message-card-grid", "aria-hidden": "true" }),
      React.createElement(
        "div",
        { className: "message-head flex items-center gap-2" },
        React.createElement(
          "div",
          { className: "message-avatar assistant-avatar w-7 h-7 rounded-full flex items-center justify-center" },
          React.createElement("span", { className: "material-symbols-outlined text-on-primary text-[14px]" }, "auto_awesome")
        ),
        React.createElement(
          "div",
          { className: "message-title-group" },
          React.createElement(MotionShimmerText, { className: "font-label-caps text-label-caps text-primary" }, "Devenv"),
          React.createElement("span", { className: "message-kicker" }, diagnostics.kicker || "assistant output")
        ),
        React.createElement(MotionBadge, { className: "message-type-pill message-type-pill-assistant", active: true }, diagnostics.badgeLabel || "Answer"),
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
        { className: "message-status-row" },
        ...statusChips.map((chip) => React.createElement("span", { key: chip, className: "message-status-chip" }, chip))
      ),
      React.createElement(
        MetalSurface,
        { className: "message-runway" },
        React.createElement("span", { className: "message-runway-beam", "aria-hidden": "true" }),
        React.createElement(
          "div",
          { className: "message-runway-head" },
          React.createElement("span", { className: "message-runway-kicker" }, diagnostics.sourceLabel || "Answer"),
          React.createElement("span", { className: "message-runway-divider", "aria-hidden": "true" }),
          React.createElement("strong", { className: "message-runway-value" }, diagnostics.routeLabel || "Direct route")
        ),
        React.createElement(
          "div",
          { className: "message-runway-copy" },
          evidenceItems.length
            ? `${evidenceItems.length} live source${evidenceItems.length === 1 ? "" : "s"} surfaced directly in the answer card.`
            : diagnostics.toolLabel
              ? `${diagnostics.toolLabel} shaped this response path.`
              : "Direct answer path with runtime provenance kept visible."
        )
      ),
      diagnostics.detail
        ? React.createElement("div", { className: "message-status-copy" }, diagnostics.detail)
        : null,
      evidenceItems.length
        ? React.createElement(
            "div",
            { className: "message-evidence-grid" },
            evidenceItems.map((item) =>
              React.createElement(
                "a",
                {
                  key: `${item.label}-${item.title}-${item.url}`,
                  className: `message-evidence-card is-${item.kind || "web"}`,
                  href: item.url,
                  target: "_blank",
                  rel: "noreferrer",
                },
                React.createElement("span", { className: "message-evidence-label" }, item.label || "Source"),
                React.createElement("strong", { className: "message-evidence-title" }, item.title),
                React.createElement("span", { className: "message-evidence-url" }, item.url.replace(/^https?:\/\//, ""))
              )
            )
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
