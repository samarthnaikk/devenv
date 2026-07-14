import React from "https://esm.sh/react@18.2.0";

export function ContextBuilderPanel({
  sources,
  selectedProvider,
  onProviderChange,
  sessions,
  activeSessionIds,
  onSelectSession,
  sessionDetail,
  builderTask,
  onBuilderTaskChange,
  includeWorkspaceScan,
  onIncludeWorkspaceScanChange,
  includePriorContext,
  onIncludePriorContextChange,
  promptResult,
  onGeneratePrompt,
  onRefreshSessions,
  onCopyPrompt,
  isLoading,
  isPreparing,
  statusMessage,
}) {
  const currentSource = sources.find((source) => source.provider === selectedProvider) || null;
  const matchedSessions = activeSessionIds.length
    ? sessions.filter((session) => activeSessionIds.includes(session.session_id))
    : sessions.slice(0, 6);

  return React.createElement(
    "section",
    { className: "content-panel context-builder-panel" },
    React.createElement(
      "div",
      { className: "context-builder-header" },
      React.createElement(
        "div",
        { className: "context-builder-copy" },
        React.createElement("div", { className: "panel-label" }, "Context Builder"),
        React.createElement("h2", { className: "terminal-title" }, "Prepare Codex Prompt"),
        React.createElement(
          "p",
          { className: "context-builder-note" },
          "Devenv automatically matches prior sessions and builds the prompt for copy-paste."
        )
      ),
      React.createElement(
        "div",
        { className: "panel-header-actions" },
        React.createElement(
          "button",
          {
            className: "context-action-button",
            type: "button",
            onClick: onRefreshSessions,
            disabled: isLoading || !selectedProvider,
          },
          isLoading ? "Refreshing..." : "Refresh"
        ),
        React.createElement(
          "button",
          {
            className: "context-action-button primary",
            type: "button",
            onClick: onGeneratePrompt,
            disabled: isPreparing,
          },
          isPreparing ? "Preparing..." : "Generate Prompt"
        )
      )
    ),
    React.createElement(
      "div",
      { className: "context-builder-toolbar" },
      React.createElement(
        "label",
        { className: "context-provider-field" },
        React.createElement("span", { className: "status-label" }, "Provider"),
        React.createElement(
          "select",
          {
            className: "context-provider-select",
            value: selectedProvider,
            onChange: (event) => onProviderChange(event.target.value),
          },
          sources.map((source) =>
            React.createElement("option", { key: source.provider, value: source.provider }, source.provider)
          )
        )
      ),
      currentSource
        ? React.createElement(
            "div",
            { className: "context-source-chip" },
            `${currentSource.summary || "Provider ready"}`
          )
        : null,
      React.createElement(
        "label",
        { className: "context-checkbox" },
        React.createElement("input", {
          type: "checkbox",
          checked: includeWorkspaceScan,
          onChange: (event) => onIncludeWorkspaceScanChange(event.target.checked),
        }),
        React.createElement("span", null, "Include repo scan")
      ),
      React.createElement(
        "label",
        { className: "context-checkbox" },
        React.createElement("input", {
          type: "checkbox",
          checked: includePriorContext,
          onChange: (event) => onIncludePriorContextChange(event.target.checked),
        }),
        React.createElement("span", null, "Include prior context")
      )
    ),
    React.createElement(
      "div",
      { className: "context-builder-body" },
      React.createElement(
        "section",
        { className: "context-section-card" },
        React.createElement(
          "div",
          { className: "context-section-heading" },
          activeSessionIds.length ? `Auto-matched Sessions (${activeSessionIds.length})` : "Recent Sessions"
        ),
        React.createElement(
          "div",
          { className: "context-session-list" },
          matchedSessions.length
            ? matchedSessions.map((session) =>
                React.createElement(
                  "button",
                  {
                    key: session.session_id,
                    type: "button",
                    className: `context-session-card${activeSessionIds.includes(session.session_id) ? " selected" : ""}`,
                    onClick: () => onSelectSession(session.session_id),
                  },
                  React.createElement(
                    "div",
                    { className: "context-session-copy" },
                    React.createElement("strong", null, session.title || "Untitled session"),
                    React.createElement(
                      "span",
                      null,
                      session.updated_at || session.workspace_path || "Unknown update time"
                    ),
                    React.createElement(
                      "p",
                      null,
                      session.preview || session.workspace_path || "No preview available yet."
                    )
                  )
                )
              )
            : React.createElement("div", { className: "tree-empty" }, "No sessions available for this provider.")
        )
      ),
      React.createElement(
        "section",
        { className: "context-section-card" },
        React.createElement("div", { className: "context-section-heading" }, "Session Detail"),
        sessionDetail
          ? React.createElement(
              "div",
              { className: "context-session-detail" },
              React.createElement("strong", null, sessionDetail.summary?.title || "Untitled session"),
              React.createElement(
                "span",
                null,
                sessionDetail.summary?.workspace_path || sessionDetail.summary?.updated_at || "No workspace hint"
              ),
              React.createElement(
                "div",
                { className: "context-message-list" },
                (sessionDetail.messages || []).slice(0, 8).map((message, index) =>
                  React.createElement(
                    "article",
                    { key: `${message.role}-${index}`, className: `context-message ${message.role}` },
                    React.createElement("div", { className: "bubble-role" }, message.role),
                    React.createElement("p", null, message.content)
                  )
                )
              )
            )
          : React.createElement(
              "div",
              { className: "tree-empty" },
              "Generate a prompt and Devenv will show the best-matching session history here."
            )
      ),
      React.createElement(
        "section",
        { className: "context-section-card context-prompt-card" },
        React.createElement("div", { className: "context-section-heading" }, "Prompt Preview"),
        React.createElement("textarea", {
          className: "context-task-input",
          rows: 4,
          value: builderTask,
          onChange: (event) => onBuilderTaskChange(event.target.value),
          placeholder: "Describe what you want Codex to do. Leave this blank to use the current chat prompt.",
        }),
        React.createElement("textarea", {
          className: "context-prompt-output",
          rows: 18,
          readOnly: true,
          value: promptResult?.prompt || "",
          placeholder: "The generated Codex prompt will appear here.",
        }),
        React.createElement(
          "div",
          { className: "context-prompt-actions" },
          React.createElement(
            "button",
            {
              className: "context-action-button",
              type: "button",
              onClick: onCopyPrompt,
              disabled: !(promptResult?.prompt || "").trim(),
            },
            "Copy Prompt"
          ),
          React.createElement("span", { className: "context-status" }, statusMessage || "")
        )
      )
    )
  );
}
