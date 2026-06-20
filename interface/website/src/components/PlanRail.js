import React from "https://esm.sh/react@18";
import { renderMarkdown } from "../lib/markdown.js";

export function PlanRail({ blueprint, runtimeState, stageTraces = [], verificationResults = [] }) {
  const tasks = blueprint?.tasks || [];
  const [selectedTask, setSelectedTask] = React.useState(null);

  if (!tasks.length) {
    return null;
  }

  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "section",
      { className: "plan-rail" },
      React.createElement(
        "div",
        { className: "plan-rail-header" },
        React.createElement("span", { className: "panel-label" }, "Execution Plan"),
        React.createElement("span", { className: "plan-state-chip" }, runtimeState || "PLANNING")
      ),
      React.createElement(
        "div",
        { className: "plan-rail-track", role: "list" },
        tasks.map((task, index) => {
          const isActive = !task.is_completed && index === (blueprint?.active_task_pointer ?? 0);
          const isDone = Boolean(task.is_completed);
          const isRepair = Boolean(task.repair_origin_checkpoint_id);
          const hasChildren = Array.isArray(task.child_checkpoint_ids) && task.child_checkpoint_ids.length > 0;
          const statusClass = isDone ? "done" : isActive ? "active" : "pending";
          const verification = verificationResults.filter((entry) => entry.checkpoint_id === task.task_id);
          return React.createElement(
            "div",
            {
              key: `${task.task_id}-${task.description}`,
              className: `plan-node ${statusClass}`,
              role: "listitem",
              title: task.description,
            },
            React.createElement("div", { className: "plan-node-dot" }, isDone ? "✓" : task.task_id),
            React.createElement(
              "div",
              { className: "plan-node-copy" },
              React.createElement("strong", null, isRepair ? `Repair ${task.task_id}` : `Step ${task.task_id}`),
              React.createElement("span", null, summarizeTask(task.description)),
              hasChildren ? React.createElement("span", { className: "panel-label" }, `Split into ${task.child_checkpoint_ids.length}`) : null,
              verification.length
                ? React.createElement("span", { className: "panel-label" }, verification.every((item) => item.success) ? "Verified" : "Needs repair")
                : null
            ),
            React.createElement(
              "button",
              {
                type: "button",
                className: "plan-node-info",
                onClick: () => setSelectedTask(task),
                "aria-label": `Inspect step ${task.task_id}`,
                title: `Inspect step ${task.task_id}`,
              },
              "i"
            )
          );
        })
      )
    ),
    selectedTask
      ? React.createElement(
          "div",
          {
            className: "plan-modal-overlay",
            onClick: () => setSelectedTask(null),
          },
          React.createElement(
            "section",
            {
              className: "plan-modal",
              onClick: (event) => event.stopPropagation(),
            },
            React.createElement(
              "div",
              { className: "plan-modal-header" },
              React.createElement("strong", null, `Step ${selectedTask.task_id}`),
              React.createElement(
                "button",
                {
                  type: "button",
                  className: "plan-modal-close",
                  onClick: () => setSelectedTask(null),
                  "aria-label": "Close step details",
                },
                "×"
              )
            ),
            React.createElement("div", {
              className: "plan-modal-body markdown-body",
              dangerouslySetInnerHTML: { __html: renderMarkdown(selectedTask.description) },
            }),
            React.createElement(
              "div",
              { className: "plan-modal-trace" },
              React.createElement("span", { className: "panel-label" }, "Checkpoint Metadata"),
              React.createElement("div", { className: "markdown-body" }, [
                `Objective: ${selectedTask.objective || selectedTask.description}`,
                `Target: ${selectedTask.target_path_hint || "chat"}`,
                `Artifact: ${selectedTask.expected_artifact || "chat"}`,
                `Verification: ${selectedTask.verification_mode || "chat"}`,
                selectedTask.repair_origin_checkpoint_id ? `Repairing checkpoint: ${selectedTask.repair_origin_checkpoint_id}` : "",
              ].filter(Boolean).join("\n"))
            ),
            selectedTask.execution_trace_log
              ? React.createElement(
                  "div",
                  { className: "plan-modal-trace" },
                  React.createElement("span", { className: "panel-label" }, "Execution Note"),
                  React.createElement("div", {
                    className: "markdown-body",
                    dangerouslySetInnerHTML: { __html: renderMarkdown(selectedTask.execution_trace_log) },
                  })
                )
              : null,
            React.createElement(
              "div",
              { className: "plan-modal-trace" },
              React.createElement("span", { className: "panel-label" }, "Stage Activity"),
              React.createElement(
                "div",
                { className: "markdown-body" },
                stageTraces
                  .filter((trace) => trace.checkpoint_id === selectedTask.task_id)
                  .map((trace) => `${trace.stage}: ${trace.summary}`)
                  .join("\n") || "No stage activity recorded."
              )
            )
          )
        )
      : null
  );
}

function summarizeTask(taskDescription) {
  const compact = String(taskDescription || "").replace(/\s+/g, " ").trim();
  return compact.length > 78 ? `${compact.slice(0, 75)}...` : compact;
}
