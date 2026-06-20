import React from "https://esm.sh/react@18";

export function PlanRail({ blueprint, runtimeState }) {
  const tasks = blueprint?.tasks || [];
  if (!tasks.length) {
    return null;
  }

  return React.createElement(
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
        const statusClass = isDone ? "done" : isActive ? "active" : "pending";
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
            React.createElement("strong", null, `Step ${task.task_id}`),
            React.createElement("span", null, task.description)
          )
        );
      })
    )
  );
}
