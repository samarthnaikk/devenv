export function validatePlanBlueprint(data) {
  if (!data || typeof data !== "object") {
    return { valid: false, error: "Blueprint must be an object with a 'tasks' or 'nodes' array" };
  }

  if (Array.isArray(data.tasks)) {
    for (let i = 0; i < data.tasks.length; i++) {
      const t = data.tasks[i];
      if (t && typeof t === "object" && !t.task_id && !t.id) {
        t.task_id = `task-${i}`;
      }
    }
    if (data.tasks.length > 1 && (!Array.isArray(data.edges) || data.edges.length === 0)) {
      data.edges = [];
      for (let i = 0; i < data.tasks.length - 1; i++) {
        data.edges.push({ from: data.tasks[i].task_id || `task-${i}`, to: data.tasks[i + 1].task_id || `task-${i + 1}` });
      }
    }
  }

  const hasTasks = Array.isArray(data.tasks);
  const hasNodes = Array.isArray(data.nodes);

  if (!hasTasks && !hasNodes) {
    return { valid: false, error: "Blueprint must contain a 'tasks' or 'nodes' array" };
  }

  if (hasNodes) {
    return validateFlowchartFormat(data);
  }

  return validateTasksFormat(data);
}

function validateTasksFormat(data) {
  const tasks = data.tasks;
  const edges = data.edges || [];

  if (tasks.length === 0) {
    return { valid: false, error: "Blueprint 'tasks' array is empty — at least one task is required" };
  }

  const taskIds = new Set();
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];
    if (!task || typeof task !== "object") {
      return { valid: false, error: `Task at index ${i} must be an object` };
    }
    const id = task.task_id || task.id;
    if (typeof id !== "string" || !id.trim()) {
      return { valid: false, error: `Task at index ${i} must have a 'task_id' (or 'id') string field` };
    }
    if (taskIds.has(id)) {
      return { valid: false, error: `Duplicate task id: ${id}` };
    }
    taskIds.add(id);
    if (typeof task.description !== "string" || !task.description.trim()) {
      return { valid: false, error: `Task ${id} must have a non-empty 'description'` };
    }
    if (typeof task.level !== "number" || task.level < 0 || !Number.isInteger(task.level)) {
      return { valid: false, error: `Task ${id} must have an integer 'level' >= 0 (e.g. 0, 1, 2) for graph layout` };
    }
  }

  if (tasks.length > 1 && edges.length === 0) {
    return { valid: false, error: "Blueprint must include an 'edges' array connecting tasks when there are multiple tasks (e.g. [{from: 'a', to: 'b'}])" };
  }

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    if (!edge || typeof edge !== "object") {
      return { valid: false, error: `Edge at index ${i} must be an object with 'from' and 'to'` };
    }
    const from = edge.from || edge.source;
    const to = edge.to || edge.target;
    if (typeof from !== "string" || !taskIds.has(from)) {
      return { valid: false, error: `Edge ${i} references unknown source task: '${from}' — must match a task's 'task_id'` };
    }
    if (typeof to !== "string" || !taskIds.has(to)) {
      return { valid: false, error: `Edge ${i} references unknown target task: '${to}' — must match a task's 'task_id'` };
    }
  }

  return { valid: true };
}

function validateFlowchartFormat(data) {
  const nodes = data.nodes;
  const edges = data.edges || [];

  if (nodes.length === 0) {
    return { valid: false, error: "Blueprint 'nodes' array is empty" };
  }

  const nodeIds = new Set();
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (!node || typeof node !== "object") {
      return { valid: false, error: `Node at index ${i} must be an object` };
    }
    if (typeof node.id !== "string" || !node.id.trim()) {
      return { valid: false, error: `Node at index ${i} must have a valid string 'id'` };
    }
    if (nodeIds.has(node.id)) {
      return { valid: false, error: `Duplicate node id: ${node.id}` };
    }
    nodeIds.add(node.id);
    if (typeof node.label !== "string" || !node.label.trim()) {
      return { valid: false, error: `Node ${node.id} must have a non-empty 'label'` };
    }
    if (typeof node.level !== "number" || node.level < 0 || !Number.isInteger(node.level)) {
      return { valid: false, error: `Node ${node.id} must have an integer 'level' >= 0 for graph layout` };
    }
  }

  if (nodes.length > 1 && edges.length === 0) {
    return { valid: false, error: "Blueprint must include an 'edges' array connecting nodes" };
  }

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    if (!edge || typeof edge !== "object") {
      return { valid: false, error: `Edge at index ${i} must be an object with 'from' and 'to'` };
    }
    const from = edge.from || edge.source;
    const to = edge.to || edge.target;
    if (typeof from !== "string" || !nodeIds.has(from)) {
      return { valid: false, error: `Edge ${i} references unknown source: ${from}` };
    }
    if (typeof to !== "string" || !nodeIds.has(to)) {
      return { valid: false, error: `Edge ${i} references unknown target: ${to}` };
    }
  }

  return { valid: true };
}

export function normalizeBlueprint(blueprint) {
  if (!blueprint || typeof blueprint !== "object") {
    return { nodes: [], edges: [] };
  }

  if (Array.isArray(blueprint.nodes)) {
    return { nodes: blueprint.nodes, edges: blueprint.edges || [] };
  }

  if (Array.isArray(blueprint.tasks)) {
    return normalizeTasksToFlowchart(blueprint);
  }

  return { nodes: [], edges: [] };
}

function normalizeTasksToFlowchart(blueprint) {
  const tasks = blueprint.tasks || [];
  const rawEdges = blueprint.edges || [];
  const activePointer = blueprint.active_task_pointer ?? 0;

  const nodes = tasks.map((t, i) => ({
    id: t.task_id || t.id || `task-${i}`,
    label: t.description || t.label || `Task ${i + 1}`,
    level: t.level,
    desc: t.desc || t.description || "",
    status: t.is_completed ? "done" : i === activePointer ? "active" : "pending",
  }));

  const seenIds = new Set(nodes.map((n) => n.id));

  const edges = rawEdges
    .map((e) => {
      const from = e.from || e.source;
      const to = e.to || e.target;
      return seenIds.has(from) && seenIds.has(to) ? { from, to } : null;
    })
    .filter(Boolean);

  return { nodes, edges };
}
