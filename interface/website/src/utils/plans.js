export const READ_ONLY_PLAN_TOOLS = [
  "list_directory",
  "locate_files",
  "read_file",
  "inspect_symbols",
  "search_text",
  "peek_lines",
  "track_symbol",
];

export function buildPlanModePrompt(userPrompt) {
  const cleaned = String(userPrompt || "").trim();
  return `PLAN MODE IS ACTIVE.

The user wants a planning graph only. Do not execute file edits. You may inspect the workspace with read-only tools if needed.

Return ONLY a valid JSON object for a multi-layer flowchart. No markdown fences. No prose before or after the JSON.

Required structure:
{
  "tasks": [
    {
      "task_id": "short-kebab-id",
      "description": "Clear actionable step",
      "level": 0
    }
  ],
  "edges": [
    { "from": "task-id", "to": "task-id" }
  ]
}

Rules:
- Use multiple nodes whenever the work can be broken down.
- Use multiple levels when some tasks depend on earlier tasks or branch from them.
- "task_id" must be unique.
- "description" must be concrete and implementation-oriented.
- "level" must be an integer >= 0.
- Include "edges" whenever there is more than one task.
- Return raw JSON only.

User request:
${cleaned}`;
}

export function shouldDisplayPlanResult(result, planModeEnabled) {
  if (planModeEnabled) return true;
  const traces = Array.isArray(result?.stage_traces) ? result.stage_traces : [];
  const checkpointCreation = traces.find((trace) => trace && trace.stage === "checkpoint_creation");
  if (checkpointCreation?.payload && typeof checkpointCreation.payload.should_plan === "boolean") {
    return checkpointCreation.payload.should_plan;
  }
  const taskCount = Array.isArray(result?.blueprint?.tasks) ? result.blueprint.tasks.length : 0;
  return taskCount > 1;
}

export function extractPlanBlueprint(result, planModeEnabled) {
  if (planModeEnabled) {
    const parsed = parsePlanBlueprintString(result?.final_response);
    if (parsed) return parsed;
  }
  return result?.blueprint || null;
}

export function parsePlanBlueprintString(content) {
  const raw = String(content || "").trim();
  if (!raw) return null;
  const candidates = buildJsonCandidates(raw);
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (isPlanBlueprint(parsed)) {
        return parsed;
      }
      if (parsed && typeof parsed === "object" && typeof parsed.content === "string") {
        const nested = parsePlanBlueprintString(parsed.content);
        if (nested) return nested;
      }
    } catch {
      // keep trying
    }
  }
  return null;
}

function buildJsonCandidates(raw) {
  const trimmed = raw.trim();
  const candidates = [trimmed];
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) {
    candidates.push(fenceMatch[1].trim());
  }
  const balanced = extractBalancedJsonObject(trimmed);
  if (balanced) {
    candidates.push(balanced);
  }
  const contentIndex = trimmed.indexOf('{"type"');
  if (contentIndex >= 0) {
    candidates.push(trimmed.slice(contentIndex));
  }
  return [...new Set(candidates.filter(Boolean))];
}

function extractBalancedJsonObject(text) {
  const start = text.indexOf("{");
  if (start < 0) return "";
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === "{") depth += 1;
    if (char === "}") depth -= 1;
    if (depth === 0) {
      return text.slice(start, index + 1);
    }
  }
  return "";
}

function isPlanBlueprint(value) {
  if (!value || typeof value !== "object") return false;
  return Array.isArray(value.tasks) || Array.isArray(value.nodes) || Array.isArray(value.steps) || Array.isArray(value.items) || Array.isArray(value.plan);
}
