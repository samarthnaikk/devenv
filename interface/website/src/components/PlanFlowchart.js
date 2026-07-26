import React from "react";
import ReactFlow, {
  Handle, Position, Background, Controls,
  MarkerType, useNodesState, useEdgesState,
} from "reactflow";
import { validatePlanBlueprint, normalizeBlueprint } from "../utils/validation.js";
import { escapeHtml } from "../utils/format.js";
import { BeamFrame, MetalSurface, MotionDeck, MotionNumber, MotionReveal, MotionShimmerText } from "./MotionPrimitives.js";

function BlueprintNode({ data }) {
  const [showModal, setShowModal] = React.useState(false);
  const statusColor = data.status === "done" ? "var(--primary)" : data.status === "active" ? "#facc15" : "var(--outline)";
  const statusIcon = data.status === "done" ? "check_circle" : data.status === "active" ? "play_circle" : "circle";

  return React.createElement(
    React.Fragment,
    null,
      React.createElement(
        "div",
        {
          className: `plan-node-shell${data.status === "active" ? " is-active" : ""}${data.status === "done" ? " is-done" : ""}`,
          style: { width: `${data.width || 320}px`, maxWidth: `${data.width || 320}px` },
        },
        React.createElement("span", { className: "plan-node-orbit", "aria-hidden": "true" }),
        React.createElement(Handle, { type: "target", position: Position.Top, style: { background: "var(--outline)", width: 8, height: 8 } }),
        React.createElement(
          "div",
          { className: "plan-node-head" },
        React.createElement(
          "span",
          { style: { fontSize: "16px", color: statusColor } },
          React.createElement("span", { className: "material-symbols-outlined", style: { fontSize: "16px", fontVariationSettings: "'FILL' 1" } }, statusIcon)
        ),
          React.createElement(
            "span",
            { className: "plan-node-level" },
            `L${data.level}`
          ),
          React.createElement("span", { className: "plan-node-status" }, formatNodeStatus(data.status))
        ),
        React.createElement(
          "div",
          { className: "plan-node-label" },
          escapeHtml(data.label || "")
        ),
        data.desc
          ? React.createElement(
              "div",
              { className: "plan-node-desc-preview" },
              escapeHtml(truncateNodeCopy(data.desc))
            )
          : null,
        React.createElement(
          "button",
          {
            type: "button",
          className: "plan-node-info",
          onClick: (e) => { e.stopPropagation(); setShowModal(true); },
          title: "Details",
        },
        React.createElement("span", { className: "material-symbols-outlined", style: { fontSize: "14px" } }, "info")
      ),
      React.createElement(Handle, { type: "source", position: Position.Bottom, style: { background: "var(--outline)", width: 8, height: 8 } })
    ),
    showModal ? React.createElement(
      "div",
      { className: "plan-node-modal-backdrop", onClick: () => setShowModal(false) },
      React.createElement(
        "div",
        { className: "plan-node-modal", onClick: (e) => e.stopPropagation() },
        React.createElement("h3", { className: "plan-node-modal-title" }, escapeHtml(data.label || "")),
        React.createElement("p", { className: "plan-node-modal-copy" }, escapeHtml(data.desc || "No description")),
        React.createElement(
          "button",
          {
            type: "button",
            className: "plan-node-modal-close",
            onClick: () => setShowModal(false),
          },
          "Close"
        )
      )
    ) : null
  );
}

const nodeTypes = { blueprint: BlueprintNode };

function layoutNodes(nodes, edges) {
  const byLevel = {};
  const incoming = new Map();
  (Array.isArray(edges) ? edges : []).forEach((edge) => {
    const target = edge.to || edge.target;
    const source = edge.from || edge.source;
    if (!target || !source) return;
    const bucket = incoming.get(target) || [];
    bucket.push(source);
    incoming.set(target, bucket);
  });
  nodes.forEach((n) => {
    const level = n.level != null ? n.level : 0;
    (byLevel[level] = byLevel[level] || []).push(n);
  });
  const H_SPACING = 48;
  const V_SPACING = 180;
  const MAX_ROW_WIDTH = 980;
  const result = [];
  const sortedLevels = Object.keys(byLevel).sort((a, b) => Number(a) - Number(b));
  const previousPositions = new Map();
  sortedLevels.forEach((level) => {
    const group = [...byLevel[level]].sort((left, right) => {
      const leftParents = (incoming.get(left.id) || []).join(",");
      const rightParents = (incoming.get(right.id) || []).join(",");
      return leftParents.localeCompare(rightParents) || left.id.localeCompare(right.id);
    });
    const levelNumber = Number(level);
    const measured = group.map((node) => ({
      node,
      width: estimateNodeWidth(node.label || ""),
      parentCenter: getParentCenter(incoming.get(node.id) || [], previousPositions),
    }));
    let cursor = 0;
    let row = 0;
    let rowWidth = 0;
    measured.forEach((entry, index) => {
      const projectedWidth = rowWidth === 0 ? entry.width : rowWidth + H_SPACING + entry.width;
      if (projectedWidth > MAX_ROW_WIDTH && rowWidth > 0) {
        row += 1;
        rowWidth = 0;
        cursor = 0;
      }
      const centeredX = entry.parentCenter != null ? entry.parentCenter - entry.width / 2 : cursor;
      const desiredX = Math.max(centeredX, cursor);
      entry.x = index === 0 ? centeredX : desiredX;
      entry.yOffset = row * 118;
      cursor = entry.x + entry.width + H_SPACING;
      rowWidth = rowWidth === 0 ? entry.width : rowWidth + H_SPACING + entry.width;
    });
    const minX = Math.min(...measured.map((entry) => entry.x));
    const maxX = Math.max(...measured.map((entry) => entry.x + entry.width));
    const offset = (minX + maxX) / 2;
    measured.forEach((entry) => {
      const x = entry.x - offset;
      previousPositions.set(entry.node.id, { x, width: entry.width });
      result.push({
        id: entry.node.id,
        type: "blueprint",
        position: { x, y: levelNumber * V_SPACING + 20 + (entry.yOffset || 0) },
        data: { label: entry.node.label, level: entry.node.level, desc: entry.node.desc || "", status: entry.node.status || "pending", width: entry.width },
      });
    });
  });
  return result;
}

function convertEdges(edges) {
  return edges.map((e) => ({
    id: `${e.from}->${e.to}`,
    source: e.from,
    target: e.to,
    type: "smoothstep",
    markerEnd: { type: MarkerType.ArrowClosed, color: "#4fdbc8" },
    style: { stroke: "#3c4947", strokeWidth: 2 },
  }));
}

function estimateNodeWidth(label) {
  const normalized = String(label || "").trim();
  return Math.min(420, Math.max(250, 180 + normalized.length * 5.5));
}

function getParentCenter(parentIds, positions) {
  const centers = parentIds
    .map((parentId) => positions.get(parentId))
    .filter(Boolean)
    .map((entry) => entry.x + entry.width / 2);
  if (!centers.length) return null;
  return centers.reduce((sum, value) => sum + value, 0) / centers.length;
}

export function PlanFlowchart({ blueprint, mode = "auto" }) {
  const normalized = React.useMemo(() => normalizeBlueprint(blueprint), [blueprint]);
  const validation = React.useMemo(() => validatePlanBlueprint({
    nodes: normalized.nodes,
    edges: normalized.edges,
  }), [normalized]);

  if (!validation.valid) {
    return React.createElement(
      MotionReveal,
      { className: "flex flex-col gap-2 w-full max-w-[88rem]" },
      React.createElement(
        "div",
        { className: "flex items-center gap-2 mb-1" },
        React.createElement(
          "div",
          { className: "w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center" },
          React.createElement("span", { className: "material-symbols-outlined text-[14px] text-primary" }, "account_tree")
        ),
        React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "Execution Plan"),
        React.createElement("span", { className: "px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-error" }, "Invalid")
      ),
      React.createElement("div", { className: "ml-3 pl-3 border-l-2 border-outline-variant font-body-md text-body-md text-error" }, validation.error || "Invalid plan data")
    );
  }

  const allDone = normalized.nodes.every((n) => n.status === "done");
  const flowNodes = React.useMemo(() => layoutNodes(normalized.nodes, normalized.edges), [normalized.nodes, normalized.edges]);
  const flowEdges = React.useMemo(() => convertEdges(normalized.edges), [normalized.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  React.useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  return React.createElement(
    BeamFrame,
    { active: normalized.nodes.some((node) => node.status === "active"), tone: "ocean", className: "flex flex-col gap-2 w-full max-w-[88rem]" },
    React.createElement(
      "div",
      { className: "plan-header flex items-center gap-2 mb-1 flex-wrap" },
      React.createElement(
        "div",
        { className: "w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center" },
        React.createElement("span", { className: "material-symbols-outlined text-[14px] text-primary" }, "account_tree")
      ),
      React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "Execution Plan"),
      React.createElement(
        MotionShimmerText,
        { className: "plan-header-copy", active: normalized.nodes.some((node) => node.status === "active") },
        normalized.nodes.some((node) => node.status === "active")
          ? "Steps are actively progressing through the graph"
          : "Blueprint is ready to inspect and execute"
      ),
      React.createElement(
        "span",
        { className: `plan-header-pill px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] ${blueprint.verification_passed ? "text-primary" : allDone ? "text-primary" : "text-on-surface-variant"}` },
        blueprint.verification_passed ? "Verified" : allDone ? "Done" : "In progress"
      ),
      React.createElement(
        "span",
        { className: "plan-header-pill px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" },
        mode === "forced" ? "Plan mode" : "Auto-planned"
      ),
      React.createElement(
        "span",
        { className: "plan-header-pill px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" },
        `${normalized.nodes.length} steps`
      ),
      React.createElement(
        "span",
        { className: "plan-header-pill px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" },
        `${normalized.edges.length} links`
      )
    ),
      React.createElement(
        MotionDeck,
        { className: "plan-summary-grid mb-2" },
        planStat("Layers", String(new Set(normalized.nodes.map((node) => node.level)).size), "Execution depth across the graph"),
        planStat("Next", nextActionLabel(normalized.nodes), "The step that should move first"),
        planStat("State", blueprint.verification_passed ? "Verified" : allDone ? "Done" : normalized.nodes.some((node) => node.status === "active") ? "Running" : "Ready", "Current graph status")
      ),
      React.createElement(
        "div",
        { className: "plan-flow-canvas", style: { height: "380px" } },
        React.createElement(
          ReactFlow,
        {
          nodes,
          edges,
          onNodesChange,
          onEdgesChange,
          nodeTypes,
          fitView: true,
          fitViewOptions: { padding: 0.28, minZoom: 0.4 },
          panOnDrag: true,
          panOnScroll: true,
          zoomOnScroll: true,
          zoomOnPinch: true,
          zoomOnDoubleClick: false,
          nodesDraggable: false,
          nodesConnectable: false,
          elementsSelectable: true,
          minZoom: 0.35,
          maxZoom: 2.5,
          proOptions: { hideAttribution: true },
        },
        React.createElement(Controls, { showInteractive: false, position: "bottom-right" }),
        React.createElement(Background, { color: "rgba(108, 130, 149, 0.2)", gap: 22, size: 1.2 })
      )
    )
  );
}

function planStat(label, value, detail) {
  const numeric = /^\d+$/.test(String(value || "").trim());
  return React.createElement(
    MetalSurface,
    { className: "plan-summary-card" },
    React.createElement("span", { className: "plan-summary-label" }, label),
    React.createElement(
      "strong",
      { className: "plan-summary-value" },
      numeric ? React.createElement(MotionNumber, { value }) : React.createElement(MotionShimmerText, { active: false }, value)
    ),
    React.createElement("span", { className: "plan-summary-detail" }, detail)
  );
}

function nextActionLabel(nodes) {
  const active = nodes.find((node) => node.status === "active");
  if (active?.label) return truncatePlanStat(active.label);
  const pending = nodes.find((node) => node.status !== "done");
  if (pending?.label) return truncatePlanStat(pending.label);
  return "Complete";
}

function truncatePlanStat(value) {
  const text = String(value || "").trim();
  return text.length > 32 ? `${text.slice(0, 29)}...` : text;
}

function formatNodeStatus(status) {
  if (status === "done") return "done";
  if (status === "active") return "live";
  return "queued";
}

function truncateNodeCopy(value) {
  const text = String(value || "").trim();
  return text.length > 84 ? `${text.slice(0, 81)}...` : text;
}
