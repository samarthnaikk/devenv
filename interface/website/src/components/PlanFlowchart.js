import React from "https://esm.sh/react@18.2.0";
import ReactFlow, {
  Handle, Position, Background, Controls,
  MarkerType, useNodesState, useEdgesState,
} from "https://esm.sh/reactflow@11?deps=react@18.2.0,react-dom@18.2.0";
import { validatePlanBlueprint, normalizeBlueprint } from "../utils/validation.js";
import { escapeHtml } from "../utils/format.js";

function BlueprintNode({ data }) {
  const [showModal, setShowModal] = React.useState(false);
  const statusColor = data.status === "done" ? "#4fdbc8" : data.status === "active" ? "#facc15" : "#3c4947";
  const statusIcon = data.status === "done" ? "check_circle" : data.status === "active" ? "play_circle" : "circle";

  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "div",
      {
        style: {
          background: "#1e2023",
          border: `1px solid ${data.status === "active" ? "#4fdbc8" : "#3c4947"}`,
          borderRadius: "8px",
          padding: "12px 16px",
          minWidth: "160px",
          position: "relative",
          color: "#e2e2e6",
          fontFamily: "Inter, sans-serif",
        },
      },
      React.createElement(Handle, { type: "target", position: Position.Top, style: { background: "#3c4947", width: 8, height: 8 } }),
      React.createElement(
        "div",
        { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" } },
        React.createElement(
          "span",
          { style: { fontSize: "16px", color: statusColor } },
          React.createElement("span", { className: "material-symbols-outlined", style: { fontSize: "16px", fontVariationSettings: "'FILL' 1" } }, statusIcon)
        ),
        React.createElement(
          "span",
          { style: { fontSize: "10px", padding: "2px 6px", borderRadius: "4px", background: "#282a2d", color: "#859490", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" } },
          `L${data.level}`
        )
      ),
      React.createElement(
        "div",
        { style: { fontSize: "13px", fontWeight: 500, lineHeight: 1.4, marginBottom: "4px" } },
        escapeHtml(data.label || "")
      ),
      React.createElement(
        "button",
        {
          type: "button",
          style: {
            position: "absolute",
            top: "8px",
            right: "8px",
            background: "none",
            border: "none",
            color: "#859490",
            cursor: "pointer",
            padding: "2px",
            fontSize: "14px",
            lineHeight: 1,
          },
          onClick: (e) => { e.stopPropagation(); setShowModal(true); },
          title: "Details",
        },
        React.createElement("span", { className: "material-symbols-outlined", style: { fontSize: "14px" } }, "info")
      ),
      React.createElement(Handle, { type: "source", position: Position.Bottom, style: { background: "#3c4947", width: 8, height: 8 } })
    ),
    showModal ? React.createElement(
      "div",
      {
        style: {
          position: "fixed", inset: 0, zIndex: 100,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.6)",
        },
        onClick: () => setShowModal(false),
      },
      React.createElement(
        "div",
        {
          style: {
            background: "#1e2023", border: "1px solid #3c4947", borderRadius: "8px",
            padding: "20px", maxWidth: "400px", width: "90%",
            color: "#e2e2e6", fontFamily: "Inter, sans-serif",
          },
          onClick: (e) => e.stopPropagation(),
        },
        React.createElement("h3", { style: { margin: "0 0 8px", fontSize: "16px", fontWeight: 600 } }, escapeHtml(data.label || "")),
        React.createElement("p", { style: { margin: 0, fontSize: "13px", color: "#bbcac6", lineHeight: 1.5 } }, escapeHtml(data.desc || "No description")),
        React.createElement(
          "button",
          {
            type: "button",
            style: {
              marginTop: "12px", padding: "6px 16px",
              background: "#333538", border: "1px solid #3c4947",
              borderRadius: "6px", color: "#e2e2e6", cursor: "pointer",
              fontFamily: "Inter, sans-serif", fontSize: "12px",
            },
            onClick: () => setShowModal(false),
          },
          "Close"
        )
      )
    ) : null
  );
}

const nodeTypes = { blueprint: BlueprintNode };

function layoutNodes(nodes) {
  const byLevel = {};
  nodes.forEach((n) => {
    const level = n.level != null ? n.level : 0;
    (byLevel[level] = byLevel[level] || []).push(n);
  });
  const H_SPACING = 220, V_SPACING = 140;
  const result = [];
  Object.keys(byLevel).sort((a, b) => Number(a) - Number(b)).forEach((level) => {
    const group = byLevel[level];
    const totalWidth = (group.length - 1) * H_SPACING;
    group.forEach((n, i) => {
      result.push({
        id: n.id,
        type: "blueprint",
        position: { x: i * H_SPACING - totalWidth / 2, y: Number(level) * V_SPACING + 20 },
        data: { label: n.label, level: n.level, desc: n.desc || "", status: n.status || "pending" },
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

export function PlanFlowchart({ blueprint, mode = "auto" }) {
  const normalized = React.useMemo(() => normalizeBlueprint(blueprint), [blueprint]);
  const validation = React.useMemo(() => validatePlanBlueprint({
    nodes: normalized.nodes,
    edges: normalized.edges,
  }), [normalized]);

  if (!validation.valid) {
    return React.createElement(
      "div",
      { className: "flex flex-col gap-2 max-w-3xl" },
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
  const flowNodes = React.useMemo(() => layoutNodes(normalized.nodes), [normalized.nodes]);
  const flowEdges = React.useMemo(() => convertEdges(normalized.edges), [normalized.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  React.useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  return React.createElement(
    "div",
    { className: "flex flex-col gap-2 max-w-3xl" },
    React.createElement(
      "div",
      { className: "flex items-center gap-2 mb-1" },
      React.createElement(
        "div",
        { className: "w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center" },
        React.createElement("span", { className: "material-symbols-outlined text-[14px] text-primary" }, "account_tree")
      ),
      React.createElement("span", { className: "font-label-caps text-label-caps text-primary" }, "Execution Plan"),
      React.createElement(
        "span",
        { className: `px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] ${blueprint.verification_passed ? "text-primary" : allDone ? "text-primary" : "text-on-surface-variant"}` },
        blueprint.verification_passed ? "Verified" : allDone ? "Done" : "In progress"
      ),
      React.createElement(
        "span",
        { className: "px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" },
        mode === "forced" ? "Plan mode" : "Auto-planned"
      ),
      React.createElement(
        "span",
        { className: "px-2 py-0.5 rounded-full bg-surface-container-highest font-code-sm text-[10px] text-on-surface-variant" },
        `${normalized.nodes.length} steps`
      )
    ),
    React.createElement(
      "div",
      { style: { height: "300px", border: "1px solid #3c4947", borderRadius: "8px", background: "#0a0c0e" } },
      React.createElement(
        ReactFlow,
        {
          nodes,
          edges,
          onNodesChange,
          onEdgesChange,
          nodeTypes,
          fitView: true,
          fitViewOptions: { padding: 0.2 },
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
        React.createElement(Background, { color: "#1e2023", gap: 20 })
      )
    )
  );
}
