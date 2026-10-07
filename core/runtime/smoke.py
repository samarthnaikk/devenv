from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

from core.logging_utils import configure_logging

from .context_builder import ContextBuilderService
from .kernel import DevenvKernel
from .models import DEFAULT_MAX_CONSECUTIVE_TOOLS, PlanningMode
from .tooling import build_runtime_tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single Devenv runtime turn without the TUI loop.")
    parser.add_argument("workspace", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("prompt", help="Single prompt to send through the runtime.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument(
        "--max-consecutive-tools",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_TOOLS,
    )
    parser.add_argument("--performance-mode", default="low", choices=("low", "medium", "high"))
    parser.add_argument("--planning-mode", default=PlanningMode.AUTO.value, choices=tuple(mode.value for mode in PlanningMode))
    parser.add_argument("--backend-preference", default="opencode")
    parser.add_argument("--enable-opencode-backend", action="store_true")
    parser.add_argument("--enable-ollama-backend", action="store_true")
    parser.add_argument("--enable-codex-backend", action="store_true")
    parser.add_argument("--allow-codex-context", action="store_true")
    parser.add_argument("--allow-opencode-context", action="store_true")
    parser.add_argument("--selected-tool", action="append", default=[])
    parser.add_argument("--continue-plan", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    workspace_path = str(Path(args.workspace).expanduser().resolve())
    configure_logging(args.log_level, workspace=workspace_path)
    kernel = DevenvKernel(
        workspace_path=workspace_path,
        db_path=args.db_path,
        vector_dir=args.vector_dir,
    )
    context_builder = ContextBuilderService(
        workspace_path,
        memory=kernel.memory,
        performance_mode=args.performance_mode,
    )
    allowed_providers: set[str] = set()
    if args.allow_codex_context:
        allowed_providers.add("codex")
    if args.allow_opencode_context:
        allowed_providers.add("opencode")
    context_builder.set_runtime_allowed_providers(allowed_providers)
    kernel.context_builder = context_builder
    for tool in build_runtime_tools(kernel.memory, context_builder=context_builder):
        kernel.register_tool(tool)
    planning_mode = PlanningMode(args.planning_mode)
    holder: dict[str, object] = {}

    def _run_turn() -> None:
        try:
            holder["result"] = kernel.execute_turn(
                args.prompt,
                max_consecutive_tools=args.max_consecutive_tools,
                planning_mode=planning_mode,
                continue_plan=args.continue_plan,
                local_only=args.local_only,
                selected_tools=args.selected_tool,
                backend_preference=args.backend_preference,
                opencode_enabled=args.enable_opencode_backend,
                ollama_enabled=args.enable_ollama_backend,
                codex_enabled=args.enable_codex_backend,
            )
        except Exception as exc:  # pragma: no cover - defensive
            holder["error"] = exc

    budget = _planning_budget_seconds()
    worker = threading.Thread(target=_run_turn, daemon=True)
    worker.start()
    worker.join(timeout=budget)
    if worker.is_alive():
        # Bounded latency: never hang the single-turn harness on a slow planning
        # model. Emit the classification so callers can still verify it.
        print(json.dumps(
            {
                "final_response": "",
                "steps": [],
                "total_usage": {},
                "metadata": {"planning_budget_exceeded": True},
                "ai_logs": [],
                "system_logs": [
                    f"Planning mode: {planning_mode.value}",
                    f"Planning budget exceeded ({budget:.0f}s).",
                ],
                "elapsed_ms": int(budget * 1000),
            },
            indent=2,
            sort_keys=True,
        ))
        kernel.close()
        return 0

    result = holder.get("result")
    kernel.close()
    if result is None:
        error = holder.get("error")
        raise SystemExit(f"Turn failed: {error}")
    print(json.dumps(
        {
            "final_response": result.final_response,
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "success": step.success,
                    "is_sandboxed_violation": step.is_sandboxed_violation,
                    "output": step.output,
                }
                for step in result.steps
            ],
            "total_usage": result.total_usage,
            "metadata": result.metadata,
            "ai_logs": result.ai_logs,
            "system_logs": result.system_logs,
            "elapsed_ms": result.elapsed_ms,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def _planning_budget_seconds() -> float:
    raw = os.getenv("DEVENV_PLANNING_BUDGET_SECONDS", "300").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 300.0


if __name__ == "__main__":
    raise SystemExit(main())
