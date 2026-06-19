from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.logging_utils import configure_logging
from core.tools.read_file import ReadFileTool

from .kernel import DevenvKernel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single Devenv runtime turn without the TUI loop.")
    parser.add_argument("workspace", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("prompt", help="Single prompt to send through the runtime.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--max-consecutive-tools", type=int, default=5)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    kernel = DevenvKernel(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
    )
    kernel.register_tool(ReadFileTool())
    result = kernel.execute_turn(args.prompt, max_consecutive_tools=args.max_consecutive_tools)
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
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
