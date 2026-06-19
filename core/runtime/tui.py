from __future__ import annotations

import argparse
from pathlib import Path

from core.logging_utils import configure_logging
from core.tools.read_file import ReadFileTool

from .kernel import DevenvKernel
from .models import RunConfig, RuntimeTurnResult


def render_banner(config: RunConfig) -> None:
    line = "=" * 80
    print(line)
    print(f" DEVENV CORE TUI v1.0 | Workspace: {config.workspace_path}")
    print(line)
    print("[SYSTEM]: Connected to Groq Local Pipeline. Memory Engine Online.")


def render_turn_result(result: RuntimeTurnResult) -> None:
    for step in result.steps:
        if step.is_sandboxed_violation:
            print(f"🔒 [SANDBOX CHECK]: {step.output}")
            continue
        status = "Success" if step.success else "Failure"
        print(f"⚙️  [EXECUTING TOOL]: {step.tool_name} -> {status}.")

    if result.final_response:
        print("\n[ASSISTANT]:")
        print(result.final_response)


def run_tui(config: RunConfig) -> int:
    kernel = DevenvKernel(
        workspace_path=config.workspace_path,
        db_path=config.db_path,
        vector_dir=config.vector_dir,
    )
    kernel.register_tool(ReadFileTool())
    render_banner(config)

    while True:
        try:
            prompt = input("devenv@local_workspace:~$ ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0

        print("⏳ [RETRIEVING MEMORY CONTEXT]...")
        print("🤖 [AI REASONING]...")
        result = kernel.execute_turn(prompt, max_consecutive_tools=config.max_consecutive_tools)
        render_turn_result(result)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Devenv runtime TUI.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--max-consecutive-tools", type=int, default=5)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = RunConfig(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
        max_consecutive_tools=args.max_consecutive_tools,
    )
    return run_tui(config)


if __name__ == "__main__":
    raise SystemExit(main())
