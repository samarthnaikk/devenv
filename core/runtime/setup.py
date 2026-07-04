from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import RunConfig, SetupCheckStatus, SetupReadiness


def inspect_setup(
    config: RunConfig,
    *,
    include_optional: bool = True,
    apply_changes: bool = False,
    warm_model_cache: bool = False,
) -> SetupReadiness:
    del apply_changes, warm_model_cache

    workspace_ready = Path(config.workspace_path).is_dir()
    required_checks = (
        SetupCheckStatus(
            name="workspace",
            required=True,
            status="ready" if workspace_ready else "failed",
            detail="Workspace path is available." if workspace_ready else "Workspace path is missing or unreadable.",
        ),
        SetupCheckStatus(
            name="python_dependencies",
            required=True,
            status="pending",
            detail="Dependency verification is defined but has not run yet.",
        ),
        SetupCheckStatus(
            name="opencode",
            required=True,
            status="pending",
            detail="OpenCode CLI verification is defined but has not run yet.",
        ),
    )
    optional_checks = (
        SetupCheckStatus(
            name="sentence_transformer_cache",
            required=False,
            status="pending",
            detail="Model cache warmup is defined but has not run yet.",
        ),
        SetupCheckStatus(
            name="web_search_prerequisites",
            required=False,
            status="pending",
            detail="Web search prerequisite checks are defined but have not run yet.",
        ),
        SetupCheckStatus(
            name="latex_pdf_toolchain",
            required=False,
            status="pending",
            detail="LaTeX prerequisite checks are defined but have not run yet.",
        ),
    )
    ready = workspace_ready
    summary = "Setup command is available. Full bootstrap checks will be implemented in follow-up steps."
    if not workspace_ready:
        summary = "Workspace path is not ready."
    return SetupReadiness(
        ready=ready,
        summary=summary,
        required_checks=required_checks,
        optional_checks=optional_checks if include_optional else (),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or prepare a Devenv workspace.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path to inspect.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--performance-mode", default="medium")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--incognito", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply idempotent setup changes when supported.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional checks in the output.")
    parser.add_argument("--warm-model-cache", action="store_true", help="Warm the local model cache when supported.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    config = RunConfig(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
        performance_mode=args.performance_mode,
        no_memory=args.no_memory,
        incognito=args.incognito,
    )
    result = inspect_setup(
        config,
        include_optional=args.include_optional,
        apply_changes=args.apply,
        warm_model_cache=args.warm_model_cache,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary)
        for check in result.required_checks:
            print(f"[required:{check.status}] {check.name} - {check.detail}")
        for check in result.optional_checks:
            print(f"[optional:{check.status}] {check.name} - {check.detail}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
