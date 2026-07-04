from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from core.memory.storage import SQLiteMemoryStore

from .models import RunConfig, SetupCheckStatus, SetupReadiness
from .state import resolve_memory_paths


def inspect_setup(
    config: RunConfig,
    *,
    include_optional: bool = True,
    apply_changes: bool = False,
    warm_model_cache: bool = False,
) -> SetupReadiness:
    workspace_ready = Path(config.workspace_path).is_dir()
    dependency_names = ("lancedb", "mcp", "sentence_transformers")
    missing_dependencies = _find_missing_dependencies(dependency_names)
    dependency_ready = not missing_dependencies
    opencode_path = shutil.which("opencode")
    opencode_ready, opencode_detail = _check_opencode(opencode_path)
    db_path, vector_dir = resolve_memory_paths(config.db_path, config.vector_dir, workspace_path=config.workspace_path)
    memory_ready, memory_detail = _ensure_workspace_state(db_path=db_path, vector_dir=vector_dir, apply_changes=apply_changes)

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
            status="ready" if dependency_ready else "failed",
            detail="All required Python dependencies are importable."
            if dependency_ready
            else f"Missing required Python dependencies: {', '.join(missing_dependencies)}.",
        ),
        SetupCheckStatus(
            name="opencode",
            required=True,
            status="ready" if opencode_ready else "failed",
            detail=opencode_detail,
        ),
        SetupCheckStatus(
            name="workspace_state",
            required=True,
            status="ready" if memory_ready else "failed",
            detail=memory_detail,
        ),
    )
    optional_checks = (
        _build_optional_check("sentence_transformer_cache", _check_sentence_transformer_cache(warm_model_cache=warm_model_cache)),
        _build_optional_check("web_search_prerequisites", _check_web_search_prerequisites()),
        _build_optional_check("latex_pdf_toolchain", _check_latex_pdf_toolchain()),
    )
    ready = all(check.status == "ready" for check in required_checks)
    summary = "Devenv setup is ready." if ready else "Devenv setup requires attention."
    if apply_changes and memory_ready:
        summary = "Devenv setup checks passed and workspace state is initialized."
    return SetupReadiness(
        ready=ready,
        summary=summary,
        required_checks=required_checks,
        optional_checks=optional_checks if include_optional else (),
        checked_at=_utc_now_iso(),
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


def _find_missing_dependencies(dependency_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for dependency_name in dependency_names:
        if importlib.util.find_spec(dependency_name) is None:
            missing.append(dependency_name)
    return missing


def _check_opencode(opencode_path: str | None) -> tuple[bool, str]:
    if not opencode_path:
        return False, "OpenCode CLI was not found on PATH."
    try:
        completed = subprocess.run(
            [opencode_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"OpenCode CLI is installed but not healthy: {exc}."
    version = (completed.stdout or completed.stderr or "").strip() or "version reported"
    return True, f"OpenCode CLI available: {version}."


def _ensure_workspace_state(*, db_path: str, vector_dir: str, apply_changes: bool) -> tuple[bool, str]:
    db_file = Path(db_path)
    vector_root = Path(vector_dir)
    if apply_changes:
        try:
            SQLiteMemoryStore(str(db_file))
            vector_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to initialize workspace memory state: {exc}."
    db_exists = db_file.is_file()
    vector_exists = vector_root.is_dir()
    if db_exists and vector_exists:
        return True, f"Workspace memory state ready at {db_file} and {vector_root}."
    missing: list[str] = []
    if not db_exists:
        missing.append(str(db_file))
    if not vector_exists:
        missing.append(str(vector_root))
    return False, f"Workspace memory state is missing: {', '.join(missing)}."


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_optional_check(name: str, result: tuple[str, str]) -> SetupCheckStatus:
    status, detail = result
    return SetupCheckStatus(name=name, required=False, status=status, detail=detail)


def _check_sentence_transformer_cache(*, warm_model_cache: bool) -> tuple[str, str]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_glob = "models--sentence-transformers--all-MiniLM-L6-v2*"
    has_local_cache = any(cache_root.glob(model_glob)) if cache_root.is_dir() else False
    if has_local_cache:
        return "ready", "Sentence-transformer cache is present locally."
    if not warm_model_cache:
        return "pending", "Sentence-transformer cache is not warmed locally yet."
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=False)
    except Exception as exc:
        return "failed", f"Model cache warmup failed: {exc}."
    return "ready", "Sentence-transformer cache warmup completed."


def _check_web_search_prerequisites() -> tuple[str, str]:
    try:
        has_urlopen = callable(getattr(urllib.request, "urlopen", None))
    except Exception as exc:
        return "failed", f"Python HTTP support is unavailable: {exc}."
    if not has_urlopen:
        return "failed", "Python HTTP support is unavailable."
    return "ready", "Python HTTP stack is available for the web_search tool."


def _check_latex_pdf_toolchain() -> tuple[str, str]:
    latex_engine = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
    renderer = shutil.which("pdftoppm") or shutil.which("mutool")
    if latex_engine and renderer:
        return "ready", f"PDF toolchain available via {Path(latex_engine).name} and {Path(renderer).name}."
    if latex_engine:
        return "pending", f"LaTeX engine available via {Path(latex_engine).name}, but PDF render validation tool is missing."
    return "pending", "No LaTeX engine detected yet."


if __name__ == "__main__":
    raise SystemExit(main())
