from __future__ import annotations

import ast
import compileall
import logging
import sys
import unittest
from pathlib import Path

from ._common import ensure_existing_path, iter_directory
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class RunDiagnosticsTool(BaseTool):
    name = "run_diagnostics"
    description = "Run summarized tests, syntax checks, and lightweight type diagnostics."

    supported_modes: tuple[str, ...] = ("tests", "lint", "types")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Diagnostic suite to run.",
                    "enum": list(self.supported_modes),
                },
                "target_path": {
                    "type": "string",
                    "description": "Optional path to a file or directory to inspect.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        target_path = kwargs.get("target_path", ".")

        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if not isinstance(target_path, str) or not target_path.strip():
            return ToolResult(success=False, output="target_path must be a non-empty string when provided", data={})

        try:
            target = ensure_existing_path(target_path)
            if mode == "tests":
                return self._run_tests(target)
            if mode == "lint":
                return self._run_lint(target)
            return self._run_types(target)
        except (FileNotFoundError, OSError, SyntaxError, ValueError) as exc:
            logger.error("run_diagnostics failed: target=%s mode=%s error=%s", target_path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _run_tests(self, target: Path) -> ToolResult:
        loader = unittest.TestLoader()
        if target.is_file():
            suite = loader.discover(str(target.parent), pattern=target.name)
        else:
            suite = loader.discover(str(target))

        result = unittest.TestResult()
        suite.run(result)
        success = result.wasSuccessful()
        logger.info("Ran diagnostics tests: target=%s success=%s", target, success)
        return ToolResult(
            success=success,
            output=f"run_diagnostics tests completed: ran={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}",
            data={
                "mode": "tests",
                "target_path": str(target),
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
            },
        )

    def _run_lint(self, target: Path) -> ToolResult:
        success = compileall.compile_dir(str(target), quiet=1) if target.is_dir() else compileall.compile_file(str(target), quiet=1)
        logger.info("Ran diagnostics lint: target=%s success=%s", target, success)
        return ToolResult(
            success=bool(success),
            output=f"run_diagnostics lint completed for {target.name}",
            data={"mode": "lint", "target_path": str(target), "passed": bool(success)},
        )

    def _run_types(self, target: Path) -> ToolResult:
        issues: list[dict[str, object]] = []
        for file_path in self._python_files(target):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                missing = []
                for argument in node.args.args:
                    if argument.arg in {"self", "cls"}:
                        continue
                    if argument.annotation is None:
                        missing.append(argument.arg)
                if node.returns is None:
                    missing.append("return")
                if missing:
                    issues.append(
                        {
                            "path": str(file_path),
                            "function": node.name,
                            "line": node.lineno,
                            "missing": missing,
                        }
                    )

        success = not issues
        logger.info("Ran diagnostics types: target=%s success=%s issue_count=%s", target, success, len(issues))
        return ToolResult(
            success=success,
            output=f"run_diagnostics types completed with {len(issues)} issue(s)",
            data={
                "mode": "types",
                "target_path": str(target),
                "passed": success,
                "issues": issues,
                "python_version": sys.version.split()[0],
            },
        )

    def _python_files(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target] if target.suffix.lower() == ".py" else []
        files: list[Path] = []
        for entry, _depth in iter_directory(target, max_depth=32):
            if entry.is_file() and entry.suffix.lower() == ".py":
                files.append(entry)
        return files
