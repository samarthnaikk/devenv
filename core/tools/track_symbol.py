from __future__ import annotations

import ast
import logging
import re

from ._common import ensure_directory, is_probably_text, iter_directory, relative_display
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TrackSymbolTool(BaseTool):
    name = "track_symbol"
    description = "Track symbol references and definitions across workspace files."

    supported_modes: tuple[str, ...] = ("references", "definitions")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to track across the workspace.",
                },
                "mode": {
                    "type": "string",
                    "description": "Whether to find references or definitions.",
                    "enum": list(self.supported_modes),
                },
                "path": {
                    "type": "string",
                    "description": "Optional root directory to scan. Defaults to the current workspace path.",
                },
            },
            "required": ["symbol", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        symbol = kwargs.get("symbol")
        mode = kwargs.get("mode")
        path = kwargs.get("path", ".")

        if not isinstance(symbol, str) or not symbol.strip():
            return ToolResult(success=False, output="Missing required argument: symbol", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="path must be a non-empty string when provided", data={})

        try:
            root = ensure_directory(path)
            files = self._collect_text_files(root)
            if mode == "references":
                matches = self._references(files, symbol, root)
            else:
                matches = self._definitions(files, symbol, root)

            logger.info("Tracked symbol: root=%s mode=%s symbol=%s match_count=%s", root, mode, symbol, len(matches))
            return ToolResult(
                success=True,
                output=f"track_symbol found {len(matches)} match(es) for '{symbol}' using {mode} mode",
                data={
                    "path": str(root),
                    "symbol": symbol,
                    "mode": mode,
                    "matches": matches,
                    "count": len(matches),
                },
            )
        except (FileNotFoundError, NotADirectoryError, OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
            logger.error("track_symbol failed: path=%s mode=%s symbol=%s error=%s", path, mode, symbol, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _collect_text_files(self, root) -> list:
        files = []
        for entry, _depth in iter_directory(root, max_depth=32):
            if entry.is_dir() or not is_probably_text(entry):
                continue
            files.append(entry)
        return files

    def _references(self, files: list, symbol: str, root) -> list[dict[str, object]]:
        pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        matches: list[dict[str, object]] = []
        for file_path in files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": str(file_path),
                            "relative_path": relative_display(file_path, root),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
        return matches

    def _definitions(self, files: list, symbol: str, root) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for file_path in files:
            if file_path.suffix.lower() != ".py":
                continue
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            for node in ast.walk(tree):
                payload = self._definition_payload(node, symbol)
                if payload is None:
                    continue
                payload["path"] = str(file_path)
                payload["relative_path"] = relative_display(file_path, root)
                matches.append(payload)
        matches.sort(key=lambda item: (item["relative_path"], item["line"]))
        return matches

    def _definition_payload(self, node: ast.AST, symbol: str) -> dict[str, object] | None:
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return {"kind": "class", "name": symbol, "line": node.lineno}
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            return {"kind": "function", "name": symbol, "line": node.lineno}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return {"kind": "assignment", "name": symbol, "line": node.lineno}
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                alias_name = alias.asname or alias.name
                if alias_name == symbol:
                    return {"kind": "import", "name": symbol, "line": node.lineno}
        return None
