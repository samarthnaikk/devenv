from __future__ import annotations

import ast
import logging

from ._common import ensure_file
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class InspectSymbolsTool(BaseTool):
    name = "inspect_symbols"
    description = "Inspect Python classes, functions, signatures, and docstrings using AST parsing."

    supported_modes: tuple[str, ...] = ("outline", "signatures", "documentation")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a Python source file.",
                },
                "mode": {
                    "type": "string",
                    "description": "Symbol inspection mode.",
                    "enum": list(self.supported_modes),
                },
            },
            "required": ["path", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        mode = kwargs.get("mode")

        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="Missing required argument: path", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})

        try:
            file_path = ensure_file(path)
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))

            if mode == "outline":
                payload = {"symbols": self._outline(tree)}
            elif mode == "signatures":
                payload = {"signatures": self._signatures(tree)}
            else:
                payload = {"documentation": self._documentation(tree)}

            logger.info("Inspected symbols: path=%s mode=%s", file_path, mode)
            return ToolResult(
                success=True,
                output=f"inspect_symbols completed for {file_path.name} using {mode} mode",
                data={"path": str(file_path), "mode": mode, **payload},
            )
        except (FileNotFoundError, IsADirectoryError, OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
            logger.error("inspect_symbols failed: path=%s mode=%s error=%s", path, mode, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _outline(self, tree: ast.AST) -> list[dict[str, object]]:
        symbols: list[dict[str, object]] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    {
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [child.name for child in node.body if isinstance(child, ast.FunctionDef)],
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                symbols.append(
                    {
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "signature": self._signature_string(node),
                    }
                )
        return symbols

    def _signatures(self, tree: ast.AST) -> list[dict[str, object]]:
        signatures: list[dict[str, object]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            signatures.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "parameters": [self._parameter_payload(argument) for argument in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns is not None else None,
                }
            )
        signatures.sort(key=lambda item: item["line"])
        return signatures

    def _documentation(self, tree: ast.AST) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        module_doc = ast.get_docstring(tree)
        if module_doc:
            docs.append({"scope": "module", "name": "__module__", "docstring": module_doc})

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                if doc:
                    docs.append({"scope": "class", "name": node.name, "docstring": doc})
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_doc = ast.get_docstring(child)
                        if method_doc:
                            docs.append({"scope": "method", "name": f"{node.name}.{child.name}", "docstring": method_doc})
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if doc:
                    docs.append({"scope": "function", "name": node.name, "docstring": doc})
        return docs

    def _signature_string(self, node: ast.FunctionDef) -> str:
        arguments = [self._parameter_text(argument) for argument in node.args.args]
        returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
        return f"{node.name}({', '.join(arguments)}){returns}"

    def _parameter_payload(self, argument: ast.arg) -> dict[str, object]:
        annotation = ast.unparse(argument.annotation) if argument.annotation is not None else None
        return {"name": argument.arg, "annotation": annotation}

    def _parameter_text(self, argument: ast.arg) -> str:
        annotation = ast.unparse(argument.annotation) if argument.annotation is not None else None
        return argument.arg if annotation is None else f"{argument.arg}: {annotation}"
