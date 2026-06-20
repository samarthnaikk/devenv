from __future__ import annotations

import logging
import re

from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex

from ._common import ensure_directory, is_probably_text, iter_directory, normalize_extension_filter, relative_display
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SearchTextTool(BaseTool):
    name = "search_text"
    description = "Search workspace text using literal, regex, or lightweight semantic matching."

    supported_modes: tuple[str, ...] = ("literal", "regex", "semantic")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or pattern to search for.",
                },
                "mode": {
                    "type": "string",
                    "description": "Search strategy.",
                    "enum": list(self.supported_modes),
                },
                "path": {
                    "type": "string",
                    "description": "Optional root directory to search. Defaults to the current workspace path.",
                },
                "ext_filter": {
                    "type": "string",
                    "description": "Optional extension filter such as '.py' or 'md'.",
                },
            },
            "required": ["query", "mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        mode = kwargs.get("mode")
        path = kwargs.get("path", ".")
        ext_filter = normalize_extension_filter(kwargs.get("ext_filter"))

        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="Missing required argument: query", data={})
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={})
        if not isinstance(path, str) or not path.strip():
            return ToolResult(success=False, output="path must be a non-empty string when provided", data={})

        try:
            root = ensure_directory(path)
            text_files = self._collect_text_files(root, ext_filter=ext_filter)

            if mode == "literal":
                matches = self._literal_search(text_files, query, root)
            elif mode == "regex":
                matches = self._regex_search(text_files, query, root)
            else:
                matches = self._semantic_search(text_files, query, root)

            logger.info("Searched text: root=%s mode=%s query=%s match_count=%s", root, mode, query, len(matches))
            return ToolResult(
                success=True,
                output=f"search_text found {len(matches)} match(es) for '{query}' using {mode} mode",
                data={
                    "path": str(root),
                    "query": query,
                    "mode": mode,
                    "ext_filter": ext_filter,
                    "matches": matches,
                    "count": len(matches),
                },
            )
        except (FileNotFoundError, NotADirectoryError, OSError, UnicodeDecodeError, re.error, ValueError) as exc:
            logger.error("search_text failed: path=%s mode=%s query=%s error=%s", path, mode, query, exc)
            return ToolResult(success=False, output=str(exc), data={})

    def _collect_text_files(self, root, *, ext_filter: str | None) -> list:
        files = []
        for entry, _depth in iter_directory(root, max_depth=32):
            if entry.is_dir():
                continue
            if ext_filter and entry.suffix.lower() != ext_filter:
                continue
            if not is_probably_text(entry):
                continue
            files.append(entry)
        return files

    def _literal_search(self, files: list, query: str, root) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        needle = query.lower()
        for file_path in files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if needle in line.lower():
                    matches.append(self._line_match(file_path, root, line_number, line))
        return matches

    def _regex_search(self, files: list, query: str, root) -> list[dict[str, object]]:
        pattern = re.compile(query)
        matches: list[dict[str, object]] = []
        for file_path in files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    matches.append(self._line_match(file_path, root, line_number, line))
        return matches

    def _semantic_search(self, files: list, query: str, root) -> list[dict[str, object]]:
        embedder = HashingEmbedder()
        index = InMemoryVectorIndex()
        file_lookup: dict[str, str] = {}
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            key = relative_display(file_path, root)
            file_lookup[key] = content
            index.upsert(key, content[:240], embedder.embed(content))

        matches = index.query(embedder.embed(query), top_k=5, min_similarity=0.15)
        payload: list[dict[str, object]] = []
        for match in matches:
            payload.append(
                {
                    "relative_path": match.node_id,
                    "path": str(root / match.node_id),
                    "similarity": round(match.similarity, 4),
                    "preview": file_lookup.get(match.node_id, "")[:240],
                }
            )
        return payload

    def _line_match(self, file_path, root, line_number: int, line: str) -> dict[str, object]:
        return {
            "path": str(file_path),
            "relative_path": relative_display(file_path, root),
            "line_number": line_number,
            "line": line,
        }
