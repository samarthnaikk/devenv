from __future__ import annotations

import html
import re
import urllib.parse

from .base import BaseTool, ToolResult
from .web_search import _fetch_text, search_web

DEFAULT_SOURCES: tuple[str, ...] = (
    "github",
    "documentation",
    "stackoverflow",
    "reddit",
    "youtube",
    "quora",
    "general",
)

SOURCE_QUERIES: dict[str, tuple[str, ...]] = {
    "github": ("site:github.com {query}",),
    "documentation": ("{query} documentation", "{query} docs"),
    "stackoverflow": ("site:stackoverflow.com {query}",),
    "reddit": ("site:reddit.com {query}",),
    "youtube": ("site:youtube.com {query}",),
    "quora": ("site:quora.com {query}",),
    "general": ("{query}",),
}


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Pull grouped outside references for a topic across GitHub, docs, forums, and general web sources."

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic, feature, or project idea to research.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of sources: github, documentation, stackoverflow, reddit, youtube, quora, general.",
                },
                "result_count": {
                    "type": "integer",
                    "description": "Maximum results per source.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="Missing required argument: query", data={"status": "invalid_input", "resources": []})

        requested_sources = kwargs.get("sources")
        if requested_sources is None:
            sources = list(DEFAULT_SOURCES)
        elif isinstance(requested_sources, list) and all(isinstance(item, str) for item in requested_sources):
            sources = [item.strip().lower() for item in requested_sources if item.strip()]
        else:
            return ToolResult(success=False, output="sources must be a list of strings", data={"status": "invalid_input", "resources": []})

        invalid_sources = [source for source in sources if source not in SOURCE_QUERIES]
        if invalid_sources:
            return ToolResult(
                success=False,
                output=f"Unsupported knowledge_search source(s): {', '.join(invalid_sources)}",
                data={"status": "invalid_input", "resources": []},
            )

        try:
            result_count = max(1, min(int(kwargs.get("result_count", 3)), 5))
        except (TypeError, ValueError):
            return ToolResult(success=False, output="result_count must be an integer between 1 and 5", data={"status": "invalid_input", "resources": []})

        query_text = query.strip()
        normalized_query = _normalize_knowledge_query(query_text)
        resources: list[dict[str, object]] = []
        errors: list[str] = []
        seen_urls: set[str] = set()
        for source in sources:
            source_results: list[dict[str, str]] = []
            if source == "github":
                source_results.extend(_search_github_repositories(normalized_query, limit=result_count))
                if source_results:
                    resources.append({"source": source, "results": source_results[:result_count]})
                    seen_urls.update(item["url"] for item in source_results[:result_count])
                    continue
            for query_template in SOURCE_QUERIES[source]:
                search_query = query_template.format(query=normalized_query)
                results, status, detail = search_web(search_query, result_count=result_count)
                if status != "ok":
                    errors.append(f"{source}: {detail}")
                    continue
                for item in results:
                    url = str(item.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    source_results.append({"title": str(item.get("title") or "").strip(), "url": url, "query": search_query})
                    seen_urls.add(url)
                    if len(source_results) >= result_count:
                        break
                if len(source_results) >= result_count:
                    break
            resources.append({"source": source, "results": source_results})

        result_total = sum(len(group["results"]) for group in resources)
        status = "ok" if result_total > 0 else "no_results"
        output = (
            f"knowledge_search gathered {result_total} resource(s) for '{query_text}'"
            if result_total > 0
            else f"knowledge_search could not find resources for '{query_text}'"
        )
        return ToolResult(
            success=result_total > 0,
            output=output,
            data={"status": status, "query": query_text, "resources": resources, "errors": errors},
        )


def _normalize_knowledge_query(query: str) -> str:
    cleaned = re.sub(
        r"\b(github|git hub|reddit|stackoverflow|stack overflow|quora|youtube|repo|repos|repositories|references|resources|examples|videos|threads|forums|similar|find|show|give|need|looking|look|for|and)\b",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned or query.strip()


def _search_github_repositories(query: str, *, limit: int) -> list[dict[str, str]]:
    search_url = f"https://github.com/search?q={urllib.parse.quote_plus(query)}&type=repositories"
    fetched = _fetch_text(search_url)
    if not fetched.success:
        return []
    raw_html = str(fetched.data.get("content") or "")
    matches = re.findall(
        r'"/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/(?:star|unstar)"',
        raw_html,
        flags=re.IGNORECASE,
    )
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for repo_name in matches:
        cleaned_repo = html.unescape(repo_name).strip("/")
        if not cleaned_repo or cleaned_repo in seen:
            continue
        seen.add(cleaned_repo)
        results.append(
            {
                "title": cleaned_repo,
                "url": f"https://github.com/{cleaned_repo}",
                "query": search_url,
            }
        )
        if len(results) >= limit:
            break
    return results
