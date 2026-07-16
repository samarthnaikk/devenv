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

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...], int], ToolResult] = {}

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
        sources = _expand_requested_sources(query_text, sources)
        cache_key = (query_text.lower(), tuple(sources), result_count)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        normalized_query = _normalize_knowledge_query(query_text)
        resources: list[dict[str, object]] = []
        errors: list[str] = []
        seen_urls: set[str] = set()
        for source in sources:
            source_results: list[dict[str, str]] = []
            attempted_queries: list[str] = []
            if source == "github":
                source_results.extend(_search_github_repositories(normalized_query, limit=result_count))
                if len(source_results) < result_count:
                    source_results.extend(
                        _search_github_repository_links_via_web(query_text, normalized_query, limit=result_count, seen_urls=seen_urls | {item["url"] for item in source_results})
                    )
                if source_results:
                    resources.append({"source": source, "query": normalized_query, "attempted_queries": [normalized_query], "results": source_results[:result_count]})
                    seen_urls.update(item["url"] for item in source_results[:result_count])
                    continue
            for search_query in _build_source_queries(query_text, normalized_query, source):
                attempted_queries.append(search_query)
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
            resources.append({"source": source, "query": normalized_query, "attempted_queries": attempted_queries, "results": source_results})

        result_total = sum(len(group["results"]) for group in resources)
        status = "ok" if result_total > 0 else "no_results"
        output = (
            f"knowledge_search gathered {result_total} resource(s) for '{query_text}'"
            if result_total > 0
            else f"knowledge_search could not find resources for '{query_text}'"
        )
        result = ToolResult(
            success=result_total > 0,
            output=output,
            data={"status": status, "query": query_text, "resources": resources, "errors": errors},
        )
        self._cache[cache_key] = result
        return result


def _normalize_knowledge_query(query: str) -> str:
    cleaned = re.sub(
        r"\b(github|git hub|reddit|stackoverflow|stack overflow|quora|youtube|repo|repos|repositories|references|resources|examples|videos|threads|forums|similar|find|show|give|need|looking|look|for|and|please|could|would|want|wanna)\b",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(to|into|in)\s+(this|the)\s+(repo|repository|project|codebase|workspace)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(add|adding|integrate|integration|build|building|create|creating)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("chatapp", "chat app")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned or query.strip()


def _build_source_queries(original_query: str, normalized_query: str, source: str) -> list[str]:
    variants = _expand_query_variants(original_query, normalized_query)
    max_variants = {
        "github": 3,
        "documentation": 2,
        "stackoverflow": 2,
        "reddit": 2,
        "youtube": 2,
        "quora": 1,
        "general": 2,
    }.get(source, 2)
    built: list[str] = []
    for variant in variants[:max_variants]:
        for query_template in SOURCE_QUERIES[source]:
            built.append(query_template.format(query=variant))
    return list(dict.fromkeys([item.strip() for item in built if item.strip()]))


def _expand_query_variants(original_query: str, normalized_query: str) -> list[str]:
    base = normalized_query.strip() or original_query.strip()
    lowered = base.lower()
    variants = [base]
    if "chat app" in lowered or re.search(r"\bchat\b", lowered):
        variants.extend(
            [
                "chat app",
                "real time chat app",
                "realtime messaging app",
                "chat application architecture",
            ]
        )
    if "pdf" in lowered:
        variants.extend(["pdf generation", "export pdf", "pdf tool integration"])
    if any(term in lowered for term in ("calendar", "scheduling", "events")):
        variants.extend(["calendar app", "event scheduling ui", "calendar feature implementation"])
    compact_original = re.sub(r"\b(codebase|repo|repository|workspace|project)\b", " ", original_query, flags=re.IGNORECASE)
    compact_original = re.sub(r"\s+", " ", compact_original).strip(" ,.-")
    if compact_original and compact_original.lower() != lowered:
        variants.append(compact_original)
    cleaned: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = re.sub(r"\s+", " ", variant).strip(" ,.-")
        lowered_variant = normalized.lower()
        if not normalized or lowered_variant in seen:
            continue
        seen.add(lowered_variant)
        cleaned.append(normalized)
    return cleaned[:6]


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


def _search_github_repository_links_via_web(
    original_query: str,
    normalized_query: str,
    *,
    limit: int,
    seen_urls: set[str],
) -> list[dict[str, str]]:
    repo_queries = [
        f"github {normalized_query}",
        f"github real time chat app" if "chat" in normalized_query.lower() else "",
        f"site:github.com {original_query}",
    ]
    results: list[dict[str, str]] = []
    seen_repo_urls = set(seen_urls)
    for search_query in [query for query in repo_queries if query]:
        web_results, status, _detail = search_web(search_query, result_count=limit)
        if status != "ok":
            continue
        for item in web_results:
            repo_url = _extract_github_repo_url(str(item.get("url") or ""))
            if not repo_url or repo_url in seen_repo_urls:
                continue
            seen_repo_urls.add(repo_url)
            results.append(
                {
                    "title": _repo_title_from_url(repo_url, fallback=str(item.get("title") or "").strip()),
                    "url": repo_url,
                    "query": search_query,
                }
            )
            if len(results) >= limit:
                return results
    return results


def _extract_github_repo_url(url: str) -> str:
    match = re.match(r"^https?://github\.com/([^/\s]+/[^/\s?#]+)", url.strip(), flags=re.IGNORECASE)
    if not match:
        return ""
    repo = match.group(1).strip("/")
    if repo.lower().endswith(("/issues", "/pulls", "/wiki")):
        return ""
    return f"https://github.com/{repo}"


def _repo_title_from_url(url: str, *, fallback: str) -> str:
    match = re.match(r"^https?://github\.com/([^/\s]+/[^/\s?#]+)", url, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return fallback or url


def _expand_requested_sources(query: str, sources: list[str]) -> list[str]:
    cleaned_sources = [source for source in sources if source in SOURCE_QUERIES]
    lowered = query.lower()
    if cleaned_sources == ["general"] and any(
        marker in lowered
        for marker in ("reference", "references", "repo", "repos", "repository", "github", "youtube", "reddit", "stackoverflow", "example", "examples", "tutorial")
    ):
        return ["github", "youtube", "documentation", "general"]
    return cleaned_sources or list(DEFAULT_SOURCES)
