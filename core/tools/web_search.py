from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web or read a specific URL with structured success and failure payloads."

    supported_modes: tuple[str, ...] = ("search", "read_url")

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(self.supported_modes),
                    "description": "Choose 'search' to search the web or 'read_url' to fetch one page.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text. Required when mode is 'search'.",
                },
                "url": {
                    "type": "string",
                    "description": "Absolute URL to read. Required when mode is 'read_url'.",
                },
                "result_count": {
                    "type": "integer",
                    "description": "Optional maximum number of search results to return.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider hint for future search backends.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs) -> ToolResult:
        mode = kwargs.get("mode")
        provider = str(kwargs.get("provider") or "duckduckgo").strip().lower()
        if not isinstance(mode, str) or mode not in self.supported_modes:
            return ToolResult(success=False, output="Missing or unsupported argument: mode", data={"status": "invalid_input"})
        if mode == "search":
            query = kwargs.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(success=False, output="Missing required argument: query", data={"status": "invalid_input"})
            result_count = kwargs.get("result_count", 5)
            try:
                normalized_count = max(1, min(int(result_count), 10))
            except (TypeError, ValueError):
                return ToolResult(success=False, output="result_count must be an integer between 1 and 10", data={"status": "invalid_input"})
            return self._search(query.strip(), provider=provider, result_count=normalized_count)
        if mode == "read_url":
            url = kwargs.get("url")
            if not isinstance(url, str) or not url.strip():
                return ToolResult(success=False, output="Missing required argument: url", data={"status": "invalid_input"})
            return self._read_url(url.strip(), provider=provider)
        return ToolResult(success=False, output="Unsupported web_search mode", data={"status": "invalid_input"})

    def _search(self, query: str, *, provider: str, result_count: int) -> ToolResult:
        if provider not in {"duckduckgo", "default"}:
            return ToolResult(
                success=False,
                output=f"Unsupported web_search provider: {provider}",
                data={"status": "unsupported_provider", "mode": "search", "provider": provider, "results": []},
            )
        search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
        fetched = _fetch_text(search_url)
        if not fetched.success:
            return ToolResult(
                success=False,
                output=fetched.output,
                data={"status": fetched.data.get("status"), "mode": "search", "provider": provider, "results": []},
            )
        results = _parse_duckduckgo_results(str(fetched.data.get("content") or ""), limit=result_count)
        return ToolResult(
            success=True,
            output=f"web_search returned {len(results)} result(s) for '{query}'",
            data={"status": "ok", "mode": "search", "provider": provider, "query": query, "results": results},
        )

    def _read_url(self, url: str, *, provider: str) -> ToolResult:
        if not _is_http_url(url):
            return ToolResult(
                success=False,
                output="url must be an absolute http or https URL",
                data={"status": "invalid_url", "mode": "read_url", "provider": provider, "content": ""},
            )
        fetched = _fetch_text(url)
        if not fetched.success:
            return ToolResult(
                success=False,
                output=fetched.output,
                data={"status": fetched.data.get("status"), "mode": "read_url", "provider": provider, "content": ""},
            )
        raw_content = str(fetched.data.get("content") or "")
        title = _extract_title(raw_content)
        content = _extract_readable_text(raw_content)
        return ToolResult(
            success=True,
            output=f"web_search read {url}",
            data={"status": "ok", "mode": "read_url", "provider": provider, "url": url, "title": title, "content": content},
        )


def _is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fetch_text(url: str) -> ToolResult:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "devenv-ai/0.1 web_search",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        return ToolResult(success=False, output=f"HTTP fetch failed with status {exc.code}", data={"status": "fetch_failed"})
    except urllib.error.URLError as exc:
        return ToolResult(success=False, output=f"Network request failed: {exc.reason}", data={"status": "network_error"})
    except ValueError:
        return ToolResult(success=False, output="Invalid URL", data={"status": "invalid_url"})
    except OSError as exc:
        return ToolResult(success=False, output=f"Failed to read URL: {exc}", data={"status": "read_failed"})
    return ToolResult(success=True, output="fetched", data={"status": "ok", "content": payload})


def _parse_duckduckgo_results(raw_html: str, *, limit: int) -> list[dict[str, str]]:
    matches = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for url, title_html in matches:
        clean_url = html.unescape(url)
        clean_title = _strip_html(title_html)
        if clean_url and clean_title:
            results.append({"title": clean_title, "url": clean_url})
        if len(results) >= limit:
            break
    return results


def _extract_title(raw_html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    return _strip_html(match.group(1)) if match else ""


def _extract_readable_text(raw_html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()
    cleaned = _normalize_space(parser.text())
    return cleaned[:4000]


def _strip_html(value: str) -> str:
    return _normalize_space(re.sub(r"<[^>]+>", " ", html.unescape(value)))


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)
