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

    supported_modes: tuple[str, ...] = ("search", "search_images", "read_url")

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
        if mode == "search_images":
            query = kwargs.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(success=False, output="Missing required argument: query", data={"status": "invalid_input"})
            result_count = kwargs.get("result_count", 5)
            try:
                normalized_count = max(1, min(int(result_count), 10))
            except (TypeError, ValueError):
                return ToolResult(success=False, output="result_count must be an integer between 1 and 10", data={"status": "invalid_input"})
            return self._search_images(query.strip(), result_count=normalized_count)
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
        results, status, detail = search_web(query, provider=provider, result_count=result_count)
        if status != "ok":
            return ToolResult(
                success=False,
                output=detail,
                data={"status": status, "mode": "search", "provider": provider, "query": query, "results": []},
            )
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

    def _search_images(self, query: str, *, result_count: int) -> ToolResult:
        results = search_image_web(query, result_count=result_count)
        if not results:
            return ToolResult(
                success=False,
                output=f"The image search did not return any results for '{query}'.",
                data={"status": "no_results", "mode": "search_images", "query": query, "results": []},
            )
        return ToolResult(
            success=True,
            output=f"web_search returned {len(results)} image result(s) for '{query}'",
            data={"status": "ok", "mode": "search_images", "query": query, "results": results},
        )


def search_web(query: str, *, provider: str = "duckduckgo", result_count: int = 5) -> tuple[list[dict[str, str]], str, str]:
    if provider not in {"duckduckgo", "default"}:
        return [], "unsupported_provider", f"Unsupported web_search provider: {provider}"

    endpoints = (
        f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}",
        f"https://www.bing.com/search?format=rss&q={urllib.parse.quote_plus(query)}",
        f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}",
    )
    failure_messages: list[str] = []
    for url in endpoints:
        fetched = _fetch_text(url)
        if not fetched.success:
            failure_messages.append(fetched.output)
            continue
        raw_html = str(fetched.data.get("content") or "")
        results = (
            _parse_bing_rss_results(raw_html, limit=result_count)
            if "bing.com/search?format=rss" in url
            else _parse_bing_results(raw_html, limit=result_count)
            if "bing.com/search" in url
            else _parse_duckduckgo_results(raw_html, limit=result_count)
        )
        if results:
            return results, "ok", "ok"
    if failure_messages:
        return [], str(fetched.data.get("status") if "fetched" in locals() else "network_error"), failure_messages[-1]
    return [], "no_results", f"The web search did not return any results for '{query}'."


def _parse_bing_rss_results(raw_xml: str, *, limit: int) -> list[dict[str, str]]:
    matches = re.findall(
        r"<item>\s*<title>(?P<title>.*?)</title>\s*<link>(?P<url>.*?)</link>",
        raw_xml,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for title_text, url in matches:
        clean_url = html.unescape(url).strip()
        clean_title = _strip_html(title_text)
        if not clean_url or not clean_title or clean_url in seen_urls:
            continue
        results.append({"title": clean_title, "url": clean_url})
        seen_urls.add(clean_url)
        if len(results) >= limit:
            break
    return results


def _is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fetch_text(url: str) -> ToolResult:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
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
    results: list[dict[str, str]] = []
    patterns = (
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        r'<a[^>]*rel="nofollow"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        r'<a[^>]*href="(?P<url>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>',
    )
    seen_urls: set[str] = set()
    for pattern in patterns:
        matches = re.findall(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL)
        for url, title_html in matches:
            clean_url = _normalize_search_result_url(url)
            clean_title = _strip_html(title_html)
            if not clean_url or not clean_title or clean_url in seen_urls:
                continue
            if clean_url.startswith("https://duckduckgo.com/") or clean_url.startswith("http://duckduckgo.com/"):
                continue
            results.append({"title": clean_title, "url": clean_url})
            seen_urls.add(clean_url)
            if len(results) >= limit:
                return results
    return results


def _parse_bing_results(raw_html: str, *, limit: int) -> list[dict[str, str]]:
    matches = re.findall(
        r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2><a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a></h2>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for url, title_html in matches:
        clean_url = html.unescape(url).strip()
        clean_title = _strip_html(title_html)
        if not clean_url or not clean_title or clean_url in seen_urls:
            continue
        results.append({"title": clean_title, "url": clean_url})
        seen_urls.add(clean_url)
        if len(results) >= limit:
            break
    return results


def search_image_web(query: str, *, result_count: int = 5) -> list[dict[str, str]]:
    image_url = f"https://www.bing.com/images/search?q={urllib.parse.quote_plus(query)}"
    fetched = _fetch_text(image_url)
    if not fetched.success:
        return []
    raw_html = str(fetched.data.get("content") or "")
    results = _parse_bing_image_results(raw_html, limit=max(result_count * 3, result_count))
    if not results:
        results = _parse_image_tags(raw_html, limit=max(result_count * 3, result_count))
    return _rank_image_results(results, query=query, limit=result_count)


def _parse_bing_image_results(raw_html: str, *, limit: int) -> list[dict[str, str]]:
    matches = re.findall(
        r'murl&quot;:&quot;(?P<url>https?://[^"&]+).*?t&quot;:&quot;(?P<title>[^"&]+)',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for url, title in matches:
        clean_url = html.unescape(url).strip()
        clean_title = html.unescape(title).strip()
        if not clean_url or clean_url in seen_urls:
            continue
        results.append({"title": clean_title or "Image result", "url": clean_url})
        seen_urls.add(clean_url)
        if len(results) >= limit:
            break
    return results


def _parse_image_tags(raw_html: str, *, limit: int) -> list[dict[str, str]]:
    matches = re.findall(r'<img[^>]+src="(?P<url>https?://[^"]+)"[^>]*alt="(?P<title>[^"]*)"', raw_html, flags=re.IGNORECASE)
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for url, title in matches:
        clean_url = html.unescape(url).strip()
        if not clean_url or clean_url in seen_urls:
            continue
        results.append({"title": _strip_html(title) or "Image result", "url": clean_url})
        seen_urls.add(clean_url)
        if len(results) >= limit:
            break
    return results


def _rank_image_results(results: list[dict[str, str]], *, query: str, limit: int) -> list[dict[str, str]]:
    query_terms = _query_terms(query)
    if not results:
        return []
    ranked = sorted(
        results,
        key=lambda item: _score_image_result(item, query_terms=query_terms),
        reverse=True,
    )
    filtered: list[dict[str, str]] = []
    for item in ranked:
        score = _score_image_result(item, query_terms=query_terms)
        if query_terms and score <= 0:
            continue
        filtered.append(item)
        if len(filtered) >= limit:
            break
    if not filtered and len(ranked) == 1:
        return ranked[:limit]
    return filtered


def _score_image_result(item: dict[str, str], *, query_terms: tuple[str, ...]) -> int:
    title = str(item.get("title") or "").lower()
    url = str(item.get("url") or "").lower()
    haystack = f"{title} {url}"
    score = 0
    for term in query_terms:
        if term in title:
            score += 3
        elif term in haystack:
            score += 1
    if any(domain in url for domain in ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")):
        score -= 3
    if any(term in haystack for term in ("diagram", "wireframe", "architecture", "dashboard", "interface", "ui", "ux", "mockup", "illustration")):
        score += 1
    return score


def _query_terms(query: str) -> tuple[str, ...]:
    stop_words = {"the", "and", "for", "with", "from", "into", "that", "this", "your"}
    parts = [part for part in re.findall(r"[a-z0-9]+", query.lower()) if len(part) >= 3 and part not in stop_words]
    return tuple(dict.fromkeys(parts))


def _normalize_search_result_url(url: str) -> str:
    cleaned = html.unescape(url or "").strip()
    if not cleaned:
        return ""
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get("uddg", [""])[0]
        return urllib.parse.unquote(target).strip()
    return cleaned


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
