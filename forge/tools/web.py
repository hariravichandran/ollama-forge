"""Web tool: search and fetch web content."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.web")

# Retry configuration for transient network errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # seconds — exponential backoff: 1.5, 2.25, 3.375...

# Transient HTTP status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class WebTool:
    """Web search and fetch operations."""

    name = "web"
    description = "Search the web and fetch content from URLs"

    CACHE_TTL = 6 * 3600  # 6 hours

    def __init__(self, working_dir: str = ".", cache_dir: str = ""):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(working_dir) / ".forge_state"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "web_search_cache.json"
        self._cache: dict[str, Any] = self._load_cache()
        # Persistent HTTP session for connection reuse
        self._http_session: Any = None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using DuckDuckGo and return results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Max results (default 5)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract text content from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a web tool function."""
        if function_name == "web_search":
            return self._search(args.get("query", ""), args.get("max_results", 5))
        elif function_name == "web_fetch":
            return self._fetch(args.get("url", ""))
        return f"Unknown function: {function_name}"

    def _search(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo with retry on transient failures."""
        if not query:
            return "Error: empty search query"

        # Check cache
        cache_key = f"search:{query}:{max_results}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))

                if not results:
                    return f"No results found for: {query}"

                lines = [f"Search results for: {query}\n"]
                for r in results:
                    title = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    lines.append(f"  {title}")
                    lines.append(f"  {href}")
                    lines.append(f"  {body}")
                    lines.append("")

                output = "\n".join(lines)
                self._set_cached(cache_key, output)
                return output

            except ImportError:
                return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("Search attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("Search failed after %d attempts: %s", MAX_RETRIES, e)

        return f"Search error (after {MAX_RETRIES} attempts): {last_error}"

    def _fetch(self, url: str) -> str:
        """Fetch text content from a URL with retry on transient failures."""
        if not url:
            return "Error: empty URL"

        # Check cache
        cache_key = f"fetch:{url}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                import requests
                if self._http_session is None:
                    self._http_session = requests.Session()
                    self._http_session.headers["User-Agent"] = "ollama-forge/0.1 (local AI assistant)"
                r = self._http_session.get(url, timeout=15)

                # Retry on transient HTTP errors
                if r.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {r.status_code}"
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                        log.warning("Fetch got HTTP %d (attempt %d), retrying in %.1fs", r.status_code, attempt + 1, delay)
                        time.sleep(delay)
                        continue

                r.raise_for_status()

                # Simple HTML to text extraction
                content = r.text
                if "<html" in content.lower():
                    content = self._html_to_text(content)

                # Truncate long content
                if len(content) > 15000:
                    content = content[:15000] + "\n... (truncated)"

                self._set_cached(cache_key, content)
                return content

            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("Fetch attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("Fetch failed after %d attempts: %s", MAX_RETRIES, e)

        return f"Fetch error (after {MAX_RETRIES} attempts): {last_error}"

    def _html_to_text(self, html: str) -> str:
        """Basic HTML to text conversion."""
        import re
        # Remove script and style tags
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"')
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Re-add paragraph breaks
        text = re.sub(r"\s{2,}", "\n\n", text)
        return text

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._cache))
        except OSError:
            pass

    def _get_cached(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry and time.time() - entry.get("ts", 0) < self.CACHE_TTL:
            return entry.get("data")
        return None

    def _set_cached(self, key: str, data: str) -> None:
        self._cache[key] = {"data": data, "ts": time.time()}
        self._save_cache()
