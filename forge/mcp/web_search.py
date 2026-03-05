"""Built-in web search MCP — DuckDuckGo, no API key required, enabled by default."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("mcp.web_search")

# Query limits
MAX_QUERY_LENGTH = 500
MAX_SEARCH_RESULTS = 20  # hard cap on results per query
MIN_CACHE_TTL = 60  # 1 minute minimum
MAX_CACHE_TTL = 86400  # 24 hours maximum
MAX_CACHE_ENTRIES = 500  # evict oldest when exceeded
MAX_BUILD_CONTEXT_QUERIES = 10  # max queries in build_context
MAX_BUILD_CONTEXT_RESULTS = 3  # results per query in build_context


class WebSearchMCP:
    """Built-in web search MCP server using DuckDuckGo.

    Enabled by default — no API key required. Provides privacy-first
    web search for any agent or conversation.
    """

    def __init__(
        self,
        max_results: int = 5,
        cache_ttl: int = 6 * 3600,
        cache_dir: str = ".forge_state",
    ):
        self.max_results = min(max(1, max_results), MAX_SEARCH_RESULTS)
        self.cache_ttl = min(max(cache_ttl, MIN_CACHE_TTL), MAX_CACHE_TTL)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "web_search_cache.json"
        self._cache = self._load_cache()
        self.enabled = True

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, str]]:
        """Search DuckDuckGo and return results.

        Returns list of dicts with keys: title, href, body
        """
        if not self.enabled:
            return []
        if not query or not query.strip():
            return []
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        n = min(max_results or self.max_results, MAX_SEARCH_RESULTS)

        # Check cache
        cache_key = f"{query}:{n}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))

            # Cache results
            self._set_cached(cache_key, results)
            log.debug("Web search: %d results for '%s'", len(results), query)
            return results

        except ImportError:
            log.error("duckduckgo-search not installed: pip install duckduckgo-search")
            return []
        except Exception as e:
            log.error("Web search error for '%s': %s", query, e)
            return []

    def search_formatted(self, query: str, max_results: int | None = None) -> str:
        """Search and return formatted string for LLM context injection."""
        results = self.search(query, max_results)
        if not results:
            return f"No web results found for: {query}"

        lines = [f"Web search results for: {query}\n"]
        for r in results:
            lines.append(f"  {r.get('title', '')}")
            lines.append(f"  {r.get('href', '')}")
            lines.append(f"  {r.get('body', '')}")
            lines.append("")

        return "\n".join(lines)

    def build_context(self, queries: list[str]) -> str:
        """Build a research context block from multiple search queries.

        Suitable for injecting into LLM prompts as background research.
        """
        if not queries:
            return ""

        sections = ["RECENT RESEARCH (web search):\n"]
        for query in queries[:MAX_BUILD_CONTEXT_QUERIES]:
            results = self.search(query)
            if results:
                sections.append(f"[{query}]")
                for r in results[:MAX_BUILD_CONTEXT_RESULTS]:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    href = r.get("href", "")
                    sections.append(f"  - {title} — {body} ({href})")
                sections.append("")

        return "\n".join(sections)

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.debug("Could not load web search cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._cache))
        except OSError as e:
            log.debug("Could not save web search cache: %s", e)

    def _get_cached(self, key: str) -> list[dict] | None:
        entry = self._cache.get(key)
        if entry and time.time() - entry.get("ts", 0) < self.cache_ttl:
            return entry.get("data")
        return None

    def _set_cached(self, key: str, data: list[dict]) -> None:
        # Evict oldest entries if cache is full
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].get("ts", 0))
            del self._cache[oldest_key]
        self._cache[key] = {"data": data, "ts": time.time()}
        self._save_cache()
