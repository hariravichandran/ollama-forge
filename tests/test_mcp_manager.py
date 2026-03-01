"""Tests for MCP manager and registry."""

import pytest
import tempfile
from pathlib import Path

from forge.mcp.registry import MCP_REGISTRY, search_registry, suggest_mcps
from forge.mcp.web_search import WebSearchMCP
from forge.mcp.natural_language import parse_mcp_request


class TestMCPRegistry:
    """Tests for the MCP registry."""

    def test_registry_has_entries(self):
        assert len(MCP_REGISTRY) >= 5

    def test_web_search_is_builtin(self):
        ws = MCP_REGISTRY.get("web-search")
        assert ws is not None
        assert ws.builtin is True

    def test_search_by_name(self):
        results = search_registry("github")
        assert len(results) >= 1
        assert any(r.name == "github" for r in results)

    def test_search_by_category(self):
        results = search_registry("search")
        assert len(results) >= 1

    def test_search_no_results(self):
        results = search_registry("xyznonexistent")
        assert len(results) == 0

    def test_suggest_database(self):
        suggestions = suggest_mcps("I need to query a database")
        names = [s.name for s in suggestions]
        assert "sqlite" in names or "postgres" in names

    def test_suggest_browser(self):
        suggestions = suggest_mcps("I need to take a screenshot of a web page")
        names = [s.name for s in suggestions]
        assert "puppeteer" in names


class TestNaturalLanguage:
    """Tests for natural language MCP request parsing."""

    def test_parse_add_request(self):
        result = parse_mcp_request("add the GitHub MCP")
        assert result["action"] == "add"
        assert result["mcp_name"] == "github"

    def test_parse_remove_request(self):
        result = parse_mcp_request("remove the slack integration")
        assert result["action"] == "remove"
        assert result["mcp_name"] == "slack"

    def test_parse_list_request(self):
        result = parse_mcp_request("show me all available MCPs")
        assert result["action"] == "list"

    def test_parse_search_request(self):
        result = parse_mcp_request("search for database tools")
        assert result["action"] == "search"

    def test_parse_fuzzy_match(self):
        result = parse_mcp_request("I need to search the web")
        assert result["action"] == "add"
        assert result["mcp_name"] == "web-search"

    def test_parse_unrecognized(self):
        result = parse_mcp_request("hello how are you")
        assert result["action"] is None


class TestWebSearchMCP:
    """Tests for the built-in web search MCP."""

    def test_init_enabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            assert mcp.enabled is True

    def test_disabled_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            mcp.enabled = False
            results = mcp.search("test")
            assert results == []

    def test_formatted_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            mcp.enabled = False
            result = mcp.search_formatted("test")
            # Should handle gracefully even if disabled
            assert result == "" or "No" in result
