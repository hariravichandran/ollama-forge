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


class TestMCPManager:
    """Tests for the MCP manager lifecycle."""

    def test_init_creates_web_search_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            assert "web-search" in mgr._config
            assert mgr._config["web-search"]["enabled"] is True

    def test_list_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            available = mgr.list_available()
            assert len(available) >= 5
            names = [a["name"] for a in available]
            assert "web-search" in names

    def test_web_search_enabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            enabled = mgr.get_enabled()
            assert "web-search" in enabled

    def test_disable_mcp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            result = mgr.disable("web-search")
            assert "disabled" in result.lower()
            assert "web-search" not in mgr.get_enabled()

    def test_enable_unknown_mcp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            result = mgr.enable("totally-fake-mcp")
            assert "Unknown" in result

    def test_disable_not_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            result = mgr.disable("github")
            assert "was not enabled" in result

    def test_get_tools_for_agent_web_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            tools = mgr.get_tools_for_agent()
            assert len(tools) >= 1
            tool_names = [t["function"]["name"] for t in tools]
            assert "web_search" in tool_names

    def test_get_tools_empty_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            mgr.disable("web-search")
            tools = mgr.get_tools_for_agent()
            assert len(tools) == 0

    def test_config_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            mgr.disable("web-search")

            # Reload
            mgr2 = MCPManager(config_path=str(config_path))
            assert "web-search" not in mgr2.get_enabled()

    def test_list_available_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.yaml"
            from forge.mcp.manager import MCPManager
            mgr = MCPManager(config_path=str(config_path))
            available = mgr.list_available()
            ws = [a for a in available if a["name"] == "web-search"][0]
            assert ws["status"] == "enabled"
            assert ws["builtin"] is True


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
