"""Registry of known MCP servers — built-in and community."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCPEntry:
    """An MCP server definition."""

    name: str
    description: str
    category: str  # "search", "data", "tools", "productivity", "development"
    builtin: bool  # ships with ollama-forge
    install_cmd: str  # how to install (pip, npm, etc.)
    package: str  # package name
    config_example: dict  # example configuration


# Built-in and well-known MCP servers
MCP_REGISTRY: dict[str, MCPEntry] = {
    "web-search": MCPEntry(
        name="web-search",
        description="Search the web using DuckDuckGo (no API key required)",
        category="search",
        builtin=True,
        install_cmd="",  # built-in, no install needed
        package="",
        config_example={"enabled": True, "max_results": 5, "cache_ttl": 21600},
    ),
    "filesystem": MCPEntry(
        name="filesystem",
        description="Read, write, and manage files on disk",
        category="tools",
        builtin=True,
        install_cmd="",
        package="",
        config_example={"enabled": True, "root_dir": "."},
    ),
    "github": MCPEntry(
        name="github",
        description="Interact with GitHub repos, issues, PRs, and actions",
        category="development",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-github",
        package="@modelcontextprotocol/server-github",
        config_example={"token_env": "GITHUB_TOKEN"},
    ),
    "brave-search": MCPEntry(
        name="brave-search",
        description="Web search using the Brave Search API",
        category="search",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-brave-search",
        package="@modelcontextprotocol/server-brave-search",
        config_example={"api_key_env": "BRAVE_API_KEY"},
    ),
    "sqlite": MCPEntry(
        name="sqlite",
        description="Query and manage SQLite databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-sqlite",
        package="@modelcontextprotocol/server-sqlite",
        config_example={"db_path": "example.db"},
    ),
    "puppeteer": MCPEntry(
        name="puppeteer",
        description="Browser automation — navigate pages, take screenshots, fill forms",
        category="tools",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-puppeteer",
        package="@modelcontextprotocol/server-puppeteer",
        config_example={},
    ),
    "memory": MCPEntry(
        name="memory",
        description="Persistent knowledge graph for long-term memory",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-memory",
        package="@modelcontextprotocol/server-memory",
        config_example={"storage_path": "~/.config/ollama-forge/memory"},
    ),
    "postgres": MCPEntry(
        name="postgres",
        description="Query and manage PostgreSQL databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-postgres",
        package="@modelcontextprotocol/server-postgres",
        config_example={"connection_string_env": "DATABASE_URL"},
    ),
    "slack": MCPEntry(
        name="slack",
        description="Send and read Slack messages",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-slack",
        package="@modelcontextprotocol/server-slack",
        config_example={"token_env": "SLACK_BOT_TOKEN"},
    ),
    "fetch": MCPEntry(
        name="fetch",
        description="Fetch and extract content from web pages",
        category="tools",
        builtin=False,
        install_cmd="pip install mcp-server-fetch",
        package="mcp-server-fetch",
        config_example={},
    ),
}


def search_registry(query: str) -> list[MCPEntry]:
    """Search the MCP registry by name, description, or category."""
    query_lower = query.lower()
    results = []
    for entry in MCP_REGISTRY.values():
        if (query_lower in entry.name.lower()
                or query_lower in entry.description.lower()
                or query_lower in entry.category.lower()):
            results.append(entry)
    return results


def suggest_mcps(context: str) -> list[MCPEntry]:
    """Suggest MCPs based on conversation context.

    Simple keyword matching — can be enhanced with LLM-based suggestions.
    """
    context_lower = context.lower()

    suggestions = []
    keyword_map = {
        "github": ["github", "repo", "pull request", "issue", "PR"],
        "sqlite": ["sqlite", "database", "sql", "query"],
        "postgres": ["postgres", "postgresql", "database"],
        "puppeteer": ["browser", "screenshot", "web page", "scrape"],
        "brave-search": ["search", "find online", "look up"],
        "slack": ["slack", "message", "channel"],
        "memory": ["remember", "recall", "knowledge"],
        "fetch": ["fetch", "download", "web page", "url"],
    }

    for mcp_name, keywords in keyword_map.items():
        if any(kw in context_lower for kw in keywords):
            entry = MCP_REGISTRY.get(mcp_name)
            if entry:
                suggestions.append(entry)

    return suggestions
