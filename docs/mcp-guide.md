# MCP Guide

## What is MCP?

MCP (Model Context Protocol) is a standard for connecting AI models to external tools and data sources. ollama-forge integrates MCP to give your agents superpowers.

## Web Search (Enabled by Default)

Web search works out of the box — no API keys needed. It uses DuckDuckGo for privacy-first search.

Just ask in chat:
> "Search the web for the latest Python 3.13 features"

Or use the CLI:
```bash
forge mcp list    # Shows web-search as enabled
```

## Available MCPs

| MCP | Category | API Key? | Description |
|-----|----------|----------|-------------|
| **web-search** | search | No | DuckDuckGo web search (built-in) |
| **filesystem** | tools | No | File operations (built-in) |
| github | development | Yes | GitHub repos, issues, PRs |
| brave-search | search | Yes | Brave Search API |
| sqlite | data | No | SQLite database queries |
| postgres | data | Yes | PostgreSQL database queries |
| puppeteer | tools | No | Browser automation |
| memory | productivity | No | Persistent knowledge graph |
| slack | productivity | Yes | Slack messaging |
| fetch | tools | No | Web page content extraction |

## Managing MCPs

### CLI
```bash
forge mcp list          # List all MCPs and their status
forge mcp add github    # Enable an MCP
forge mcp remove github # Disable an MCP
forge mcp search sql    # Search the registry
```

### Natural Language (in chat)
> "Add a PostgreSQL MCP"
> "I need to query a SQLite database"
> "Remove the GitHub integration"

### Suggesting MCPs

The agent can suggest relevant MCPs based on your conversation:
> "I'm working on a project that uses a PostgreSQL database"
> Agent: "You might want to enable the postgres MCP. Run: forge mcp add postgres"

## Configuration

MCP configuration is stored in `~/.config/ollama-forge/mcp.yaml`:

```yaml
web-search:
  enabled: true
  max_results: 5
  cache_ttl: 21600

github:
  enabled: true
  token_env: GITHUB_TOKEN
```

## Adding Custom MCPs

MCP servers that follow the Model Context Protocol can be added to the registry in `forge/mcp/registry.py`. Community contributions welcome!
