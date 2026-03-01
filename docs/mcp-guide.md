# MCP Guide

The comprehensive reference for Model Context Protocol (MCP) integration in ollama-forge.

---

## Table of Contents

- [What is MCP?](#what-is-mcp)
- [Built-in MCPs](#built-in-mcps)
- [MCP Directory](#mcp-directory)
  - [Search & Web](#search--web)
  - [Databases](#databases)
  - [Development Tools](#development-tools)
  - [Productivity](#productivity)
  - [File & Data](#file--data)
  - [AI & ML](#ai--ml)
  - [Communication](#communication)
  - [Cloud & Infrastructure](#cloud--infrastructure)
  - [Browser Automation](#browser-automation)
- [Managing MCPs](#managing-mcps)
- [Configuration](#configuration)
- [Creating Custom MCPs](#creating-custom-mcps)

---

## What is MCP?

**Model Context Protocol (MCP)** is an open standard that defines how AI models connect to external tools, data sources, and services. Think of it as a universal adapter: instead of building custom integrations for every tool, MCP provides a single protocol that any AI agent can use to interact with any MCP-compatible server.

**Key concepts:**

- **MCP Server** -- A process that exposes tools, resources, or prompts over a standardized JSON-RPC 2.0 transport (typically stdio or HTTP+SSE).
- **MCP Client** -- The AI agent (ollama-forge, in our case) that discovers and calls tools provided by MCP servers.
- **Tools** -- Functions that the AI can invoke. Each tool has a name, description, and a JSON Schema defining its parameters.
- **Resources** -- Read-only data the AI can access (files, database schemas, API docs).
- **Prompts** -- Reusable prompt templates provided by the server.

**How it works in ollama-forge:**

1. You enable an MCP server (e.g., `forge mcp add github`).
2. ollama-forge starts the server process and discovers its tools.
3. The agent sees the tools in its tool definitions and can call them during conversation.
4. Results flow back through the same protocol.

**Official specification:** [https://modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## Built-in MCPs

ollama-forge ships with two MCP servers that are enabled by default. No installation or API keys are required.

### web-search

Privacy-first web search powered by DuckDuckGo. Available out of the box.

```yaml
# Default configuration (already active)
web-search:
  enabled: true
  max_results: 5
  cache_ttl: 21600  # 6 hours
```

**Usage in chat:**
> "Search the web for the latest Ollama release notes"

**Tools provided:**
| Tool | Description |
|------|-------------|
| `web_search` | Search the web using DuckDuckGo |
| `web_search_news` | Search recent news articles |

### filesystem

Local file operations: read, write, search, list directories. Enabled by default with safety checks.

```yaml
# Default configuration (already active)
filesystem:
  enabled: true
  allowed_paths:
    - "."           # Current directory
    - "~"           # Home directory
```

**Tools provided:**
| Tool | Description |
|------|-------------|
| `read_file` | Read the contents of a file |
| `write_file` | Write content to a file |
| `edit_file` | Make targeted edits to a file |
| `list_directory` | List files and directories |
| `search_files` | Search for files by name pattern |
| `search_content` | Search file contents with regex |

---

## MCP Directory

A comprehensive directory of MCP servers organized by category. Each entry includes installation instructions, API key requirements, and configuration examples.

### Search & Web

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **web-search** | No | Built-in | DuckDuckGo search, no setup needed |
| **brave-search** | Yes | `npm install -g @anthropic/mcp-brave-search` | Brave Search API with rich snippets |
| **fetch** | No | `npm install -g @anthropic/mcp-fetch` | Extract content from web pages |
| **tavily** | Yes | `npm install -g tavily-mcp` | AI-powered search with answer synthesis |
| **exa** | Yes | `npm install -g exa-mcp-server` | Neural/semantic search engine |

#### web-search (built-in)

No installation needed. Uses DuckDuckGo for privacy-first search without API keys.

```yaml
web-search:
  enabled: true
  max_results: 5
  cache_ttl: 21600
```

#### brave-search

Brave Search API with support for web search, news, and local results. Requires a Brave Search API key from [https://brave.com/search/api/](https://brave.com/search/api/).

```bash
npm install -g @anthropic/mcp-brave-search
forge mcp add brave-search
```

```yaml
brave-search:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-brave-search"]
  env:
    BRAVE_API_KEY: ${BRAVE_API_KEY}
```

#### fetch

Extract readable content from any URL. Converts HTML to markdown for clean LLM consumption. No API key needed.

```bash
npm install -g @anthropic/mcp-fetch
forge mcp add fetch
```

```yaml
fetch:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-fetch"]
```

#### tavily

AI-powered search that returns synthesized answers alongside source links. Requires a Tavily API key from [https://tavily.com](https://tavily.com).

```bash
npm install -g tavily-mcp
forge mcp add tavily
```

```yaml
tavily:
  enabled: true
  command: npx
  args: ["-y", "tavily-mcp"]
  env:
    TAVILY_API_KEY: ${TAVILY_API_KEY}
```

#### exa

Neural search engine that understands meaning, not just keywords. Excellent for finding similar content or research papers. Requires an Exa API key from [https://exa.ai](https://exa.ai).

```bash
npm install -g exa-mcp-server
forge mcp add exa
```

```yaml
exa:
  enabled: true
  command: npx
  args: ["-y", "exa-mcp-server"]
  env:
    EXA_API_KEY: ${EXA_API_KEY}
```

---

### Databases

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **sqlite** | No | `npm install -g @anthropic/mcp-sqlite` | Query and manage local SQLite databases |
| **postgres** | No* | `npm install -g @anthropic/mcp-postgres` | Connect to PostgreSQL databases |
| **mysql** | No* | `npm install -g @benborla29/mcp-server-mysql` | Connect to MySQL databases |
| **redis** | No* | `npm install -g @punkpeye/mcp-server-redis` | Redis key-value store operations |
| **mongodb** | No* | `npm install -g mcp-mongo-server` | MongoDB document database operations |

*No API key, but requires database connection credentials.

#### sqlite

Query local SQLite databases. The agent can explore schemas, run queries, and analyze data. No credentials needed -- just point it at a `.db` file.

```bash
npm install -g @anthropic/mcp-sqlite
forge mcp add sqlite
```

```yaml
sqlite:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-sqlite", "--db-path", "/path/to/database.db"]
```

**Tools provided:** `read_query`, `write_query`, `create_table`, `list_tables`, `describe_table`, `append_insight`

#### postgres

Connect to PostgreSQL (including TimescaleDB). Supports read and write operations, schema inspection, and query execution.

```bash
npm install -g @anthropic/mcp-postgres
forge mcp add postgres
```

```yaml
postgres:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-postgres", "postgresql://user:password@localhost:5432/dbname"]
```

**Tools provided:** `query`, `list_tables`, `describe_table`

#### mysql

Connect to MySQL databases for queries and schema exploration.

```bash
npm install -g @benborla29/mcp-server-mysql
forge mcp add mysql
```

```yaml
mysql:
  enabled: true
  command: npx
  args: ["-y", "@benborla29/mcp-server-mysql"]
  env:
    MYSQL_HOST: localhost
    MYSQL_PORT: "3306"
    MYSQL_USER: ${MYSQL_USER}
    MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    MYSQL_DATABASE: ${MYSQL_DATABASE}
```

#### redis

Interact with Redis for key-value operations, pub/sub, and data structure manipulation.

```bash
npm install -g @punkpeye/mcp-server-redis
forge mcp add redis
```

```yaml
redis:
  enabled: true
  command: npx
  args: ["-y", "@punkpeye/mcp-server-redis"]
  env:
    REDIS_URL: "redis://localhost:6379"
```

#### mongodb

Connect to MongoDB for document operations, aggregation pipelines, and collection management.

```bash
npm install -g mcp-mongo-server
forge mcp add mongodb
```

```yaml
mongodb:
  enabled: true
  command: npx
  args: ["-y", "mcp-mongo-server"]
  env:
    MONGODB_URI: "mongodb://localhost:27017"
    MONGODB_DATABASE: ${MONGODB_DATABASE}
```

---

### Development Tools

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **github** | Yes | `npm install -g @anthropic/mcp-github` | GitHub repos, issues, PRs, actions |
| **gitlab** | Yes | `npm install -g @zereight/mcp-gitlab` | GitLab repos, issues, merge requests |
| **git** | No | `npm install -g @anthropic/mcp-git` | Local git operations (log, diff, blame) |
| **docker** | No | `npm install -g mcp-docker` | Docker container and image management |
| **kubernetes** | No | `npm install -g @anthropic/mcp-kubernetes` | Kubernetes cluster management |

#### github

Full GitHub integration: browse repos, manage issues and PRs, trigger workflows, search code. Requires a GitHub personal access token.

```bash
npm install -g @anthropic/mcp-github
forge mcp add github
```

```yaml
github:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-github"]
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}
```

**Tools provided:** `create_issue`, `list_issues`, `create_pull_request`, `search_repositories`, `get_file_contents`, `push_files`, `create_branch`, `list_commits`, and more.

#### gitlab

GitLab integration for managing repositories, issues, merge requests, and pipelines. Requires a GitLab personal access token.

```bash
npm install -g @zereight/mcp-gitlab
forge mcp add gitlab
```

```yaml
gitlab:
  enabled: true
  command: npx
  args: ["-y", "@zereight/mcp-gitlab"]
  env:
    GITLAB_TOKEN: ${GITLAB_TOKEN}
    GITLAB_URL: "https://gitlab.com"  # or your self-hosted instance
```

#### git

Local git operations without leaving the chat. Inspect history, diffs, branches, and blame information.

```bash
npm install -g @anthropic/mcp-git
forge mcp add git
```

```yaml
git:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-git"]
```

**Tools provided:** `git_log`, `git_diff`, `git_show`, `git_blame`, `git_branch`, `git_status`

#### docker

Manage Docker containers, images, volumes, and networks. Requires Docker to be installed and the Docker socket accessible.

```bash
npm install -g mcp-docker
forge mcp add docker
```

```yaml
docker:
  enabled: true
  command: npx
  args: ["-y", "mcp-docker"]
```

**Tools provided:** `list_containers`, `start_container`, `stop_container`, `container_logs`, `list_images`, `run_container`

#### kubernetes

Manage Kubernetes clusters: inspect pods, deployments, services, and logs. Uses your local kubeconfig.

```bash
npm install -g @anthropic/mcp-kubernetes
forge mcp add kubernetes
```

```yaml
kubernetes:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-kubernetes"]
```

**Tools provided:** `get_pods`, `get_deployments`, `get_services`, `get_logs`, `describe_resource`, `apply_manifest`

---

### Productivity

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **slack** | Yes | `npm install -g @anthropic/mcp-slack` | Read/send Slack messages, manage channels |
| **google-drive** | Yes | `npm install -g @anthropic/mcp-google-drive` | Access Google Drive files and folders |
| **notion** | Yes | `npm install -g notion-mcp-server` | Read and manage Notion workspace pages |
| **linear** | Yes | `npm install -g @anthropic/mcp-linear` | Linear issue tracking and project management |
| **todoist** | Yes | `npm install -g todoist-mcp-server` | Manage Todoist tasks and projects |

#### slack

Send and read Slack messages, list channels, search message history. Requires a Slack Bot Token with appropriate scopes.

```bash
npm install -g @anthropic/mcp-slack
forge mcp add slack
```

```yaml
slack:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-slack"]
  env:
    SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
```

**Tools provided:** `send_message`, `list_channels`, `get_channel_history`, `search_messages`, `get_thread_replies`

#### google-drive

Access files in Google Drive: list, read, and search documents. Requires Google OAuth credentials.

```bash
npm install -g @anthropic/mcp-google-drive
forge mcp add google-drive
```

```yaml
google-drive:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-google-drive"]
  env:
    GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID}
    GOOGLE_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET}
    GOOGLE_REFRESH_TOKEN: ${GOOGLE_REFRESH_TOKEN}
```

#### notion

Read and manage pages in a Notion workspace. Requires a Notion integration token.

```bash
npm install -g notion-mcp-server
forge mcp add notion
```

```yaml
notion:
  enabled: true
  command: npx
  args: ["-y", "notion-mcp-server"]
  env:
    NOTION_API_KEY: ${NOTION_API_KEY}
```

#### linear

Manage issues, projects, and cycles in Linear. Requires a Linear API key.

```bash
npm install -g @anthropic/mcp-linear
forge mcp add linear
```

```yaml
linear:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-linear"]
  env:
    LINEAR_API_KEY: ${LINEAR_API_KEY}
```

#### todoist

Create, update, and manage tasks and projects in Todoist. Requires a Todoist API token.

```bash
npm install -g todoist-mcp-server
forge mcp add todoist
```

```yaml
todoist:
  enabled: true
  command: npx
  args: ["-y", "todoist-mcp-server"]
  env:
    TODOIST_API_TOKEN: ${TODOIST_API_TOKEN}
```

---

### File & Data

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **filesystem** | No | Built-in | File read/write/search (built-in) |
| **memory** | No | `npm install -g @anthropic/mcp-memory` | Persistent knowledge graph across sessions |
| **pandoc** | No | `npm install -g mcp-pandoc` | Convert documents between formats |
| **markdown** | No | `npm install -g @anthropic/mcp-markdown` | Advanced markdown processing and rendering |

#### filesystem (built-in)

See [Built-in MCPs](#built-in-mcps) above. No installation needed.

#### memory

Persistent knowledge graph that retains information across chat sessions. The agent can store entities, relationships, and observations that survive restarts.

```bash
npm install -g @anthropic/mcp-memory
forge mcp add memory
```

```yaml
memory:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-memory"]
```

**Tools provided:** `create_entities`, `create_relations`, `add_observations`, `search_nodes`, `open_nodes`, `delete_entities`, `delete_relations`

#### pandoc

Convert documents between formats (Markdown, HTML, PDF, DOCX, LaTeX, and more). Requires [Pandoc](https://pandoc.org/) installed on the system.

```bash
npm install -g mcp-pandoc
forge mcp add pandoc
```

```yaml
pandoc:
  enabled: true
  command: npx
  args: ["-y", "mcp-pandoc"]
```

**Tools provided:** `convert_document`, `list_formats`

#### markdown

Advanced markdown processing: parse, transform, extract sections, generate table of contents.

```bash
npm install -g @anthropic/mcp-markdown
forge mcp add markdown
```

```yaml
markdown:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-markdown"]
```

---

### AI & ML

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **openai** | Yes | `npm install -g openai-mcp-server` | Call OpenAI models (GPT, DALL-E, embeddings) |
| **huggingface** | Yes | `npm install -g huggingface-mcp-server` | Access HuggingFace models and datasets |

#### openai

Integrate OpenAI API calls into your agent workflow. Useful for tasks where a cloud model complements the local Ollama model (e.g., image generation with DALL-E, embeddings). Requires an OpenAI API key.

```bash
npm install -g openai-mcp-server
forge mcp add openai
```

```yaml
openai:
  enabled: true
  command: npx
  args: ["-y", "openai-mcp-server"]
  env:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
```

#### huggingface

Browse and interact with HuggingFace models, datasets, and spaces. Requires a HuggingFace access token for private resources.

```bash
npm install -g huggingface-mcp-server
forge mcp add huggingface
```

```yaml
huggingface:
  enabled: true
  command: npx
  args: ["-y", "huggingface-mcp-server"]
  env:
    HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN}
```

---

### Communication

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **email** | Yes* | `npm install -g email-mcp-server` | Read and send email via IMAP/SMTP |
| **discord** | Yes | `npm install -g discord-mcp-server` | Discord bot: read/send messages, manage servers |

*Requires IMAP/SMTP credentials (email account login).

#### email

Read, search, and send emails via IMAP and SMTP. Works with Gmail, Outlook, and any IMAP-compatible provider.

```bash
npm install -g email-mcp-server
forge mcp add email
```

```yaml
email:
  enabled: true
  command: npx
  args: ["-y", "email-mcp-server"]
  env:
    IMAP_HOST: "imap.gmail.com"
    IMAP_PORT: "993"
    SMTP_HOST: "smtp.gmail.com"
    SMTP_PORT: "587"
    EMAIL_USER: ${EMAIL_USER}
    EMAIL_PASSWORD: ${EMAIL_PASSWORD}
```

**Tools provided:** `read_inbox`, `search_emails`, `send_email`, `read_email`

#### discord

Interact with Discord servers: read messages, send messages, manage channels. Requires a Discord bot token.

```bash
npm install -g discord-mcp-server
forge mcp add discord
```

```yaml
discord:
  enabled: true
  command: npx
  args: ["-y", "discord-mcp-server"]
  env:
    DISCORD_BOT_TOKEN: ${DISCORD_BOT_TOKEN}
```

---

### Cloud & Infrastructure

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **aws** | Yes | `npm install -g aws-mcp-server` | AWS services (S3, EC2, Lambda, etc.) |
| **cloudflare** | Yes | `npm install -g @anthropic/mcp-cloudflare` | Cloudflare DNS, Workers, Pages management |
| **vercel** | Yes | `npm install -g vercel-mcp-server` | Vercel project deployments and management |

#### aws

Interact with AWS services: S3 buckets, EC2 instances, Lambda functions, CloudWatch logs, and more. Uses standard AWS credentials.

```bash
npm install -g aws-mcp-server
forge mcp add aws
```

```yaml
aws:
  enabled: true
  command: npx
  args: ["-y", "aws-mcp-server"]
  env:
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    AWS_REGION: "us-east-1"
```

#### cloudflare

Manage Cloudflare resources: DNS records, Workers, Pages, R2 storage. Requires a Cloudflare API token.

```bash
npm install -g @anthropic/mcp-cloudflare
forge mcp add cloudflare
```

```yaml
cloudflare:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-cloudflare"]
  env:
    CLOUDFLARE_API_TOKEN: ${CLOUDFLARE_API_TOKEN}
    CLOUDFLARE_ACCOUNT_ID: ${CLOUDFLARE_ACCOUNT_ID}
```

#### vercel

Manage Vercel projects, deployments, and environment variables. Requires a Vercel access token.

```bash
npm install -g vercel-mcp-server
forge mcp add vercel
```

```yaml
vercel:
  enabled: true
  command: npx
  args: ["-y", "vercel-mcp-server"]
  env:
    VERCEL_TOKEN: ${VERCEL_TOKEN}
```

---

### Browser Automation

| MCP | API Key? | Install | Description |
|-----|----------|---------|-------------|
| **puppeteer** | No | `npm install -g @anthropic/mcp-puppeteer` | Browser automation, screenshots, scraping |
| **playwright** | No | `npm install -g @anthropic/mcp-playwright` | Advanced browser automation with multi-browser support |

#### puppeteer

Automate browser interactions: navigate pages, fill forms, take screenshots, extract data. Uses headless Chromium.

```bash
npm install -g @anthropic/mcp-puppeteer
forge mcp add puppeteer
```

```yaml
puppeteer:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-puppeteer"]
```

**Tools provided:** `navigate`, `screenshot`, `click`, `type`, `evaluate`, `get_content`

#### playwright

Advanced browser automation supporting Chromium, Firefox, and WebKit. More robust than Puppeteer for complex automation scenarios.

```bash
npm install -g @anthropic/mcp-playwright
forge mcp add playwright
```

```yaml
playwright:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-playwright"]
```

**Tools provided:** `navigate`, `screenshot`, `click`, `fill`, `select`, `evaluate`, `get_content`, `pdf`

---

## Managing MCPs

### CLI Commands

```bash
# List all MCPs and their status (enabled/disabled)
forge mcp list

# Add (enable) an MCP server
forge mcp add github

# Remove (disable) an MCP server
forge mcp remove github

# Search the MCP registry by keyword
forge mcp search database
forge mcp search sql
forge mcp search browser
```

### Natural Language (in chat)

You can manage MCPs directly from the chat interface using natural language:

> "Add a PostgreSQL MCP so I can query my database"

> "I need to search GitHub issues"

> "Remove the Slack integration"

> "What MCPs are available for databases?"

The agent understands these requests and runs the appropriate `forge mcp` commands.

### Smart Suggestions

ollama-forge can detect when an MCP would be helpful based on your conversation:

> **You:** "I need to analyze data in my SQLite database at /tmp/analytics.db"
>
> **Agent:** "I see you want to work with a SQLite database. The `sqlite` MCP can help. Want me to enable it? Run: `forge mcp add sqlite`"

> **You:** "Can you check if there are any open issues on our GitHub repo?"
>
> **Agent:** "You'll need the `github` MCP for that. Run: `forge mcp add github` and set your `GITHUB_TOKEN` environment variable."

---

## Configuration

### Config File Location

MCP configuration is stored in:

```
~/.config/ollama-forge/mcp.yaml
```

This file is created automatically when you first run `forge mcp add`. You can also edit it manually.

### File Format

The configuration file uses YAML with one top-level key per MCP server:

```yaml
# MCP server name
server-name:
  # Whether this server is active
  enabled: true

  # Command to start the MCP server process
  command: npx
  args: ["-y", "@namespace/mcp-server-name"]

  # Environment variables passed to the server process
  env:
    API_KEY: ${API_KEY}          # Reference to shell environment variable
    STATIC_VALUE: "some-value"   # Hardcoded value

  # Server-specific settings (varies by MCP)
  max_results: 10
  cache_ttl: 3600
```

### Full Example

A realistic configuration with multiple MCPs enabled:

```yaml
# Built-in: web search (enabled by default)
web-search:
  enabled: true
  max_results: 5
  cache_ttl: 21600

# Built-in: filesystem (enabled by default)
filesystem:
  enabled: true
  allowed_paths:
    - "."
    - "~"

# GitHub integration
github:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-github"]
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}

# Local SQLite databases
sqlite:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-sqlite", "--db-path", "~/data/app.db"]

# PostgreSQL (TimescaleDB on custom port)
postgres:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-postgres", "postgresql://user:pass@localhost:5433/trading"]

# Persistent memory across sessions
memory:
  enabled: true
  command: npx
  args: ["-y", "@anthropic/mcp-memory"]

# Browser automation
puppeteer:
  enabled: false  # Available but disabled
  command: npx
  args: ["-y", "@anthropic/mcp-puppeteer"]
```

### Environment Variables

Secrets should **never** be hardcoded in `mcp.yaml`. Use `${VAR_NAME}` references that resolve from your shell environment or a `.env` file:

```bash
# In your ~/.bashrc or .env file:
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export BRAVE_API_KEY="BSAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```yaml
# In mcp.yaml -- references resolve at server startup:
github:
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}
```

---

## Creating Custom MCPs

You can build your own MCP server to expose any tool or data source to ollama-forge agents. MCP servers communicate over **JSON-RPC 2.0** using **stdio** (stdin/stdout) as the default transport.

### Prerequisites

- Python 3.10+ (or Node.js 18+ for TypeScript servers)
- The `mcp` Python package: `pip install mcp`

### Step 1: Project Structure

```
my-mcp-server/
  my_mcp_server/
    __init__.py
    server.py
  pyproject.toml
```

### Step 2: Define Your Server

Create `my_mcp_server/server.py`:

```python
"""A custom MCP server that provides weather data."""

import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)


# Create the MCP server instance
server = Server("weather-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Define the tools this server provides."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'San Francisco')",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="get_forecast",
            description="Get 5-day weather forecast for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days (1-5)",
                        "default": 3,
                    },
                },
                "required": ["city"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle incoming tool calls."""
    if name == "get_weather":
        city = arguments["city"]
        units = arguments.get("units", "celsius")
        # In a real server, call a weather API here
        weather_data = {
            "city": city,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "condition": "partly cloudy",
            "humidity": 65,
            "wind_speed": "12 km/h",
        }
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(weather_data, indent=2),
                )
            ]
        )

    elif name == "get_forecast":
        city = arguments["city"]
        days = arguments.get("days", 3)
        forecast = [
            {"day": i + 1, "high": 20 + i, "low": 12 + i, "condition": "sunny"}
            for i in range(days)
        ]
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"city": city, "forecast": forecast}, indent=2),
                )
            ]
        )

    else:
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"Unknown tool: {name}")
            ],
            isError=True,
        )


async def main():
    """Run the server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Step 3: Package Configuration

Create `pyproject.toml`:

```toml
[project]
name = "my-mcp-weather"
version = "0.1.0"
description = "MCP server providing weather data"
requires-python = ">=3.10"
dependencies = ["mcp>=1.0.0"]

[project.scripts]
my-mcp-weather = "my_mcp_server.server:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### Step 4: Test Locally

```bash
# Install in development mode
pip install -e .

# Test by running the server directly (it communicates over stdio)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m my_mcp_server.server
```

### Step 5: Register with ollama-forge

Add your custom server to `~/.config/ollama-forge/mcp.yaml`:

```yaml
weather:
  enabled: true
  command: python
  args: ["-m", "my_mcp_server.server"]
```

Or if you published it to PyPI/npm:

```yaml
weather:
  enabled: true
  command: uvx
  args: ["my-mcp-weather"]
```

### Step 6: Publish to the MCP Registry

To make your server available to the broader community:

1. **Publish to PyPI** (Python) or **npm** (Node.js).
2. **Submit to the MCP registry** at [https://modelcontextprotocol.io](https://modelcontextprotocol.io) by opening a pull request to the official servers list.
3. **Include a README** with:
   - What tools the server provides
   - Required environment variables
   - Example configuration
   - Any system dependencies

### MCP Protocol Reference

Key JSON-RPC 2.0 methods that an MCP server must implement:

| Method | Direction | Description |
|--------|-----------|-------------|
| `initialize` | Client -> Server | Handshake with protocol version and capabilities |
| `tools/list` | Client -> Server | List all available tools |
| `tools/call` | Client -> Server | Execute a tool with given arguments |
| `resources/list` | Client -> Server | List available resources (optional) |
| `resources/read` | Client -> Server | Read a resource by URI (optional) |
| `prompts/list` | Client -> Server | List available prompts (optional) |
| `prompts/get` | Client -> Server | Get a prompt by name (optional) |
| `notifications/tools/list_changed` | Server -> Client | Notify client that tools changed (optional) |

**Transport options:**
- **stdio** (default) -- Server reads JSON-RPC from stdin, writes to stdout. Simplest to implement. ollama-forge manages the process lifecycle.
- **HTTP + SSE** -- Server listens on an HTTP port. Client connects via Server-Sent Events for server-to-client messages and POST for client-to-server messages. Useful for remote or shared servers.

Full protocol specification: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## Quick Reference

### Common Workflows

**Add an MCP and configure it:**
```bash
forge mcp add github
# Edit ~/.config/ollama-forge/mcp.yaml to add your GITHUB_TOKEN
forge mcp list  # Verify it shows as enabled
```

**Search for an MCP by keyword:**
```bash
forge mcp search database    # Shows sqlite, postgres, mysql, mongodb, redis
forge mcp search browser     # Shows puppeteer, playwright
forge mcp search search      # Shows web-search, brave-search, tavily, exa
```

**Temporarily disable without removing config:**
```yaml
# In mcp.yaml, set enabled to false:
github:
  enabled: false  # Disabled but config preserved
  command: npx
  args: ["-y", "@anthropic/mcp-github"]
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}
```

### MCP Summary Table

| MCP | Category | API Key? | Built-in? |
|-----|----------|----------|-----------|
| web-search | Search & Web | No | Yes |
| brave-search | Search & Web | Yes | No |
| fetch | Search & Web | No | No |
| tavily | Search & Web | Yes | No |
| exa | Search & Web | Yes | No |
| sqlite | Databases | No | No |
| postgres | Databases | No* | No |
| mysql | Databases | No* | No |
| redis | Databases | No* | No |
| mongodb | Databases | No* | No |
| github | Development | Yes | No |
| gitlab | Development | Yes | No |
| git | Development | No | No |
| docker | Development | No | No |
| kubernetes | Development | No | No |
| slack | Productivity | Yes | No |
| google-drive | Productivity | Yes | No |
| notion | Productivity | Yes | No |
| linear | Productivity | Yes | No |
| todoist | Productivity | Yes | No |
| filesystem | File & Data | No | Yes |
| memory | File & Data | No | No |
| pandoc | File & Data | No | No |
| markdown | File & Data | No | No |
| openai | AI & ML | Yes | No |
| huggingface | AI & ML | Yes | No |
| email | Communication | Yes* | No |
| discord | Communication | Yes | No |
| aws | Cloud & Infra | Yes | No |
| cloudflare | Cloud & Infra | Yes | No |
| vercel | Cloud & Infra | Yes | No |
| puppeteer | Browser | No | No |
| playwright | Browser | No | No |

*Database MCPs require connection credentials; email requires IMAP/SMTP credentials.
