# Architecture Overview

## System Layers

ollama-forge is organized in layers, from user-facing interfaces down to hardware:

```
User Interfaces (CLI, TUI, Web UI)
        │
Agent Orchestrator + Tracker
        │
Tools Layer (filesystem, shell, git, web, MCP)
        │
LLM Layer (Ollama client, context compression, model management)
        │
Hardware Detection (GPU, CPU, RAM, profiles, ROCm config)
```

## Package Structure

```
forge/
├── cli.py              # Click-based CLI — all user commands
├── config.py           # YAML + env config management
│
├── hardware/           # Hardware detection
│   ├── detect.py       # GPU/CPU/RAM detection via sysfs, /proc
│   ├── profiles.py     # Hardware → model/config mapping
│   └── rocm.py         # AMD ROCm-specific optimization
│
├── llm/                # LLM interface
│   ├── client.py       # OllamaClient — generate, chat, stream, tools
│   ├── models.py       # Model catalogue, size estimation, recommendations
│   └── context.py      # Context compression (sliding summary, progressive)
│
├── tools/              # Agent tools
│   ├── filesystem.py   # File read/write/edit/search
│   ├── shell.py        # Command execution with safety checks
│   ├── git.py          # Git operations
│   └── web.py          # Web search and fetch
│
├── mcp/                # Model Context Protocol
│   ├── manager.py      # MCP server lifecycle
│   ├── registry.py     # Known MCP server catalogue
│   ├── web_search.py   # Built-in DuckDuckGo search (default on)
│   └── natural_language.py  # NL-based MCP management
│
├── agents/             # Agent framework
│   ├── base.py         # BaseAgent — chat loop, tool dispatch, memory
│   ├── orchestrator.py # Multi-agent coordination
│   ├── tracker.py      # Agent system tracking (single/multi)
│   ├── coder.py        # Coding agent preset
│   └── researcher.py   # Research agent preset
│
├── community/          # Crowdsourced improvement
│   ├── ideas.py        # Anonymous idea collection
│   └── self_improve.py # Self-improvement agent (evaluate, implement, test)
│
├── ui/                 # User interfaces
│   ├── terminal.py     # Textual TUI (Phase 2)
│   └── web/app.py      # FastAPI Web UI (Phase 2)
│
└── utils/              # Shared utilities
    ├── env.py          # .env file loading
    └── logging.py      # Rich-based structured logging
```

## Key Design Decisions

### 1. Local-First, No Cloud Dependencies

Everything runs on your machine. No API keys required for core functionality. Web search uses DuckDuckGo (no API key needed).

### 2. Hardware-Aware Model Selection

Instead of hardcoding model names, the system detects available GPU memory and selects appropriate models. This means the same code runs on a laptop with 8 GB iGPU and a workstation with 48 GB dGPU.

### 3. Context Compression

Local models have limited context windows (4K-32K tokens typically). Rather than silently truncating conversations, ollama-forge uses LLM-powered summarization to compress older messages while preserving key information.

### 4. Tool Calling via Ollama API

Agents use Ollama's native tool calling API. Each tool provides `get_tool_definitions()` that returns Ollama-compatible function definitions. The agent loops on tool calls until a final text response is produced.

### 5. YAML-Based Agent Definitions

Custom agents are defined in simple YAML files. No Python code needed — just specify a name, system prompt, tools, and model. This makes it accessible to users who aren't developers.

### 6. Community Self-Improvement

The self-improvement agent is part of the codebase itself. It reads community ideas, searches for latest advances, and implements improvements — all using the same local LLM. Changes go to `main` first, get tested, then promote to `stable` (max 1 per 2 days).

## Branch Policy

- **`main`**: Active development. Self-improvement agent commits here. Multiple contributors can push concurrently (pull-rebase-push pattern).
- **`stable`**: Tested improvements only. Max 1 auto-push per 2 days. Users who want stability should track this branch.

## Data Flow: Chat Message

```
User types message
    │
    ▼
CLI (forge/cli.py) receives input
    │
    ▼
AgentOrchestrator routes to active agent
    │
    ▼
BaseAgent.chat():
    ├── Add message to history
    ├── ContextCompressor.compress() if needed
    ├── Build messages + system prompt
    ├── OllamaClient.chat() with tool definitions
    │       │
    │       ▼
    │   Ollama generates response
    │       │
    │       ├── If tool_calls: execute tools, loop back to chat()
    │       └── If text: return final response
    │
    ▼
CLI displays response to user
```
