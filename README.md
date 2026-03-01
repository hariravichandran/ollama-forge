# ollama-forge

**Batteries-included local AI framework for Ollama.**

A free, open-source alternative to paid AI coding assistants. Run powerful AI agents locally on your machine — no cloud APIs, no subscriptions, no data leaving your computer.

## What is ollama-forge?

ollama-forge turns your machine into a full AI development environment:

- **Hardware auto-detection** — Detects your GPU (AMD ROCm, NVIDIA CUDA, Apple Silicon, Intel) and optimizes Ollama automatically
- **CPU-only support** — No GPU? No problem. Uses RAM-based model sizing with optimized thread counts
- **Cross-platform** — Linux, macOS, and Windows support
- **Context compression** — Intelligent conversation summarization so you never hit context limits
- **Multi-agent framework** — Create, manage, and orchestrate single-agent and multi-agent systems
- **Cascading agents** — Auto-switch to a larger model when the smaller one gets stuck
- **Codebase indexing** — Indexes symbols across Python, JS/TS, Go, Rust for fast search and agent context
- **Fuzzy edit matching** — Tolerates whitespace/indentation mistakes from local LLMs with 75%+ similarity fallback
- **Sandboxed execution** — Run code in isolated environments with resource limits and timeouts
- **Multi-file edit planner** — Dependency-aware cross-file refactoring with atomic rollback
- **Session persistence** — Save, resume, and export chat sessions
- **FIM tab completion** — OpenAI-compatible `/v1/completions` endpoint for editor integrations
- **Multi-modal images** — Pass images to vision-capable models via CLI or API
- **Git undo** — Safely revert any `[forge]`-tagged agent commit
- **Permission system** — Configurable approval levels for tool actions (auto-approve reads, confirm writes, always confirm shell)
- **MCP integration** — Web search enabled by default, 30+ MCPs available. See the [MCP Guide](docs/mcp-guide.md)
- **QA agent** — Auto-generates and runs tests for code changes before they ship
- **Self-improvement** — The framework improves itself using community ideas and latest AI research
- **OpenAI-compatible API** — Drop-in REST API for Continue.dev and other OpenAI clients
- **Terminal UI** — Rich Textual-based TUI with model picker, agent switching, keyboard shortcuts
- **Web UI** — Browser-based chat with dark theme, streaming responses, hardware status panel
- **Docker support** — `docker compose up` for instant setup with Ollama included

Optimized for **AMD iGPU + ROCm** (our primary target), but works with NVIDIA GPUs, Apple Silicon, and CPU-only setups too.

## Quick Start

### Option A: Install with Claude (Recommended)

If you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, just open the repo and Claude will guide you through setup — detecting your hardware, recommending models, and configuring everything automatically:

```bash
git clone https://github.com/hariravichandran/ollama-forge.git
cd ollama-forge
claude
```

Claude reads the project's `CLAUDE.md` and walks you through each step interactively.

### Option B: Script Install

```bash
git clone https://github.com/hariravichandran/ollama-forge.git
cd ollama-forge
bash install.sh
```

### Option C: Docker

```bash
git clone https://github.com/hariravichandran/ollama-forge.git
cd ollama-forge
docker compose up
# Web UI at http://localhost:8080
```

Uncomment the GPU lines in `docker-compose.yml` for AMD ROCm or NVIDIA GPU acceleration.

### Option D: Manual Setup

```bash
git clone https://github.com/hariravichandran/ollama-forge.git
cd ollama-forge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start chatting
forge chat
```

## Features

### Hardware-Aware

```bash
forge hardware          # See your detected hardware and profile
forge models recommend  # Get model recommendations for your GPU
forge benchmark         # Benchmark inference speed
```

ollama-forge detects your GPU, CPU, and RAM, then selects optimal model sizes, context windows, and batch sizes automatically. Supports AMD ROCm, NVIDIA CUDA, Apple Silicon (Metal), and CPU-only.

**CPU-only systems** are first-class citizens — when no GPU is detected, ollama-forge uses available RAM for model sizing, optimizes thread counts (leaving cores for the OS), and recommends appropriately-sized models. Slower than GPU but fully functional.

### Built-in Agents

| Agent | Description | Tools |
|-------|-------------|-------|
| **assistant** | General-purpose chat with all tools | filesystem, shell, git, web |
| **coder** | Code editing, debugging, project management | filesystem, shell, git |
| **researcher** | Web search, summarization, citations | web, filesystem |

```bash
forge chat                    # Chat with default assistant
forge chat --agent coder      # Chat with the coding agent
forge chat --agent researcher # Chat with the research agent
```

### Agent Templates

Pre-built templates for common use cases — just copy and customize:

| Template | Temperature | Best For |
|----------|-------------|----------|
| **writer** | 0.7 | Articles, blog posts, technical writing |
| **data_analyst** | 0.3 | Data exploration, SQL, visualizations |
| **devops** | 0.3 | Infrastructure, Docker, CI/CD, monitoring |
| **tutor** | 0.6 | Learning, explanations, guided exercises |
| **creative** | 0.9 | Brainstorming, stories, creative projects |
| **sysadmin** | 0.2 | System administration, troubleshooting |
| **reviewer** | 0.3 | Code review, quality analysis, security audit |
| **planner** | 0.5 | Project planning, task breakdown, architecture |

### Cascading Agents

Auto-switch to a larger model when the primary one gets stuck:

```bash
forge chat --cascade    # Enable automatic model escalation
```

The cascade chain adapts to your hardware:
- compact: 3b → 7b
- standard: 7b → 14b
- workstation: 14b → 32b

### Permission System

Agents ask for approval before dangerous actions:

| Action Type | Default Level | Examples |
|-------------|---------------|----------|
| Read operations | Auto-approve | File reads, web searches, git status |
| Write operations | Confirm once | File writes, edits |
| Dangerous operations | Always confirm | Shell commands, git push, git commit |

```bash
forge chat --auto-approve  # Skip all permission prompts (use with caution)
```

### Create Your Own Agents

```bash
forge agent create            # Interactive agent creation
forge agent list              # List all agents
forge agent run my-agent      # Run a custom agent
```

Or create a YAML file in `agents/`:

```yaml
name: my-agent
description: "Specialized assistant for data analysis"
model: qwen2.5-coder:7b
system_prompt: |
  You are a data analysis expert. Help users analyze data,
  create visualizations, and derive insights.
tools:
  - filesystem
  - shell
  - web
temperature: 0.3
```

### MCP (Model Context Protocol)

Web search is enabled by default — no API keys needed. 30+ MCPs available across 9 categories.

```bash
forge mcp list          # List available MCPs
forge mcp add github    # Add GitHub integration
forge mcp search sql    # Search for database MCPs
```

Or manage MCPs in natural language during chat:
> "Add a PostgreSQL MCP so I can query my database"

See the full [MCP Guide](docs/mcp-guide.md) for all available MCPs, configuration, and how to create your own.

### Codebase Indexing

Index your project for fast symbol search and context retrieval:

```bash
forge index             # Build/update the codebase index
forge search "ClassName" # Search symbols, files, and content
```

Supports Python, JavaScript/TypeScript, Go, and Rust. Persistent index in `.forge/index/` with incremental updates. Agents use the index to find relevant code before making changes.

### Sandboxed Execution

Agents run code in isolated sandboxes with resource limits:
- Timeout enforcement (default 30s)
- Memory limits on Linux/macOS
- Temporary directory isolation — sandbox artifacts don't pollute your project
- Supports Python scripts, shell commands, and test runners

### Fuzzy Edit Matching

Local LLMs sometimes get whitespace or indentation slightly wrong. ollama-forge uses fuzzy matching as a fallback when exact string replacement fails — finding the closest match above a 75% similarity threshold. This makes file edits significantly more reliable with smaller models.

### Multi-File Edit Planning

For cross-file refactoring, the edit planner:
1. Analyzes Python import dependencies to determine edit order
2. Generates a coordinated edit plan using the LLM
3. Executes changes atomically — all edits succeed, or all roll back

### Session Persistence

Save and resume chat sessions:

```bash
# Sessions are saved automatically — resume by ID
forge chat --session abc12345
```

Supports partial ID matching, markdown/JSON export, and auto-generated titles.

### Fill-in-the-Middle (FIM)

Tab-completion endpoint for editor integrations (Continue.dev, etc.):

```
POST /v1/completions
{"prompt": "def fibonacci(", "suffix": "    return result", "max_tokens": 100}
```

Uses Ollama's native FIM support with the `suffix` parameter.

### Multi-Modal Images

Pass images to vision-capable models:

```bash
forge chat --image screenshot.png    # Analyze an image in chat
```

### Git Undo

Safely revert agent-made changes:

```bash
forge undo    # Revert the last [forge]-tagged commit
```

All agent commits are tagged with `[forge]` for easy identification and safe reversal.

### Context Compression

Never hit context limits again. ollama-forge automatically compresses older conversation history while preserving key information (code blocks, decisions, file paths).

Three strategies:
- **Sliding summary** (default) — Summarize old messages, keep recent ones verbatim
- **Progressive** — Multi-pass compression, removing low-information content first
- **Truncate** — Simple sliding window (fastest)

### Model Management

```bash
forge models              # List installed models
forge models pull X       # Pull a new model
forge models recommend    # Hardware-specific recommendations
forge models auto-update  # Update all installed models
forge models remove X     # Remove a model
```

### Community Ideas

ollama-forge collects improvement ideas anonymously from all users (no GitHub CLI needed):

```bash
forge idea submit "Add support for PDF reading"  # Submit an idea
forge idea list                                    # See community ideas
```

Ideas are stored locally and shared to help improve the framework. Opt out with `FORGE_COMMUNITY_IDEAS=0` in `.env`.

### Self-Improvement Agent (Opt-In)

The self-improvement agent uses your spare CPU/GPU resources to iterate on ollama-forge. **Disabled by default** — you choose to enable it:

```bash
forge self-improve --enable     # Enable + run (first time)
forge self-improve              # Run after enabling
forge self-improve -n 5         # Run 5 iterations
```

How it works:
1. Reads community ideas and searches for latest AI/LLM advances
2. Evaluates and implements the most promising improvements
3. **QA agent** generates and runs test cases for the changes
4. Code review checks for security issues and bugs
5. Only ships changes that pass ALL validation
6. **Contributors**: Creates a GitHub PR for review (requires `gh` CLI)
7. **Maintainers**: Pushes directly to `main`, promotes to `stable` (max 1 per 2 days)

Disable anytime with `FORGE_SELF_IMPROVE=0` in `.env`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ CLI (forge)  │  │ Terminal UI │  │    Web UI      │  │
│  └──────┬───────┘  └──────┬──────┘  └───────┬────────┘  │
│         └─────────────────┼─────────────────┘            │
│                           ▼                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │      Agent Orchestrator + Permission Manager       │  │
│  │  • Single-agent and multi-agent systems            │  │
│  │  • Cascading models, tool dispatch, compression    │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Tools + MCP Servers (30+ available)        │  │
│  │  filesystem │ shell │ git │ web │ sandbox │ index  │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │           LLM Layer (Ollama)                        │  │
│  │  context compression │ model mgmt │ FIM │ streaming │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │        Hardware Detection (GPU/CPU/RAM)             │  │
│  │  AMD ROCm │ NVIDIA CUDA │ Apple Silicon │ CPU-only │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Branch Policy

- **`main`** — Active development. Maintainer's self-improvement agent commits here.
- **`stable`** — Tested improvements only. Max 1 auto-push per 2 days (maintainer only).

Community contributions arrive as PRs against `main`. Only the repo owner merges PRs.

## Hardware Profiles

| Profile | GPU Memory | Recommended Model | Context Window |
|---------|-----------|-------------------|---------------|
| compact | < 8 GB | qwen2.5-coder:3b | 4,096 |
| standard | 8-20 GB | qwen2.5-coder:7b | 8,192 |
| workstation | 20-60 GB | qwen2.5-coder:14b | 32,768 |
| high_memory | 60+ GB | qwen2.5-coder:32b | 65,536 |

**CPU-only systems**: Profile is selected by available RAM (total minus 4 GB OS reservation). Thread count is capped 2 below your total to keep the system responsive. Batch sizes are reduced for better CPU throughput.

## Requirements

- **OS**: Linux (Ubuntu 22.04+, Fedora 39+, Arch), macOS 12+, Windows 10+
- **Python**: 3.10+
- **Ollama**: Latest version
- **GPU**: AMD with ROCm (recommended), NVIDIA with CUDA, Apple Silicon (Metal), or CPU-only

## Contributing

**Submit ideas** (no GitHub account needed):
```bash
forge idea submit "your idea here"
```

**Run the self-improvement agent** (creates PRs automatically):
```bash
forge self-improve --enable   # Opt in and start contributing
```

**Manual contributions**:
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and run tests (`python -m pytest tests/`)
4. Submit a pull request

Only the repo owner and their AI agents have direct push access. All community contributions go through PRs.

## Documentation

- [MCP Guide](docs/mcp-guide.md) — All 30+ MCPs, configuration, creating custom MCPs
- [Quick Start](docs/quickstart.md) — 5-minute getting started guide
- [Hardware Guide](docs/hardware-guide.md) — AMD ROCm setup, troubleshooting
- [Architecture](docs/architecture.md) — System architecture overview

## License

Apache 2.0 — See [LICENSE](LICENSE)
