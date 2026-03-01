# ollama-forge

**Batteries-included local AI framework for Ollama.**

A free, open-source alternative to paid AI coding assistants. Run powerful AI agents locally on your machine — no cloud APIs, no subscriptions, no data leaving your computer.

## What is ollama-forge?

ollama-forge turns your Linux machine into a full AI development environment:

- **Hardware auto-detection** — Detects your GPU (AMD ROCm, NVIDIA CUDA) and optimizes Ollama automatically
- **Context compression** — Intelligent conversation summarization so you never hit context limits
- **Multi-agent framework** — Create, manage, and orchestrate single-agent and multi-agent systems
- **MCP integration** — Web search enabled by default, add more tools via natural language
- **Self-improvement** — The framework improves itself using community ideas and latest AI research
- **Auto-update models** — Keep your local models up to date with one command

Optimized for **AMD iGPU + ROCm** (our primary target), but works with NVIDIA GPUs and CPU-only setups too.

## Quick Start

```bash
# One-command install
git clone https://github.com/ollama-forge/ollama-forge.git
cd ollama-forge
bash install.sh

# Or manual setup
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

ollama-forge detects your GPU, CPU, and RAM, then selects optimal model sizes, context windows, and batch sizes automatically. For AMD GPUs, it configures ROCm environment variables (`HSA_OVERRIDE_GFX_VERSION`, `OLLAMA_FLASH_ATTENTION`, etc.).

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

Web search is enabled by default — no API keys needed.

```bash
forge mcp list          # List available MCPs
forge mcp add github    # Add GitHub integration
forge mcp search sql    # Search for database MCPs
```

Or manage MCPs in natural language during chat:
> "Add a PostgreSQL MCP so I can query my database"

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

### Community & Self-Improvement

ollama-forge can improve itself:

```bash
forge idea submit "Add support for PDF reading"  # Submit an idea
forge idea list                                    # See community ideas
forge self-improve                                 # Run the self-improvement agent
```

The self-improvement agent:
1. Reads community ideas and searches for latest AI advances
2. Evaluates and implements the most promising improvements
3. Runs tests before committing changes
4. Commits to `main`, promotes tested changes to `stable` (max 1 per 2 days)

Ideas are collected anonymously by default. Opt out with `FORGE_COMMUNITY_IDEAS=0` in `.env`.

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
│  │          Agent Orchestrator + Tracker               │  │
│  │  • Single-agent and multi-agent systems            │  │
│  │  • Tool dispatch, context compression              │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │            Tools + MCP Servers                      │  │
│  │  filesystem │ shell │ git │ web search │ ...       │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │           LLM Layer (Ollama)                        │  │
│  │  context compression │ model management │ streaming │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       ▼                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │        Hardware Detection (GPU/CPU/RAM)             │  │
│  │  AMD ROCm │ NVIDIA CUDA │ CPU-only │ auto-config   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Branch Policy

- **`main`** — Active development. The self-improvement agent commits here.
- **`stable`** — Tested improvements only. Max 1 auto-push per 2 days.

## Hardware Profiles

| Profile | GPU Memory | Recommended Model | Context Window |
|---------|-----------|-------------------|---------------|
| compact | < 8 GB | qwen2.5-coder:3b | 4,096 |
| standard | 8-20 GB | qwen2.5-coder:7b | 8,192 |
| workstation | 20-60 GB | qwen2.5-coder:14b | 32,768 |
| high_memory | 60+ GB | qwen2.5-coder:32b | 65,536 |

## Requirements

- **OS**: Linux (Ubuntu 22.04+, Fedora 39+, Arch)
- **Python**: 3.10+
- **Ollama**: Latest version
- **GPU**: AMD with ROCm (recommended), NVIDIA with CUDA, or CPU-only

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Submit ideas: `forge idea submit "your idea here"`
4. Or just use the self-improvement agent: `forge self-improve`

## License

Apache 2.0 — See [LICENSE](LICENSE)
