# CLAUDE.md — Instructions for AI Agents

This file is read by Claude Code, Cursor, Copilot, and other AI coding agents.
All agents working on this repository MUST follow these rules.

---

## Project Overview

**ollama-forge** is a batteries-included, open-source framework for local AI development
using Ollama. It targets Linux machines with AMD iGPU + ROCm but supports other GPU
configurations. Key features: hardware auto-detection, context compression, multi-agent
orchestration, and MCP (Model Context Protocol) integration with web search enabled by
default.

## Key Commands

```bash
source .venv/bin/activate     # Always activate venv first
forge hardware                # Show detected hardware & profile
forge chat                    # Interactive chat with default agent
forge models recommend        # Recommend models for your hardware
forge mcp list                # List active MCP servers
python -m pytest tests/       # Run test suite
```

## Project Structure

- **`forge/`** — Core Python package
  - `hardware/` — GPU/CPU/RAM detection, hardware profiles, ROCm setup
  - `llm/` — Ollama client, model management, context compression
  - `mcp/` — MCP server lifecycle, web search, registry
  - `agents/` — Base agent, orchestrator, coder agent, researcher agent
  - `tools/` — Built-in tools (filesystem, shell, git, web)
  - `ui/` — Terminal UI (Textual) and Web UI (FastAPI)
  - `utils/` — Shared utilities (env, logging)
- **`agents/`** — User-created agent definitions (YAML)
- **`docs/`** — Documentation
- **`tests/`** — Test suite
- **`scripts/`** — Utility scripts (ROCm setup, benchmarks)

## Security: No Secrets in Git

- All user config lives in `.env` (gitignored)
- Use `os.environ.get("VAR_NAME")` for any configurable values
- Never hardcode API keys, tokens, or credentials
- `.env.example` contains templates with comments only

## Code Style

- Python 3.10+ (use type hints, dataclasses, pathlib)
- Keep modules focused — one responsibility per file
- Prefer simple, readable code over clever abstractions
- All public functions need docstrings
- Tests use pytest with no mocks against real Ollama when possible

## Architecture Principles

1. **Local-first**: Everything runs locally. No cloud dependencies.
2. **Hardware-aware**: Auto-detect and optimize for available hardware.
3. **Batteries-included**: Web search, coding tools, and agents work out of the box.
4. **Extensible**: Users add agents via YAML, MCPs via natural language.
5. **No vendor lock-in**: Works with any Ollama-compatible model.
