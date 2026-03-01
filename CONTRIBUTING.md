# Contributing to ollama-forge

Thanks for your interest in contributing! This guide helps you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/hariravichandran/ollama-forge.git
cd ollama-forge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[all]"
pip install pytest

# Verify setup
python -m pytest tests/ -q
forge doctor
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -q

# Specific test file
python -m pytest tests/test_hardware_detect.py -v

# Tests that require Ollama running will auto-skip if unavailable
```

## Project Structure

```
forge/                     # Core package
  hardware/                # GPU/CPU/RAM detection
  llm/                     # Ollama client, context compression, benchmarks
  agents/                  # Agent framework (base, orchestrator, sessions)
  tools/                   # Built-in tools (filesystem, shell, git, web)
  mcp/                     # MCP server management
  ui/                      # Terminal UI (Textual) and Web UI (FastAPI)
  api/                     # OpenAI-compatible API server
  community/               # Ideas collector, self-improvement
tests/                     # Test suite
docs/                      # Documentation
```

## Code Style

- Python 3.10+ with type hints
- Docstrings on all public functions
- Use `forge.utils.logging.get_logger()` for logging
- Never hardcode secrets or API keys
- Prefer editing existing files over creating new ones

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `python -m pytest tests/ -q`
5. Commit with a clear message
6. Open a pull request

## What to Contribute

- Bug fixes
- New tool integrations
- Hardware detection improvements (especially for less-common GPUs)
- MCP registry entries for useful MCP servers
- Documentation improvements
- Test coverage

## Reporting Issues

Open an issue at https://github.com/hariravichandran/ollama-forge/issues with:
- What you expected to happen
- What actually happened
- Your hardware (GPU, RAM) and OS
- Output of `forge doctor`
