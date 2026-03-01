# Quick Start Guide

Get up and running with ollama-forge in 5 minutes.

## Prerequisites

- Linux (Ubuntu 22.04+, Fedora 39+, or Arch-based)
- Python 3.10+
- 8+ GB RAM (16+ GB recommended for 7B models)

## Step 1: Install

```bash
git clone https://github.com/ollama-forge/ollama-forge.git
cd ollama-forge
bash install.sh
```

The installer will:
1. Install system dependencies
2. Detect your GPU and install ROCm if you have an AMD GPU
3. Install Ollama
4. Set up a Python virtual environment
5. Recommend and pull a model for your hardware

## Step 2: Activate

```bash
source .venv/bin/activate
```

## Step 3: Chat

```bash
forge chat
```

That's it! You're chatting with a local AI model.

## Step 4: Explore

```bash
# Check your hardware profile
forge hardware

# See recommended models
forge models recommend

# Try the coding agent
forge chat --agent coder

# Search the web
# (just ask in chat — web search is enabled by default)

# Create a custom agent
forge agent create
```

## What's Next?

- Read the [Hardware Guide](hardware-guide.md) to optimize for your GPU
- Learn about [MCP integration](mcp-guide.md) to add more tools
- Create custom agents with the [Agents Guide](agents-guide.md)
- Understand [Context Compression](context-compression.md) to handle long conversations
