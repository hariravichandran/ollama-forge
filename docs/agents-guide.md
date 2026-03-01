# Agents Guide

## Built-in Agents

ollama-forge comes with three built-in agents:

### Assistant (Default)
General-purpose chat with access to all tools. Good for everyday tasks.

### Coder
Specialized for code editing, debugging, and project management. Uses lower temperature (0.3) for more precise outputs. Has access to filesystem, shell, and git tools.

### Researcher
Specialized for web search, information synthesis, and citation tracking. Uses web and filesystem tools.

## Switching Agents

```bash
# Via CLI flag
forge chat --agent coder

# In chat (use /agent command)
/agent coder
/agent researcher
/agent assistant

# List all agents
/agents
```

## Creating Custom Agents

### Method 1: CLI
```bash
forge agent create
# Follow the interactive prompts
```

### Method 2: YAML File

Create a file in the `agents/` directory:

```yaml
# agents/data-analyst.yaml
name: data-analyst
description: "Data analysis and visualization expert"
model: qwen2.5-coder:7b
system_prompt: |
  You are a data analysis expert. Help users:
  - Load and explore datasets (CSV, JSON, databases)
  - Clean and transform data using pandas
  - Create visualizations using matplotlib/seaborn
  - Perform statistical analysis
  - Generate reports with findings

  Always explain your methodology. Show your code.
tools:
  - filesystem
  - shell
  - web
temperature: 0.3
max_context: 8192
```

The agent will be automatically loaded when you start forge.

## Agent Systems (Single & Multi-Agent)

### Single-Agent Systems
One agent handles everything. Good for focused tasks.

```bash
forge chat --agent coder  # Single agent: just the coder
```

### Multi-Agent Systems
Multiple agents collaborate on complex tasks. The orchestrator routes messages.

In chat, switch between agents as needed:
```
/agent researcher   → "Find the latest best practices for API design"
/agent coder       → "Now implement a REST API based on those findings"
```

### Tracking Agent Systems

ollama-forge tracks your agent systems:

```bash
forge agent list  # Shows all agents and tracked systems
```

## Available Tools

Each agent can use a combination of these tools:

| Tool | Name | What It Does |
|------|------|-------------|
| filesystem | `filesystem` | Read, write, edit, search files |
| shell | `shell` | Execute shell commands |
| git | `git` | Git status, diff, log, commit |
| web | `web` | Web search (DuckDuckGo) and URL fetch |

## Tips

- **Be specific in system prompts** — The more detailed your system prompt, the better the agent performs.
- **Choose the right temperature** — Use 0.1-0.3 for code/factual tasks, 0.5-0.8 for creative tasks.
- **Pick minimal tools** — Only give agents the tools they need. More tools = more potential for confusion.
- **Start small** — Begin with a single agent, then add more as needed.
