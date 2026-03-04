"""Agent orchestrator: multi-agent coordination and message routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from forge.agents.base import BaseAgent, AgentConfig, load_agent_from_yaml
from forge.llm.client import OllamaClient
from forge.utils.logging import get_logger

log = get_logger("agents.orchestrator")


class AgentOrchestrator:
    """Coordinates multiple agents, routes messages, manages lifecycle.

    Supports both single-agent and multi-agent workflows:
    - Single agent: user chats with one agent directly
    - Multi-agent: orchestrator routes tasks to specialized agents
    """

    def __init__(self, client: OllamaClient, working_dir: str = "."):
        self.client = client
        self.working_dir = working_dir
        self.agents: dict[str, BaseAgent] = {}
        self.active_agent: str = ""

        # Load built-in agents
        self._register_builtin_agents()

        # Load user-defined agents from agents/ directory
        self._load_user_agents()

    def _register_builtin_agents(self) -> None:
        """Register the built-in agents."""
        # Base assistant (default)
        self.register_agent(BaseAgent(
            client=self.client,
            config=AgentConfig(
                name="assistant",
                system_prompt=(
                    "You are a helpful AI assistant running locally via Ollama. "
                    "You can search the web, read/write files, and run commands. "
                    "Be concise and practical. When the user asks you to do something, "
                    "use your tools to actually do it — don't just explain how."
                ),
                tools=["filesystem", "shell", "git", "web"],
                description="General-purpose assistant with all tools",
            ),
            working_dir=self.working_dir,
        ))

        # Coder agent
        self.register_agent(BaseAgent(
            client=self.client,
            config=AgentConfig(
                name="coder",
                system_prompt=(
                    "You are an expert coding assistant. You help users write, debug, "
                    "and refactor code. You can read and edit files, run commands, and "
                    "use git. Always explain your changes. Write clean, well-documented code. "
                    "Prefer editing existing files over creating new ones."
                ),
                tools=["filesystem", "shell", "git"],
                temperature=0.3,
                description="Coding assistant — writes, debugs, and refactors code",
            ),
            working_dir=self.working_dir,
        ))

        # Researcher agent
        self.register_agent(BaseAgent(
            client=self.client,
            config=AgentConfig(
                name="researcher",
                system_prompt=(
                    "You are a research assistant. You search the web, read documents, "
                    "and synthesize information into clear, well-cited summaries. "
                    "Always cite your sources. Be thorough but concise."
                ),
                tools=["web", "filesystem"],
                temperature=0.5,
                description="Research assistant — searches web, summarizes findings",
            ),
            working_dir=self.working_dir,
        ))

        # Set default active agent
        self.active_agent = "assistant"

    def _load_user_agents(self) -> None:
        """Load user-defined agents from YAML files in agents/ directory."""
        agents_dir = Path(self.working_dir) / "agents"
        if not agents_dir.exists():
            return

        for yaml_file in sorted(agents_dir.glob("*.yaml")) + sorted(agents_dir.glob("*.yml")):
            try:
                agent = load_agent_from_yaml(str(yaml_file), self.client, self.working_dir)
                self.register_agent(agent)
                log.info("Loaded user agent: %s from %s", agent.config.name, yaml_file.name)
            except Exception as e:
                log.warning("Failed to load agent from %s: %s", yaml_file, e)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.config.name] = agent

    def switch_agent(self, name: str) -> str:
        """Switch the active agent (case-insensitive)."""
        name = name.strip()
        # Case-insensitive lookup
        if name not in self.agents:
            lower_map = {k.lower(): k for k in self.agents}
            if name.lower() in lower_map:
                name = lower_map[name.lower()]
            else:
                available = ", ".join(self.agents.keys())
                return f"Unknown agent: {name}. Available: {available}"

        self.active_agent = name
        return f"Switched to agent: {name} — {self.agents[name].config.description}"

    def chat(self, message: str) -> str:
        """Route a message to the active agent."""
        # Check for agent switching commands
        if message.startswith("/agent "):
            name = message[7:].strip()
            return self.switch_agent(name)

        if message == "/agents":
            return self._list_agents()

        agent = self.agents.get(self.active_agent)
        if not agent:
            return "No active agent. Use /agent <name> to select one."

        return agent.chat(message)

    def create_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None = None,
        model: str = "",
        temperature: float = 0.7,
        save: bool = True,
    ) -> str:
        """Create a new agent from parameters.

        Validates inputs before creating. If save=True, writes a YAML
        definition to the agents/ directory.
        """
        # Validate inputs
        errors = self._validate_agent_params(name, description, system_prompt, tools, temperature)
        if errors:
            return f"Cannot create agent: {'; '.join(errors)}"

        if name in self.agents:
            return f"Agent '{name}' already exists. Choose a different name."

        config = AgentConfig(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or ["filesystem", "shell", "web"],
            temperature=temperature,
            description=description,
        )

        agent = BaseAgent(client=self.client, config=config, working_dir=self.working_dir)
        self.register_agent(agent)

        # Save to YAML
        if save:
            agents_dir = Path(self.working_dir) / "agents"
            agents_dir.mkdir(exist_ok=True)
            yaml_path = agents_dir / f"{name}.yaml"
            data = {
                "name": name,
                "description": description,
                "model": model,
                "system_prompt": system_prompt,
                "tools": config.tools,
                "temperature": temperature,
                "max_context": config.max_context,
            }
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            log.info("Created agent '%s' and saved to %s", name, yaml_path)
            return f"Agent '{name}' created and saved to {yaml_path}"

        return f"Agent '{name}' created (in-memory only)"

    def delete_agent(self, name: str) -> str:
        """Delete an agent."""
        if name in ("assistant", "coder", "researcher"):
            return f"Cannot delete built-in agent: {name}"

        if name not in self.agents:
            return f"Agent '{name}' not found"

        del self.agents[name]

        # Remove YAML file if it exists
        yaml_path = Path(self.working_dir) / "agents" / f"{name}.yaml"
        if yaml_path.exists():
            yaml_path.unlink()

        if self.active_agent == name:
            self.active_agent = "assistant"

        return f"Agent '{name}' deleted"

    def _list_agents(self) -> str:
        """List all registered agents."""
        lines = ["Registered agents:\n"]
        for name, agent in self.agents.items():
            marker = " *" if name == self.active_agent else "  "
            lines.append(f"  {marker} {name:20s} {agent.config.description}")
        lines.append(f"\nActive: {self.active_agent}")
        lines.append("Use /agent <name> to switch")
        return "\n".join(lines)

    def get_all_stats(self) -> dict[str, Any]:
        """Get stats for all agents."""
        return {name: agent.get_stats() for name, agent in self.agents.items()}

    @staticmethod
    def _validate_agent_params(
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None,
        temperature: float,
    ) -> list[str]:
        """Validate agent creation parameters. Returns list of errors."""
        from forge.tools import BUILTIN_TOOLS

        errors: list[str] = []

        if not name or not name.strip():
            errors.append("name cannot be empty")
        elif not name.replace("-", "").replace("_", "").isalnum():
            errors.append("name must be alphanumeric (hyphens and underscores allowed)")

        if not system_prompt or not system_prompt.strip():
            errors.append("system_prompt cannot be empty")

        if not (0.0 <= temperature <= 2.0):
            errors.append(f"temperature must be 0.0-2.0, got {temperature}")

        if tools:
            unknown = [t for t in tools if t not in BUILTIN_TOOLS]
            if unknown:
                errors.append(f"unknown tools: {', '.join(unknown)} (available: {', '.join(BUILTIN_TOOLS.keys())})")

        return errors
