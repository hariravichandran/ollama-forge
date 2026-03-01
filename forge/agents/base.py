"""Base agent: chat loop with tool use, memory, and context compression."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from forge.llm.client import OllamaClient
from forge.llm.context import ContextCompressor
from forge.tools import BUILTIN_TOOLS
from forge.utils.logging import get_logger

log = get_logger("agents.base")


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str = "assistant"
    model: str = ""  # uses default from hardware profile if empty
    system_prompt: str = (
        "You are a helpful AI assistant running locally via Ollama. "
        "You can use tools to help answer questions and complete tasks. "
        "Be concise and practical."
    )
    tools: list[str] = field(default_factory=lambda: ["filesystem", "shell", "web"])
    temperature: float = 0.7
    max_context: int = 8192
    description: str = "General-purpose assistant"


class BaseAgent:
    """Core agent with chat loop, tool dispatch, and context management.

    This is the foundation for all agents. Users interact with it directly
    for general queries, or it delegates to specialized agents via the orchestrator.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: AgentConfig | None = None,
        working_dir: str = ".",
    ):
        self.client = client
        self.config = config or AgentConfig()
        self.working_dir = working_dir
        self.messages: list[dict[str, str]] = []
        self.compressor = ContextCompressor(
            client=client,
            max_tokens=self.config.max_context,
        )

        # Initialize tools
        self._tools: dict[str, Any] = {}
        for tool_name in self.config.tools:
            tool_class = BUILTIN_TOOLS.get(tool_name)
            if tool_class:
                self._tools[tool_name] = tool_class(working_dir=working_dir)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for Ollama tool calling."""
        definitions = []
        for tool in self._tools.values():
            definitions.extend(tool.get_tool_definitions())
        return definitions

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Handles tool calls automatically — loops until the agent produces
        a final text response (no more tool calls).
        """
        self.messages.append({"role": "user", "content": user_message})

        # Compress context if needed
        compressed = self.compressor.compress(self.messages)

        # Build messages with system prompt
        messages = [{"role": "system", "content": self.config.system_prompt}] + compressed

        # Get tool definitions
        tools = self.get_tool_definitions()

        max_tool_rounds = 10
        for _ in range(max_tool_rounds):
            result = self.client.chat(
                messages=messages,
                tools=tools if tools else None,
                temperature=self.config.temperature,
            )

            if "error" in result:
                error_msg = f"LLM error: {result['error']}"
                log.error(error_msg)
                self.messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            # Check for tool calls
            tool_calls = result.get("tool_calls")
            if tool_calls:
                # Add assistant message with tool calls
                assistant_msg = result.get("response", "")
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

                # Execute each tool call
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", {})

                    log.info("Tool call: %s(%s)", func_name, json.dumps(func_args)[:200])
                    tool_result = self._execute_tool(func_name, func_args)

                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                    })
            else:
                # Final response — no tool calls
                response = result.get("response", "")
                self.messages.append({"role": "assistant", "content": response})
                return response

        return "Max tool rounds reached. Please try a simpler request."

    def stream_chat(self, user_message: str):
        """Stream a response (no tool use — for simple chat)."""
        self.messages.append({"role": "user", "content": user_message})
        compressed = self.compressor.compress(self.messages)
        messages = [{"role": "system", "content": self.config.system_prompt}] + compressed

        full_response = []
        for chunk in self.client.stream_chat(messages):
            full_response.append(chunk)
            yield chunk

        self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def _execute_tool(self, function_name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the appropriate tool handler."""
        for tool in self._tools.values():
            definitions = tool.get_tool_definitions()
            tool_names = [d["function"]["name"] for d in definitions]
            if function_name in tool_names:
                return tool.execute(function_name, args)

        return f"Unknown tool function: {function_name}"

    def reset(self) -> None:
        """Clear conversation history and context cache."""
        self.messages.clear()
        self.compressor.reset()

    def get_stats(self) -> dict[str, Any]:
        """Get agent usage statistics."""
        return {
            "name": self.config.name,
            "model": self.client.model,
            "messages": len(self.messages),
            "llm_stats": {
                "total_calls": self.client.stats.total_calls,
                "total_tokens": self.client.stats.total_tokens,
                "avg_time_s": round(self.client.stats.avg_time_s, 2),
                "errors": self.client.stats.errors,
            },
        }


def load_agent_from_yaml(yaml_path: str, client: OllamaClient, working_dir: str = ".") -> BaseAgent:
    """Load an agent from a YAML configuration file."""
    import yaml
    from pathlib import Path

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {yaml_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = AgentConfig(
        name=data.get("name", path.stem),
        model=data.get("model", ""),
        system_prompt=data.get("system_prompt", AgentConfig.system_prompt),
        tools=data.get("tools", ["filesystem", "shell", "web"]),
        temperature=data.get("temperature", 0.7),
        max_context=data.get("max_context", 8192),
        description=data.get("description", ""),
    )

    if config.model:
        client.switch_model(config.model)

    return BaseAgent(client=client, config=config, working_dir=working_dir)
