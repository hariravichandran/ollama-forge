"""Base agent: chat loop with tool use, memory, and context compression."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from forge.agents.permissions import PermissionManager
from forge.agents.rules import load_project_rules
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
        permissions: PermissionManager | None = None,
    ):
        self.client = client
        self.config = config or AgentConfig()
        self.working_dir = working_dir
        self.permissions = permissions or PermissionManager()
        self.messages: list[dict[str, str]] = []
        self.compressor = ContextCompressor(
            client=client,
            max_tokens=self.config.max_context,
        )

        # Load project rules and prepend to system prompt
        rules = load_project_rules(working_dir)
        if rules:
            self._system_prompt = (
                f"{self.config.system_prompt}\n\n"
                f"--- Project Rules ---\n{rules}"
            )
        else:
            self._system_prompt = self.config.system_prompt

        # Initialize tools
        self._tools: dict[str, Any] = {}
        for tool_name in self.config.tools:
            tool_class = BUILTIN_TOOLS.get(tool_name)
            if tool_class:
                self._tools[tool_name] = tool_class(working_dir=working_dir)

        # Cache tool definitions (rebuilt only when tools change)
        self._cached_tool_defs: list[dict[str, Any]] | None = None

        # Build function_name → tool instance map for O(1) dispatch
        self._function_tool_map: dict[str, Any] = {}
        for tool in self._tools.values():
            for defn in tool.get_tool_definitions():
                func_name = defn.get("function", {}).get("name", "")
                if func_name:
                    self._function_tool_map[func_name] = tool

        # Circuit breaker: track consecutive failures per tool function
        self._tool_failure_counts: dict[str, int] = {}
        self._tool_circuit_threshold = 3  # open circuit after N consecutive failures

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for Ollama tool calling (cached)."""
        if self._cached_tool_defs is None:
            definitions = []
            for tool in self._tools.values():
                definitions.extend(tool.get_tool_definitions())
            self._cached_tool_defs = definitions
        return self._cached_tool_defs

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Handles tool calls automatically — loops until the agent produces
        a final text response (no more tool calls).
        """
        self.messages.append({"role": "user", "content": user_message})

        # Compress context if needed
        compressed = self.compressor.compress(self.messages)

        # Build messages with system prompt (includes project rules if any)
        messages = [{"role": "system", "content": self._system_prompt}] + compressed

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
        """Stream a response, yielding typed event dicts.

        Yields dicts with 'type' key: 'text', 'tool_call', 'done', 'error'.
        Handles tool calls inline — executes tools and includes results in stream.
        """
        self.messages.append({"role": "user", "content": user_message})
        compressed = self.compressor.compress(self.messages)
        messages = [{"role": "system", "content": self._system_prompt}] + compressed
        tools = self.get_tool_definitions()

        full_response = []
        tool_results = []
        for event in self.client.stream_chat(messages, tools=tools if tools else None):
            event_type = event.get("type", "")

            if event_type == "text":
                full_response.append(event.get("content", ""))
                yield event
            elif event_type == "tool_call":
                # Execute tool calls
                for tc in event.get("tool_calls", []):
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", {})
                    log.info("Stream tool call: %s(%s)", func_name, json.dumps(func_args)[:200])
                    tool_result = self._execute_tool(func_name, func_args)
                    tool_results.append({"name": func_name, "result": tool_result})
                    yield {"type": "tool_result", "name": func_name, "result": tool_result}
                yield event
            elif event_type == "done":
                yield event
                break
            elif event_type == "error":
                yield event
                break
            else:
                yield event

        # Store tool results in conversation history so they persist across turns
        for tr in tool_results:
            self.messages.append({"role": "tool", "content": tr["result"]})
        self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def _execute_tool(self, function_name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the appropriate tool handler.

        Implements a circuit breaker pattern: after N consecutive failures
        for the same tool function, the circuit "opens" and the tool is
        temporarily skipped to prevent cascading failures.

        Checks permissions before executing. If the user denies
        the action, returns a message indicating denial.
        Wraps execution in try/except to prevent tool errors from
        crashing the chat loop.
        """
        # Circuit breaker: skip tools that have failed too many times in a row
        failure_count = self._tool_failure_counts.get(function_name, 0)
        if failure_count >= self._tool_circuit_threshold:
            log.warning(
                "Circuit breaker open for '%s' (%d consecutive failures), skipping",
                function_name, failure_count,
            )
            return (
                f"Tool '{function_name}' is temporarily unavailable after "
                f"{failure_count} consecutive failures. Try a different approach."
            )

        # Check permission before executing
        if not self.permissions.check(function_name, context=args):
            log.info("Permission denied for %s", function_name)
            return f"Action '{function_name}' was denied by the user."

        # Use cached function→tool mapping for O(1) lookup
        tool = self._function_tool_map.get(function_name)
        if tool:
            try:
                result = tool.execute(function_name, args)
                # Success — reset failure counter
                self._tool_failure_counts[function_name] = 0
                return result
            except Exception as e:
                log.error("Tool execution error for %s: %s", function_name, e)
                self._tool_failure_counts[function_name] = failure_count + 1
                return f"Tool error in '{function_name}': {e}"

        return f"Unknown tool function: {function_name}"

    def reset_circuit_breaker(self, function_name: str | None = None) -> None:
        """Reset circuit breaker for a specific tool or all tools.

        Call this when the underlying issue has been resolved
        (e.g., a service restarted, a file restored).
        """
        if function_name:
            self._tool_failure_counts.pop(function_name, None)
        else:
            self._tool_failure_counts.clear()

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
