"""MCP server lifecycle management — install, start, stop, configure."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from forge.mcp.registry import MCP_REGISTRY, MCPEntry
from forge.mcp.web_search import WebSearchMCP
from forge.utils.logging import get_logger

log = get_logger("mcp.manager")

# Installation retry settings
MCP_INSTALL_MAX_RETRIES = 3
MCP_INSTALL_BACKOFF_BASE = 2.0  # seconds


class MCPManager:
    """Manages MCP server lifecycle and configuration.

    Handles installation, startup, shutdown, and configuration of MCP servers.
    Web search is enabled by default.
    """

    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path("~/.config/ollama-forge/mcp.yaml").expanduser()

        self._config: dict[str, Any] = self._load_config()
        self._active_servers: dict[str, Any] = {}
        self._server_health: dict[str, bool] = {}  # tracks last known health status

        # Built-in web search — always available
        self.web_search = WebSearchMCP()

        # Ensure web search is enabled by default
        if "web-search" not in self._config:
            self._config["web-search"] = {"enabled": True}
            self._save_config()

    def list_available(self) -> list[dict[str, Any]]:
        """List all known MCP servers with their status."""
        results = []
        for name, entry in MCP_REGISTRY.items():
            status = "built-in" if entry.builtin else "available"
            config = self._config.get(name, {})
            if config.get("enabled"):
                status = "enabled"
            elif name in self._config and not config.get("enabled", True):
                status = "disabled"

            results.append({
                "name": name,
                "description": entry.description,
                "category": entry.category,
                "status": status,
                "builtin": entry.builtin,
            })
        return results

    def enable(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Enable an MCP server."""
        entry = MCP_REGISTRY.get(name)
        if not entry:
            return f"Unknown MCP: {name}. Use 'forge mcp search' to find available MCPs."

        # Install if needed (with retries)
        if not entry.builtin and entry.install_cmd:
            log.info("Installing MCP: %s", name)
            last_error = ""
            for attempt in range(MCP_INSTALL_MAX_RETRIES):
                try:
                    result = subprocess.run(
                        entry.install_cmd, shell=True, capture_output=True, text=True, timeout=120,
                    )
                    if result.returncode == 0:
                        break
                    last_error = result.stderr
                    if attempt < MCP_INSTALL_MAX_RETRIES - 1:
                        delay = MCP_INSTALL_BACKOFF_BASE ** (attempt + 1)
                        log.warning(
                            "MCP install attempt %d failed, retrying in %.0fs: %s",
                            attempt + 1, delay, result.stderr[:100],
                        )
                        time.sleep(delay)
                except subprocess.TimeoutExpired:
                    last_error = "installation timed out"
                    if attempt < MCP_INSTALL_MAX_RETRIES - 1:
                        log.warning("MCP install timed out (attempt %d), retrying", attempt + 1)
                        continue
            else:
                return f"Failed to install {name} after {MCP_INSTALL_MAX_RETRIES} attempts: {last_error}"

        # Save config
        mcp_config = config or entry.config_example.copy()
        mcp_config["enabled"] = True
        self._config[name] = mcp_config
        self._save_config()

        log.info("Enabled MCP: %s", name)
        return f"MCP '{name}' enabled successfully"

    def disable(self, name: str) -> str:
        """Disable an MCP server."""
        if name in self._config:
            self._config[name]["enabled"] = False
            self._save_config()
            return f"MCP '{name}' disabled"
        return f"MCP '{name}' was not enabled"

    def get_enabled(self) -> list[str]:
        """Get list of enabled MCP names."""
        enabled = []
        for name, config in self._config.items():
            if config.get("enabled", False):
                enabled.append(name)
        return enabled

    def get_tools_for_agent(self) -> list[dict[str, Any]]:
        """Get all tool definitions from enabled MCPs."""
        tools = []

        for name in self.get_enabled():
            entry = MCP_REGISTRY.get(name)
            if not entry:
                continue

            if name == "web-search":
                # Built-in web search
                tools.append({
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web using DuckDuckGo",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                            },
                            "required": ["query"],
                        },
                    },
                })
            else:
                # External MCP — expose a generic tool using the MCP name
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp_{name.replace('-', '_')}",
                        "description": entry.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "description": "Action to perform"},
                                "params": {"type": "string", "description": "Parameters as JSON string"},
                            },
                            "required": ["action"],
                        },
                    },
                })

        return tools

    def health_check(self) -> dict[str, bool]:
        """Check health of all enabled MCPs.

        Returns a dict mapping MCP name to health status (True=healthy).
        Built-in MCPs are always healthy. External MCPs are checked
        by verifying their binary/command exists.
        """
        results: dict[str, bool] = {}
        for name in self.get_enabled():
            entry = MCP_REGISTRY.get(name)
            if not entry:
                results[name] = False
                continue

            if entry.builtin:
                results[name] = True
            else:
                # Check if the start command's binary exists
                healthy = self._check_mcp_binary(entry)
                results[name] = healthy
                if not healthy:
                    log.warning("MCP '%s' health check failed", name)

        self._server_health = results
        return results

    @staticmethod
    def _check_mcp_binary(entry: MCPEntry) -> bool:
        """Check if an MCP's package binary is importable/available."""
        if not entry.package:
            return True  # No package means it's a built-in or not applicable
        # Try to check if the package is installed
        try:
            result = subprocess.run(
                ["python3", "-c", f"import importlib; importlib.import_module('{entry.package.replace('-', '_')}')"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                pass
        return {}

    def _save_config(self) -> None:
        """Save MCP configuration to disk with atomic write."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to temp file first, then rename for atomicity
            tmp_path = self.config_path.with_suffix(".yaml.tmp")
            content = yaml.dump(self._config, default_flow_style=False)
            tmp_path.write_text(content)
            tmp_path.rename(self.config_path)
        except OSError as e:
            log.error("Failed to save MCP config: %s", e)
