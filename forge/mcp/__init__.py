"""MCP (Model Context Protocol) integration for ollama-forge."""

from forge.mcp.manager import MCPManager
from forge.mcp.registry import MCP_REGISTRY
from forge.mcp.web_search import WebSearchMCP

__all__ = ["MCPManager", "MCP_REGISTRY", "WebSearchMCP"]
