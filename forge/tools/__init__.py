"""Built-in tools for agents: filesystem, shell, git, web."""

from forge.tools.filesystem import FilesystemTool
from forge.tools.shell import ShellTool
from forge.tools.git import GitTool
from forge.tools.web import WebTool

BUILTIN_TOOLS: dict[str, type] = {
    "filesystem": FilesystemTool,
    "shell": ShellTool,
    "git": GitTool,
    "web": WebTool,
}

__all__ = ["FilesystemTool", "ShellTool", "GitTool", "WebTool", "BUILTIN_TOOLS"]
