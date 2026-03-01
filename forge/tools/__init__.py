"""Built-in tools for agents: filesystem, shell, git, web, codebase, sandbox."""

from forge.tools.filesystem import FilesystemTool
from forge.tools.shell import ShellTool
from forge.tools.git import GitTool
from forge.tools.web import WebTool
from forge.tools.sandbox import SandboxTool

BUILTIN_TOOLS: dict[str, type] = {
    "filesystem": FilesystemTool,
    "shell": ShellTool,
    "git": GitTool,
    "web": WebTool,
    "sandbox": SandboxTool,
}

__all__ = [
    "FilesystemTool", "ShellTool", "GitTool", "WebTool", "SandboxTool",
    "BUILTIN_TOOLS",
]
