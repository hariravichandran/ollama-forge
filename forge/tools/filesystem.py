"""Filesystem tool: read, write, edit, glob, grep files."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.filesystem")


class FilesystemTool:
    """File operations for agents."""

    name = "filesystem"
    description = "Read, write, edit, search, and list files"

    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir).resolve()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path (relative to working directory)"},
                            "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                            "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates or overwrites)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace a specific string in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "old_string": {"type": "string", "description": "Exact text to find and replace"},
                            "new_string": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search file contents for a regex pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern to search for"},
                            "glob": {"type": "string", "description": "File glob to limit search (e.g., '*.py')"},
                            "max_results": {"type": "integer", "description": "Maximum results (default 20)"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a filesystem tool function."""
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_files": self._list_files,
            "search_files": self._search_files,
        }
        handler = handlers.get(function_name)
        if not handler:
            return f"Unknown function: {function_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to working directory, with safety checks."""
        resolved = (self.working_dir / path).resolve()
        # Prevent directory traversal outside working directory
        if not str(resolved).startswith(str(self.working_dir)):
            raise ValueError(f"Path escapes working directory: {path}")
        return resolved

    def _read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"
        if resolved.is_dir():
            return f"Path is a directory: {path}"

        content = resolved.read_text(errors="replace")
        lines = content.splitlines()

        if start_line or end_line:
            start = max(0, start_line - 1)
            end = end_line if end_line else len(lines)
            lines = lines[start:end]
            numbered = [f"{i + start + 1:4d}  {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        # Add line numbers
        numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    def _write_file(self, path: str, content: str) -> str:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Written {len(content)} chars to {path}"

    def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"

        content = resolved.read_text()
        count = content.count(old_string)
        if count == 0:
            return f"String not found in {path}"
        if count > 1:
            return f"String found {count} times in {path} — provide more context to make it unique"

        new_content = content.replace(old_string, new_string, 1)
        resolved.write_text(new_content)
        return f"Replaced 1 occurrence in {path}"

    def _list_files(self, pattern: str) -> str:
        matches = sorted(self.working_dir.glob(pattern))
        # Limit results
        if len(matches) > 100:
            matches = matches[:100]
            suffix = f"\n... and {len(matches) - 100} more"
        else:
            suffix = ""

        lines = [str(m.relative_to(self.working_dir)) for m in matches]
        return "\n".join(lines) + suffix if lines else "No files found"

    def _search_files(self, pattern: str, glob: str = "**/*", max_results: int = 20) -> str:
        regex = re.compile(pattern, re.IGNORECASE)
        results: list[str] = []

        for file_path in self.working_dir.glob(glob):
            if not file_path.is_file():
                continue
            try:
                content = file_path.read_text(errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel = file_path.relative_to(self.working_dir)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (limited to {max_results} results)"
            except (PermissionError, IsADirectoryError):
                continue

        return "\n".join(results) if results else "No matches found"
