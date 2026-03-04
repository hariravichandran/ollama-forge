"""Filesystem tool: read, write, edit, glob, grep files.

Includes fuzzy matching for edit operations — when local LLMs produce
slightly incorrect search strings (whitespace, indentation), the fuzzy
matcher finds the best match instead of failing.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.filesystem")

# Minimum similarity ratio for fuzzy matching (0.0 to 1.0)
FUZZY_MATCH_THRESHOLD = 0.75

# Maximum file size for reading (10 MB)
MAX_READ_SIZE = 10 * 1024 * 1024

# Binary file extensions — refuse to read as text
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".o", ".a",
    ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".wav", ".flac", ".ogg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".class", ".jar", ".war",
}


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
        """Resolve a path relative to working directory, with safety checks.

        Validates against directory traversal and symlink escapes.
        """
        target = self.working_dir / path
        resolved = target.resolve()
        # Prevent directory traversal outside working directory
        if not str(resolved).startswith(str(self.working_dir)):
            raise ValueError(f"Path escapes working directory: {path}")
        # Symlink safety: if the original path is a symlink, verify its
        # resolved target is still within the working directory
        if target.is_symlink():
            link_target = target.resolve()
            if not str(link_target).startswith(str(self.working_dir)):
                raise ValueError(f"Symlink target escapes working directory: {path}")
        return resolved

    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Check if a file is likely binary based on extension and content.

        Uses extension check first (fast), then falls back to reading
        a small sample and checking for null bytes.
        """
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        # Check first 8KB for null bytes (heuristic for binary content)
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except (OSError, PermissionError):
            return False

    def _read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"
        if resolved.is_dir():
            return f"Path is a directory: {path}"

        # Check file size
        file_size = resolved.stat().st_size
        if file_size > MAX_READ_SIZE:
            size_mb = file_size / (1024 * 1024)
            return f"File too large: {path} ({size_mb:.1f} MB, max {MAX_READ_SIZE // (1024*1024)} MB)"

        # Check for binary files
        if self._is_binary(resolved):
            return f"Cannot read binary file: {path} (extension: {resolved.suffix})"

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

        # Try exact match first
        count = content.count(old_string)
        if count == 1:
            new_content = content.replace(old_string, new_string, 1)
            resolved.write_text(new_content)
            return f"Replaced 1 occurrence in {path}"

        if count > 1:
            return f"String found {count} times in {path} — provide more context to make it unique"

        # Exact match failed — try fuzzy matching
        # This handles whitespace/indentation mismatches from local LLMs
        match_result = self._fuzzy_find(content, old_string)
        if match_result:
            actual_text, similarity = match_result
            new_content = content.replace(actual_text, new_string, 1)
            resolved.write_text(new_content)
            return f"Replaced 1 occurrence in {path} (fuzzy match, {similarity:.0%} similar)"

        return f"String not found in {path} (exact and fuzzy match failed)"

    def _fuzzy_find(self, content: str, search: str) -> tuple[str, float] | None:
        """Find the closest match to search string in content using fuzzy matching.

        Returns (actual_text, similarity_ratio) or None if no good match found.
        """
        search_lines = search.splitlines()
        content_lines = content.splitlines()
        n_search = len(search_lines)

        if n_search == 0:
            return None

        best_match = None
        best_ratio = 0.0

        # Slide a window of search_lines length over content_lines
        for i in range(len(content_lines) - n_search + 1):
            window = content_lines[i:i + n_search]
            window_text = "\n".join(window)

            # Quick length check to avoid expensive comparison
            len_ratio = len(search) / max(len(window_text), 1)
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue

            ratio = difflib.SequenceMatcher(None, search, window_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window_text

        if best_match and best_ratio >= FUZZY_MATCH_THRESHOLD:
            return best_match, best_ratio

        return None

    def _list_files(self, pattern: str) -> str:
        matches = sorted(self.working_dir.glob(pattern))
        # Limit results
        total = len(matches)
        if total > 100:
            matches = matches[:100]
            suffix = f"\n... and {total - 100} more"
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
            # Skip binary files in search
            if self._is_binary(file_path):
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
