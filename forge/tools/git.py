"""Git tool: repository operations for agents."""

from __future__ import annotations

import subprocess
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.git")


class GitTool:
    """Git operations for agents."""

    name = "git"
    description = "Git repository operations (status, diff, log, commit)"

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show the working tree status (modified, staged, untracked files)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show changes in the working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "staged": {"type": "boolean", "description": "Show staged changes (default: unstaged)"},
                            "file": {"type": "string", "description": "Specific file to diff (optional)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": "Show recent commit history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "description": "Number of commits (default 10)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Stage and commit changes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage (default: all modified)"},
                        },
                        "required": ["message"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a git tool function."""
        handlers = {
            "git_status": self._status,
            "git_diff": self._diff,
            "git_log": self._log,
            "git_commit": self._commit,
        }
        handler = handlers.get(function_name)
        if not handler:
            return f"Unknown function: {function_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Error: {e}"

    def _run_git(self, *args: str, timeout: int = 30) -> str:
        """Run a git command and return output."""
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.working_dir,
        )
        output = result.stdout
        if result.stderr and result.returncode != 0:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"

    def _status(self) -> str:
        return self._run_git("status", "--short")

    def _diff(self, staged: bool = False, file: str = "") -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.extend(["--", file])
        return self._run_git(*args)

    def _log(self, count: int = 10) -> str:
        return self._run_git("log", f"--oneline", f"-{count}")

    def _commit(self, message: str, files: list[str] | None = None) -> str:
        if files:
            for f in files:
                self._run_git("add", f)
        else:
            self._run_git("add", "-A")

        return self._run_git("commit", "-m", message)
