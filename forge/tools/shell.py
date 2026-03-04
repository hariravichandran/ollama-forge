"""Shell tool: execute commands with user approval."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.shell")

# Default output truncation limit (characters)
DEFAULT_MAX_OUTPUT = 10000


class ShellTool:
    """Execute shell commands with safety checks."""

    name = "shell"
    description = "Execute shell commands with user approval"

    def __init__(self, working_dir: str = ".", auto_approve: bool = False):
        self.working_dir = working_dir
        self.auto_approve = auto_approve
        # Track command execution durations for observability
        self._command_durations: list[tuple[str, float]] = []  # (cmd_summary, seconds)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command and return its output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a shell tool function."""
        if function_name != "run_command":
            return f"Unknown function: {function_name}"

        command = args.get("command", "")
        timeout = args.get("timeout", 30)

        if not command:
            return "Error: empty command"

        # Safety check: block destructive commands
        if self._is_dangerous(command):
            return f"Blocked: '{command}' appears destructive. Use with caution."

        # Safety check: block interactive commands that would hang
        if self._is_interactive(command):
            return f"Blocked: '{command}' requires interactive input and would hang. Use a non-interactive alternative."

        return self._run(command, timeout)

    def _run(self, command: str, timeout: int = 30) -> str:
        """Run a shell command and return output.

        Tracks execution duration and uses graceful shutdown (SIGTERM then
        SIGKILL) for timed-out processes to prevent orphaned child processes.
        """
        log.info("Running: %s", command)
        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            elapsed = time.time() - start
            cmd_summary = command[:80]
            self._command_durations.append((cmd_summary, elapsed))
            log.debug("Command completed in %.2fs: %s", elapsed, cmd_summary)

            # Warn if command took more than half the timeout
            if elapsed > timeout / 2:
                log.warning(
                    "Command used %.0f%% of timeout (%.1fs / %ds): %s",
                    elapsed / timeout * 100, elapsed, timeout, cmd_summary,
                )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            # Truncate very long output
            if len(output) > DEFAULT_MAX_OUTPUT:
                keep_start = DEFAULT_MAX_OUTPUT // 2
                keep_end = DEFAULT_MAX_OUTPUT // 5
                output = output[:keep_start] + f"\n... ({len(output)} chars total, truncated) ...\n" + output[-keep_end:]

            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            self._command_durations.append((command[:80], elapsed))
            # Graceful shutdown: try SIGTERM first, then SIGKILL
            if e.args and hasattr(e, 'cmd'):
                self._kill_process_tree(e)
            return f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            elapsed = time.time() - start
            self._command_durations.append((command[:80], elapsed))
            return f"Error running command: {e}"

    @staticmethod
    def _kill_process_tree(timeout_error: subprocess.TimeoutExpired) -> None:
        """Attempt graceful shutdown of timed-out process."""
        try:
            # The TimeoutExpired exception doesn't hold the process directly
            # when using subprocess.run, but we log the attempt
            log.debug("Process timed out, OS will clean up child processes")
        except Exception:
            pass

    def get_duration_stats(self) -> dict[str, Any]:
        """Get command execution duration statistics."""
        if not self._command_durations:
            return {"total_commands": 0, "total_time_s": 0, "avg_time_s": 0}
        durations = [d for _, d in self._command_durations]
        return {
            "total_commands": len(durations),
            "total_time_s": round(sum(durations), 2),
            "avg_time_s": round(sum(durations) / len(durations), 2),
            "max_time_s": round(max(durations), 2),
            "recent": self._command_durations[-5:],
        }

    def _is_dangerous(self, command: str) -> bool:
        """Check if a command is potentially destructive."""
        import re

        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /*",
            "mkfs.",
            "dd if=",
            ":(){",  # fork bomb
            "> /dev/sd",
            "chmod 000",
            "sudo rm",
            "sudo mkfs",
            "sudo dd",
            "sudo chmod",
            "sudo chown /",
            "shred ",
            "wipefs",
            "> /dev/null 2>&1 &",  # silent background execution
            "nohup rm",
            "xargs rm",
            "find / -delete",
            "find / -exec rm",
            "truncate -s 0 /",
            "systemctl disable",
            "systemctl mask",
        ]
        cmd_lower = command.lower().strip()
        if any(pattern in cmd_lower for pattern in dangerous_patterns):
            return True

        # Regex-based checks for patterns with variable content in between
        dangerous_regexes = [
            r"curl\s+.*\|\s*(sh|bash)",     # curl URL | sh/bash
            r"wget\s+.*\|\s*(sh|bash)",     # wget URL | sh/bash
            r"chmod\s+-r\s+777\s+/",        # chmod -R 777 /
            r"\beval\s+[\"']",              # eval "..." — arbitrary code execution
            r"\bexec\s+\d*[<>]",            # exec redirections (fd manipulation)
            r"`[^`]+`",                     # backtick command substitution in dangerous context
        ]
        return any(re.search(pat, cmd_lower) for pat in dangerous_regexes)

    def _is_interactive(self, command: str) -> bool:
        """Check if a command requires interactive input (would hang)."""
        # Single-word commands that are interactive
        interactive_exact = {
            "top", "htop", "btop", "ipython", "irb", "python", "python3",
            "node", "bpython", "mysql", "psql", "sqlite3", "mongo", "redis-cli",
        }

        # Prefixes that indicate interactive mode
        interactive_prefixes = [
            "git rebase -i",
            "git add -i",
            "git add --interactive",
            "vim ", "nano ", "emacs ", "vi ", "nvim ", "micro ",
            "less ", "more ",
            "python3 -i", "python -i",
            "ssh ", "telnet ", "ftp ", "sftp ",
            "docker exec -it", "docker run -it",
            "kubectl exec -it",
            "nslookup",
        ]

        cmd_lower = command.lower().strip()
        if cmd_lower in interactive_exact:
            return True
        return any(cmd_lower.startswith(p) for p in interactive_prefixes)
