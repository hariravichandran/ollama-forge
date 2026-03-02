"""Shell tool: execute commands with user approval."""

from __future__ import annotations

import subprocess
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("tools.shell")


class ShellTool:
    """Execute shell commands with safety checks."""

    name = "shell"
    description = "Execute shell commands with user approval"

    def __init__(self, working_dir: str = ".", auto_approve: bool = False):
        self.working_dir = working_dir
        self.auto_approve = auto_approve

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
        """Run a shell command and return output."""
        log.info("Running: %s", command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            # Truncate very long output
            if len(output) > 10000:
                output = output[:5000] + f"\n... ({len(output)} chars total, truncated) ...\n" + output[-2000:]

            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Error running command: {e}"

    def _is_dangerous(self, command: str) -> bool:
        """Check if a command is potentially destructive."""
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /*",
            "mkfs.",
            "dd if=",
            ":(){",  # fork bomb
            "> /dev/sd",
            "chmod -R 777 /",
            "chmod 000",
            "curl | sh",
            "curl | bash",
            "wget | sh",
            "wget | bash",
            "sudo rm",
            "sudo mkfs",
            "sudo dd",
            "sudo chmod",
            "sudo chown /",
        ]
        cmd_lower = command.lower().strip()
        return any(pattern in cmd_lower for pattern in dangerous_patterns)

    def _is_interactive(self, command: str) -> bool:
        """Check if a command requires interactive input (would hang)."""
        # Single-word commands that are interactive
        interactive_exact = {
            "top", "htop", "ipython", "irb", "python", "python3", "node",
        }

        # Prefixes that indicate interactive mode
        interactive_prefixes = [
            "git rebase -i",
            "git add -i",
            "git add --interactive",
            "vim ", "nano ", "emacs ", "vi ",
            "less ", "more ",
            "python3 -i", "python -i",
            "ssh ", "telnet ", "ftp ",
        ]

        cmd_lower = command.lower().strip()
        if cmd_lower in interactive_exact:
            return True
        return any(cmd_lower.startswith(p) for p in interactive_prefixes)
