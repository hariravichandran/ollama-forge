"""Agent permission system — prompts users for approval before dangerous actions.

This module provides a configurable permission model that agents use to determine
whether an action needs user approval. The system distinguishes between:

- **Auto-approve**: Safe, read-only operations (file reads, web searches, git status)
- **Confirm once**: Approve an action type for the session (file writes, git commits)
- **Always confirm**: Dangerous operations that require approval every time (shell commands,
  file deletions, service restarts, git push)

Users can customize permission levels in their config or via CLI flags.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from forge.utils.logging import get_logger

log = get_logger("agents.permissions")


class PermissionLevel(Enum):
    """How to handle permission for an action."""

    AUTO_APPROVE = "auto"        # No confirmation needed
    CONFIRM_ONCE = "once"        # Ask once per session, then auto-approve
    ALWAYS_CONFIRM = "always"    # Ask every time


@dataclass
class ActionPermission:
    """Permission definition for a specific action."""

    action: str
    description: str
    level: PermissionLevel
    category: str  # "file", "shell", "git", "web", "system"


# Default permission map — categorizes all known agent actions
DEFAULT_PERMISSIONS: dict[str, ActionPermission] = {
    # File operations
    "read_file": ActionPermission("read_file", "Read a file", PermissionLevel.AUTO_APPROVE, "file"),
    "list_files": ActionPermission("list_files", "List files in directory", PermissionLevel.AUTO_APPROVE, "file"),
    "search_files": ActionPermission("search_files", "Search file contents", PermissionLevel.AUTO_APPROVE, "file"),
    "write_file": ActionPermission("write_file", "Write/create a file", PermissionLevel.CONFIRM_ONCE, "file"),
    "edit_file": ActionPermission("edit_file", "Edit an existing file", PermissionLevel.CONFIRM_ONCE, "file"),

    # Shell operations
    "run_command": ActionPermission("run_command", "Execute a shell command", PermissionLevel.ALWAYS_CONFIRM, "shell"),

    # Git operations
    "git_status": ActionPermission("git_status", "Show git status", PermissionLevel.AUTO_APPROVE, "git"),
    "git_diff": ActionPermission("git_diff", "Show git diff", PermissionLevel.AUTO_APPROVE, "git"),
    "git_log": ActionPermission("git_log", "Show git log", PermissionLevel.AUTO_APPROVE, "git"),
    "git_commit": ActionPermission("git_commit", "Create a git commit", PermissionLevel.ALWAYS_CONFIRM, "git"),

    # Web operations
    "web_search": ActionPermission("web_search", "Search the web", PermissionLevel.AUTO_APPROVE, "web"),
    "web_fetch": ActionPermission("web_fetch", "Fetch a web page", PermissionLevel.AUTO_APPROVE, "web"),
}


class PermissionManager:
    """Manages action permissions and user approvals.

    The permission manager sits between agents and tools, intercepting
    actions that require user approval.

    Usage:
        perms = PermissionManager()

        # Check before executing
        if perms.check("run_command", context={"command": "ls -la"}):
            # Execute the command
            ...
        else:
            # User denied
            ...
    """

    def __init__(
        self,
        permissions: dict[str, ActionPermission] | None = None,
        auto_approve_all: bool = False,
        prompt_fn: Callable[[str], bool] | None = None,
    ):
        if permissions is not None:
            self.permissions = permissions
        else:
            # Deep copy to avoid mutating the global defaults
            self.permissions = {
                k: ActionPermission(action=v.action, description=v.description, level=v.level, category=v.category)
                for k, v in DEFAULT_PERMISSIONS.items()
            }
        self.auto_approve_all = auto_approve_all
        self._session_approvals: set[str] = set()
        self._prompt_fn = prompt_fn or self._default_prompt

    def check(self, action: str, context: dict[str, Any] | None = None) -> bool:
        """Check if an action is allowed. May prompt the user.

        Args:
            action: The action name (e.g., "run_command", "write_file")
            context: Optional context about the action (e.g., {"command": "rm -rf /"})

        Returns:
            True if the action is approved, False if denied.
        """
        if self.auto_approve_all:
            return True

        perm = self.permissions.get(action)
        if not perm:
            # Unknown action — default to always confirm
            log.warning("Unknown action '%s' — requiring confirmation", action)
            return self._prompt_user(action, f"Unknown action: {action}", context)

        if perm.level == PermissionLevel.AUTO_APPROVE:
            return True

        if perm.level == PermissionLevel.CONFIRM_ONCE:
            if action in self._session_approvals:
                return True
            approved = self._prompt_user(action, perm.description, context)
            if approved:
                self._session_approvals.add(action)
            return approved

        if perm.level == PermissionLevel.ALWAYS_CONFIRM:
            return self._prompt_user(action, perm.description, context)

        return False

    def approve_for_session(self, action: str) -> None:
        """Pre-approve an action for the rest of the session."""
        self._session_approvals.add(action)

    def set_level(self, action: str, level: PermissionLevel) -> None:
        """Change the permission level for an action."""
        if action in self.permissions:
            self.permissions[action].level = level
        else:
            self.permissions[action] = ActionPermission(
                action=action,
                description=action,
                level=level,
                category="custom",
            )

    def reset_session(self) -> None:
        """Clear all session approvals."""
        self._session_approvals.clear()

    def _prompt_user(self, action: str, description: str, context: dict | None) -> bool:
        """Prompt the user for approval."""
        context_str = ""
        if context:
            # Show relevant context
            for key, value in context.items():
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                context_str += f"\n  {key}: {val_str}"

        prompt = f"[Permission] {description}"
        if context_str:
            prompt += context_str

        return self._prompt_fn(prompt)

    @staticmethod
    def _default_prompt(message: str) -> bool:
        """Default terminal prompt for permission requests."""
        try:
            print(f"\n{message}")
            response = input("  Allow? [y/N/always] ").strip().lower()
            return response in ("y", "yes", "always")
        except (EOFError, KeyboardInterrupt):
            return False


class AutoApproveManager(PermissionManager):
    """Permission manager that auto-approves everything.

    Use this for non-interactive contexts (scripts, self-improvement agent, tests).
    """

    def __init__(self):
        super().__init__(auto_approve_all=True)
