"""Agent permission system — prompts users for approval before dangerous actions.

This module provides a configurable permission model that agents use to determine
whether an action needs user approval. The system distinguishes between:

- **Auto-approve**: Safe, read-only operations (file reads, web searches, git status)
- **Confirm once**: Approve an action type for the session (file writes, git commits)
- **Always confirm**: Dangerous operations that require approval every time (shell commands,
  file deletions, service restarts, git push)

Users can customize permission levels in their config or via CLI flags.
Includes an audit log for security review and dangerous command detection.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from forge.utils.logging import get_logger

log = get_logger("agents.permissions")

# Audit log rotation
MAX_AUDIT_LOG_ENTRIES = 100_000  # Rotate after this many entries
AUDIT_LOG_CHECK_INTERVAL = 1000  # Check size every N writes

# Secret patterns to redact in audit logs
SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|apikey)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"(password|passwd|pwd)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"(token|secret|auth)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"(sk-|ghp_|gho_|glpat-|xoxb-|xoxp-)\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
]

# Permission request rate limiting
MAX_PROMPTS_PER_MINUTE = 20
RATE_LIMIT_WINDOW_S = 60  # 1 minute window for rate limiting

# Display limits
MAX_CONTEXT_VALUE_DISPLAY = 100  # max chars for context values in prompts
MAX_CONTEXT_VALUE_AUDIT = 200  # max chars for context values in audit log
MAX_ACTION_NAME_LENGTH = 100  # max action name length

# Dangerous shell command patterns that should trigger extra warnings
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "Recursive delete from root"),
    (r"rm\s+-rf\s+~", "Recursive delete of home directory"),
    (r"mkfs\.", "Format filesystem"),
    (r"dd\s+if=", "Raw disk write"),
    (r">\s*/dev/sd", "Write to raw device"),
    (r"chmod\s+-[rR]\s+777\s+/", "Recursive world-writable permissions on root"),
    (r"curl\s+.*\|\s*(sh|bash)", "Piping remote script to shell"),
    (r"wget\s+.*\|\s*(sh|bash)", "Piping remote script to shell"),
    (r"DROP\s+(TABLE|DATABASE)", "SQL destructive operation"),
    (r"TRUNCATE\s+TABLE", "SQL truncate"),
    (r"DELETE\s+FROM\s+\w+\s*;?\s*$", "SQL delete without WHERE clause"),
    (r"pkill\s+-9", "Force kill processes"),
    (r"kill\s+-9\s+1\b", "Kill init/systemd"),
    (r"systemctl\s+(stop|disable|mask)\s+(sshd|firewalld|iptables)", "Disable critical service"),
]


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
    actions that require user approval. Includes audit logging and
    dangerous command detection.

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
        audit_file: str | Path | None = None,
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

        # Audit log — records all permission decisions
        self._audit_file: Path | None = None
        self._audit_write_count = 0
        if audit_file:
            self._audit_file = Path(audit_file)
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

        # Rate limiting for permission prompts
        self._prompt_timestamps: list[float] = []

    def check(self, action: str, context: dict[str, Any] | None = None) -> bool:
        """Check if an action is allowed. May prompt the user.

        Detects dangerous commands and flags them before prompting.
        All decisions are logged to the audit trail.

        Args:
            action: The action name (e.g., "run_command", "write_file")
            context: Optional context about the action (e.g., {"command": "rm -rf /"})

        Returns:
            True if the action is approved, False if denied.
        """
        if not action or len(action) > MAX_ACTION_NAME_LENGTH:
            log.warning("Invalid action name (empty or too long): %s", action[:50] if action else "")
            return False
        # Check for dangerous patterns in context
        danger = self._detect_dangerous(action, context)
        if danger:
            log.warning("Dangerous action detected: %s — %s", action, danger)

        if self.auto_approve_all:
            self._audit_log(action, "auto_approved", context, danger)
            return True

        perm = self.permissions.get(action)
        if not perm:
            # Unknown action — default to always confirm
            log.warning("Unknown action '%s' — requiring confirmation", action)
            approved = self._prompt_user(action, f"Unknown action: {action}", context, danger)
            self._audit_log(action, "approved" if approved else "denied", context, danger)
            return approved

        if perm.level == PermissionLevel.AUTO_APPROVE and not danger:
            self._audit_log(action, "auto_approved", context, danger)
            return True

        if perm.level == PermissionLevel.CONFIRM_ONCE and not danger:
            if action in self._session_approvals:
                self._audit_log(action, "session_approved", context, danger)
                return True
            approved = self._prompt_user(action, perm.description, context, danger)
            if approved:
                self._session_approvals.add(action)
            self._audit_log(action, "approved" if approved else "denied", context, danger)
            return approved

        # ALWAYS_CONFIRM or dangerous action
        approved = self._prompt_user(action, perm.description if perm else action, context, danger)
        self._audit_log(action, "approved" if approved else "denied", context, danger)
        return approved

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

    def _prompt_user(
        self, action: str, description: str, context: dict | None,
        danger: str = "",
    ) -> bool:
        """Prompt the user for approval, with danger warning if applicable.

        Includes rate limiting to prevent agents from spamming prompts.
        """
        # Rate limit: check prompts per minute
        if self._is_rate_limited():
            log.warning("Permission prompt rate limit exceeded for action: %s", action)
            return False

        context_str = ""
        if context:
            # Show relevant context
            for key, value in context.items():
                val_str = str(value)
                if len(val_str) > MAX_CONTEXT_VALUE_DISPLAY:
                    val_str = val_str[:MAX_CONTEXT_VALUE_DISPLAY] + "..."
                context_str += f"\n  {key}: {val_str}"

        prompt = f"[Permission] {description}"
        if danger:
            prompt = f"[WARNING: {danger}] {description}"
        if context_str:
            prompt += context_str

        self._prompt_timestamps.append(time.time())
        return self._prompt_fn(prompt)

    def _is_rate_limited(self) -> bool:
        """Check if permission prompts have exceeded rate limit."""
        now = time.time()
        cutoff = now - RATE_LIMIT_WINDOW_S
        self._prompt_timestamps = [t for t in self._prompt_timestamps if t > cutoff]
        return len(self._prompt_timestamps) >= MAX_PROMPTS_PER_MINUTE

    @staticmethod
    def _detect_dangerous(action: str, context: dict[str, Any] | None) -> str:
        """Detect dangerous patterns in action context.

        Returns a warning string if dangerous, empty string if safe.
        """
        if not context:
            return ""

        # Build a searchable string from all context values
        search_text = " ".join(str(v) for v in context.values())

        for pattern, description in DANGEROUS_PATTERNS:
            if re.search(pattern, search_text, re.IGNORECASE):
                return description

        return ""

    def _audit_log(
        self, action: str, decision: str,
        context: dict[str, Any] | None, danger: str = "",
    ) -> None:
        """Append a permission decision to the audit log.

        Includes secret redaction and automatic log rotation.
        """
        if not self._audit_file:
            return

        entry = {
            "ts": time.time(),
            "action": action,
            "decision": decision,
            "danger": danger,
        }
        if context:
            # Store truncated + redacted context for auditing
            entry["context"] = {
                k: self._redact_secrets(str(v)[:MAX_CONTEXT_VALUE_AUDIT]) for k, v in context.items()
            }

        try:
            with open(self._audit_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._audit_write_count += 1

            # Periodic rotation check
            if self._audit_write_count % AUDIT_LOG_CHECK_INTERVAL == 0:
                self._rotate_audit_log()
        except OSError as e:
            log.debug("Audit log write failed: %s", e)

    @staticmethod
    def _redact_secrets(text: str) -> str:
        """Redact potential secrets from text before logging.

        Detects API keys, passwords, tokens, and other credentials.
        """
        for pattern in SECRET_PATTERNS:
            text = pattern.sub("***REDACTED***", text)
        return text

    def _rotate_audit_log(self) -> None:
        """Rotate audit log if it exceeds MAX_AUDIT_LOG_ENTRIES.

        Compresses old log to .gz and starts a fresh log.
        """
        if not self._audit_file or not self._audit_file.exists():
            return

        try:
            line_count = sum(1 for _ in open(self._audit_file))
            if line_count < MAX_AUDIT_LOG_ENTRIES:
                return

            # Compress old log
            gz_path = self._audit_file.with_suffix(".log.gz")
            with open(self._audit_file, "rb") as f_in:
                with gzip.open(gz_path, "ab") as f_out:
                    f_out.write(f_in.read())

            # Truncate the current log
            self._audit_file.write_text("")
            self._audit_write_count = 0
            log.info("Rotated audit log (%d entries) → %s", line_count, gz_path)
        except OSError as e:
            log.warning("Audit log rotation failed: %s", e)

    def get_audit_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        if not self._audit_file or not self._audit_file.exists():
            return {"entries": 0}

        decisions: dict[str, int] = {}
        try:
            for line in self._audit_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    d = entry.get("decision", "unknown")
                    decisions[d] = decisions.get(d, 0) + 1
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            log.debug("Could not read audit log for stats: %s", e)

        return {
            "entries": sum(decisions.values()),
            "decisions": decisions,
        }

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
