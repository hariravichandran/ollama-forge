"""Git tool: repository operations for agents.

Includes auto-commit, undo, branch management, and LLM-generated commit messages.
Agents can make changes and auto-commit with descriptive messages, and users
can undo agent commits with a single command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, TYPE_CHECKING

from forge.utils.logging import get_logger

if TYPE_CHECKING:
    from forge.llm.client import OllamaClient

log = get_logger("tools.git")

# Tag prefix for agent-generated commits (used by undo)
AGENT_COMMIT_TAG = "[forge]"


class GitTool:
    """Git operations for agents.

    Features:
    - Basic: status, diff, log, commit
    - Auto-commit: stage + commit with LLM-generated message after edits
    - Undo: revert the last agent commit safely (git revert, not reset)
    - Branch: create/switch branches per task
    - Stash: save/restore work in progress
    """

    name = "git"
    description = "Git repository operations (status, diff, log, commit, undo, branch)"

    def __init__(self, working_dir: str = ".", client: "OllamaClient | None" = None):
        self.working_dir = working_dir
        self.client = client

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
                    "description": "Stage and commit changes with a descriptive message",
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
            {
                "type": "function",
                "function": {
                    "name": "git_undo",
                    "description": "Undo the last agent-made commit using git revert (safe, creates a new commit)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_create_branch",
                    "description": "Create and switch to a new branch for a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Branch name (e.g., 'fix/auth-bug')"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_stash",
                    "description": "Stash or restore work in progress",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["save", "pop", "list"], "description": "Stash action"},
                            "message": {"type": "string", "description": "Stash message (for save)"},
                        },
                        "required": ["action"],
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
            "git_undo": self._undo,
            "git_create_branch": self._create_branch,
            "git_stash": self._stash,
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
        return self._run_git("log", "--oneline", f"-{count}")

    def _commit(self, message: str, files: list[str] | None = None) -> str:
        # Check for unresolved merge conflicts before committing
        conflict_check = self._check_conflicts()
        if conflict_check:
            return conflict_check

        if files:
            for f in files:
                self._run_git("add", f)
        else:
            self._run_git("add", "-A")

        # Tag agent commits for undo tracking
        full_message = f"{AGENT_COMMIT_TAG} {message}"
        return self._run_git("commit", "-m", full_message)

    def _check_conflicts(self) -> str:
        """Check for unresolved merge conflicts in tracked files.

        Checks both git status indicators (UU, AA, DD) and conflict
        markers in file content (<<<<<<<, =======, >>>>>>>).

        Returns an error message if conflicts found, empty string if clean.
        """
        status = self._run_git("status", "--porcelain")
        if status == "(no output)":
            return ""

        conflict_files = []
        modified_files = []
        for line in status.splitlines():
            # 'UU', 'AA', 'DD' etc. indicate unmerged files
            if line[:2] in ("UU", "AA", "DD", "AU", "UA", "DU", "UD"):
                conflict_files.append(line[3:].strip())
            # Track modified files for marker check
            elif line[:2].strip():
                modified_files.append(line[3:].strip())

        # Also check modified files for conflict markers in content
        marker_conflicts = self._check_conflict_markers(modified_files)
        conflict_files.extend(marker_conflicts)

        if conflict_files:
            unique = list(dict.fromkeys(conflict_files))  # dedup preserving order
            files_str = ", ".join(unique[:5])
            return f"Cannot commit: unresolved merge conflicts in {files_str}. Resolve conflicts first."
        return ""

    def _check_conflict_markers(self, files: list[str]) -> list[str]:
        """Check file content for git conflict markers (<<<<<<<, >>>>>>>).

        Returns list of files containing unresolved conflict markers.
        """
        import re
        conflict_re = re.compile(r"^(<{7}|={7}|>{7})", re.MULTILINE)
        marker_files = []
        for f in files:
            file_path = Path(self.working_dir) / f
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                content = file_path.read_text(errors="replace")
                if conflict_re.search(content):
                    marker_files.append(f)
            except (OSError, PermissionError):
                continue
        return marker_files

    def _undo(self) -> str:
        """Undo the last agent commit using git revert (safe).

        Only reverts commits tagged with AGENT_COMMIT_TAG.
        """
        # Find the last agent commit
        log_output = self._run_git("log", "--oneline", "-20")
        for line in log_output.splitlines():
            if AGENT_COMMIT_TAG in line:
                commit_hash = line.split()[0]
                result = self._run_git("revert", "--no-edit", commit_hash)
                return f"Reverted agent commit {commit_hash}: {result}"

        return "No agent commits found to undo"

    def _create_branch(self, name: str) -> str:
        """Create and switch to a new branch.

        Checks if the branch already exists before creating.
        """
        # Check if branch already exists
        existing = self._run_git("branch", "--list", name)
        if existing and existing != "(no output)":
            return f"Branch '{name}' already exists. Use 'git checkout {name}' to switch to it."
        result = self._run_git("checkout", "-b", name)
        return result

    def _stash(self, action: str, message: str = "") -> str:
        """Stash operations."""
        if action == "save":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            return self._run_git(*args)
        elif action == "pop":
            return self._run_git("stash", "pop")
        elif action == "list":
            return self._run_git("stash", "list")
        return f"Unknown stash action: {action}"

    # --- Auto-commit workflow ---

    def auto_commit(self, files_changed: list[str], description: str = "") -> str:
        """Auto-commit changes with an LLM-generated commit message.

        Args:
            files_changed: List of file paths that were modified.
            description: Brief description of what was changed (helps LLM).

        Returns:
            Commit result string.
        """
        # Check if there are actual changes
        status = self._status()
        if "nothing to commit" in status.lower() or status == "(no output)":
            return "Nothing to commit"

        # Generate commit message
        message = self._generate_commit_message(files_changed, description)

        # Stage specific files
        for f in files_changed:
            self._run_git("add", f)

        # Commit with agent tag
        full_message = f"{AGENT_COMMIT_TAG} {message}"
        result = self._run_git("commit", "-m", full_message)
        log.info("Auto-committed: %s", message)
        return result

    def _generate_commit_message(self, files_changed: list[str], description: str) -> str:
        """Generate a commit message using LLM or from description."""
        if not self.client:
            # No LLM — use description or generic message
            if description:
                return description[:72]
            return f"Update {', '.join(f[:30] for f in files_changed[:3])}"

        # Get the diff for context
        diff = self._run_git("diff", "--cached", "--stat")
        if diff == "(no output)":
            # Stage first to get diff
            for f in files_changed:
                self._run_git("add", f)
            diff = self._run_git("diff", "--cached", "--stat")

        prompt = (
            f"Write a concise git commit message (max 72 chars first line) for these changes.\n"
            f"Files: {', '.join(files_changed)}\n"
            f"Description: {description}\n"
            f"Diff stats:\n{diff}\n\n"
            f"Reply with ONLY the commit message, no quotes or explanation."
        )

        try:
            result = self.client.generate(
                prompt=prompt,
                system="You write concise, conventional git commit messages. Use imperative mood (e.g., 'Add feature', 'Fix bug'). Max 72 chars first line.",
                timeout=30,
                temperature=0.3,
            )
            message = result.get("response", "").strip()
            # Clean up: remove quotes, limit length
            message = message.strip('"\'')
            first_line = message.splitlines()[0][:72] if message else ""
            # Validate the generated message
            first_line = self._validate_commit_message(first_line)
            return first_line or description[:72] or "Update files"
        except Exception:
            return description[:72] if description else "Update files"

    @staticmethod
    def _validate_commit_message(message: str) -> str:
        """Validate and clean a commit message.

        Ensures:
        - Non-empty after stripping
        - First line <= 72 chars
        - No excessive repetition
        - Strips trailing periods (conventional commits style)
        """
        message = message.strip()
        if not message:
            return ""

        # Truncate first line
        first_line = message.splitlines()[0][:72]

        # Detect excessive repetition (e.g., "update update update")
        words = first_line.lower().split()
        if words and len(set(words)) == 1 and len(words) > 2:
            return ""  # reject repetitive messages

        # Strip trailing period (conventional commits style)
        first_line = first_line.rstrip(".")

        return first_line

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        return self._run_git("rev-parse", "--abbrev-ref", "HEAD")

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        status = self._run_git("status", "--porcelain")
        return bool(status and status != "(no output)")

    def get_agent_commits(self, count: int = 10) -> list[str]:
        """Get recent agent-made commits."""
        log_output = self._run_git("log", "--oneline", f"-{count * 2}")
        return [
            line for line in log_output.splitlines()
            if AGENT_COMMIT_TAG in line
        ][:count]
