"""Project rules: per-project custom instructions for agents.

Similar to Cursor's .cursorrules, Claude Code's CLAUDE.md, or Aider's conventions.
Users create a `.forge-rules` file in their project root to customize agent behavior
for that specific codebase.

The rules file is plain text (Markdown supported) and is automatically prepended
to the agent's system prompt when working in that directory.

Supports hierarchical rules:
1. Global rules: ~/.config/ollama-forge/rules.md
2. Project rules: .forge-rules (in project root)
3. Directory rules: .forge-rules (in subdirectory — overrides project-level)

Rules are merged top-down: global → project → directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from forge.utils.logging import get_logger

log = get_logger("agents.rules")

# File names to look for (in priority order)
RULES_FILENAMES = [".forge-rules", "FORGE.md", "CLAUDE.md"]

# Global rules location
GLOBAL_RULES_PATH = Path.home() / ".config" / "ollama-forge" / "rules.md"


def load_project_rules(working_dir: str = ".") -> str:
    """Load and merge rules from global, project, and directory levels.

    Returns the combined rules text, or empty string if no rules found.
    """
    parts = []

    # 1. Global rules
    global_rules = _read_rules_file(GLOBAL_RULES_PATH)
    if global_rules:
        parts.append(f"# Global Rules\n{global_rules}")

    # 2. Walk up from working_dir to find project-level rules
    project_rules, project_dir = _find_project_rules(working_dir)
    if project_rules:
        parts.append(f"# Project Rules\n{project_rules}")

    # 3. Directory-level rules (if different from project root)
    work_path = Path(working_dir).resolve()
    if project_dir and work_path != project_dir:
        dir_rules = _find_rules_in_dir(work_path)
        if dir_rules:
            parts.append(f"# Directory Rules\n{dir_rules}")

    combined = "\n\n".join(parts)
    if combined:
        log.info("Loaded project rules (%d chars)", len(combined))
    return combined


def _find_project_rules(working_dir: str) -> tuple[str, Optional[Path]]:
    """Walk up from working_dir to find the nearest rules file.

    Stops at filesystem root or home directory.
    """
    current = Path(working_dir).resolve()
    home = Path.home()

    while current != current.parent:
        rules = _find_rules_in_dir(current)
        if rules:
            return rules, current

        # Don't search above home directory
        if current == home:
            break
        current = current.parent

    return "", None


def _find_rules_in_dir(directory: Path) -> str:
    """Look for a rules file in a specific directory."""
    for filename in RULES_FILENAMES:
        rules_path = directory / filename
        content = _read_rules_file(rules_path)
        if content:
            return content
    return ""


def _read_rules_file(path: Path) -> str:
    """Read a rules file if it exists."""
    if path.exists() and path.is_file():
        try:
            content = path.read_text().strip()
            if content:
                log.debug("Read rules from %s", path)
                return content
        except (OSError, UnicodeDecodeError) as e:
            log.warning("Could not read rules file %s: %s", path, e)
    return ""


def create_rules_template(directory: str = ".") -> str:
    """Create a template .forge-rules file in the given directory."""
    path = Path(directory) / ".forge-rules"
    if path.exists():
        return f"Rules file already exists: {path}"

    template = """# .forge-rules — Custom instructions for AI agents

# This file is automatically read by ollama-forge when working in this directory.
# Add project-specific instructions, coding conventions, and constraints here.

## Project Overview
# Describe your project briefly so the agent understands the context.

## Coding Style
# - Language and version requirements
# - Formatting preferences (tabs vs spaces, line length, etc.)
# - Naming conventions (camelCase, snake_case, etc.)

## Architecture
# - Key patterns used (MVC, microservices, etc.)
# - Important directories and their purposes
# - Dependencies and frameworks

## Constraints
# - Things the agent should NEVER do
# - Security requirements
# - Performance considerations

## Testing
# - How to run tests
# - Test framework preferences
# - Coverage requirements
"""
    path.write_text(template)
    return f"Created rules template: {path}"
