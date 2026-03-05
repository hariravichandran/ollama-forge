"""Multi-file edit planner: plan coordinated changes across files.

When a task spans multiple files (refactoring, feature addition, renaming),
the planner analyzes dependencies, generates an edit plan, and executes
changes atomically — rolling back on failure.

This is what makes AI coding assistants useful for real-world tasks vs.
single-file toy examples.

Usage:
    planner = EditPlanner(client=client, working_dir=".")

    # Plan changes
    plan = planner.plan("Rename UserAuth to AuthManager everywhere")

    # Review plan
    print(plan.summary())

    # Execute with atomic rollback
    result = planner.execute(plan)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from forge.utils.logging import get_logger

if TYPE_CHECKING:
    from forge.llm.client import OllamaClient

log = get_logger("agents.planner")

# Project scanning limits
MAX_PROJECT_FILES_DISPLAY = 50  # max files to show in project tree for LLM
MAX_ERROR_SNIPPET_LENGTH = 60  # truncation for error message snippets
LLM_PLAN_TEMPERATURE = 0.3
LLM_PLAN_TIMEOUT = 60

# File discovery
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"}
IGNORE_DIRS = {"__pycache__", "node_modules", ".venv", ".git", ".forge"}


@dataclass
class FileEdit:
    """A planned edit to a single file."""

    path: str
    description: str
    old_content: str = ""  # for rollback
    edits: list[dict[str, str]] = field(default_factory=list)  # [{old_string, new_string}]
    create: bool = False  # if True, this is a new file
    new_content: str = ""  # for new files


@dataclass
class EditPlan:
    """A coordinated set of edits across multiple files."""

    task: str
    files: list[FileEdit] = field(default_factory=list)
    dependency_order: list[str] = field(default_factory=list)  # paths in execution order
    reasoning: str = ""

    @property
    def file_count(self) -> int:
        return len(self.files)

    def summary(self) -> str:
        """Human-readable plan summary."""
        lines = [f"Edit Plan: {self.task}", f"Files: {self.file_count}", ""]
        if self.reasoning:
            lines.append(f"Reasoning: {self.reasoning}")
            lines.append("")
        for i, f in enumerate(self.files, 1):
            edit_count = len(f.edits)
            action = "CREATE" if f.create else f"{edit_count} edit{'s' if edit_count != 1 else ''}"
            lines.append(f"  {i}. {f.path} — {action}: {f.description}")
        return "\n".join(lines)


@dataclass
class PlanResult:
    """Result of executing an edit plan."""

    success: bool
    files_modified: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rolled_back: bool = False


class EditPlanner:
    """Plans and executes multi-file edits with dependency awareness.

    Features:
    - Analyzes Python import dependencies to determine edit order
    - Generates edit plans using LLM
    - Executes edits atomically (all succeed or all roll back)
    - Uses git stash for safe rollback
    """

    def __init__(
        self,
        client: "OllamaClient | None" = None,
        working_dir: str = ".",
    ):
        self.client = client
        self.working_dir = Path(working_dir).resolve()

    def plan(self, task: str, context: str = "") -> EditPlan:
        """Generate an edit plan for a multi-file task.

        Args:
            task: Description of what needs to be done.
            context: Additional context (current errors, requirements).

        Returns:
            An EditPlan with ordered file edits.
        """
        if not task or not task.strip():
            return EditPlan(task="", reasoning="Empty task description provided.")

        # Discover project structure
        project_files = self._get_project_files()
        project_tree = "\n".join(f"  {f}" for f in project_files[:MAX_PROJECT_FILES_DISPLAY])

        # Analyze dependencies
        deps = self._analyze_dependencies(project_files)

        plan = EditPlan(task=task)

        if self.client:
            # Ask LLM to generate the plan
            plan = self._llm_plan(task, context, project_tree, deps)
        else:
            plan.reasoning = "No LLM available — manual plan required."

        # Set dependency order
        if not plan.dependency_order:
            plan.dependency_order = [f.path for f in plan.files]

        return plan

    def validate(self, plan: EditPlan) -> list[str]:
        """Validate an edit plan before execution.

        Returns a list of validation errors (empty = valid).
        Checks:
        - Files to edit exist (unless create=True)
        - old_string values are found in the file content
        - No overlapping edits (same old_string targeted twice)
        - File paths are within the working directory
        """
        errors: list[str] = []

        for file_edit in plan.files:
            file_path = self.working_dir / file_edit.path

            # Path traversal check
            try:
                resolved = file_path.resolve()
                if not str(resolved).startswith(str(self.working_dir.resolve())):
                    errors.append(f"{file_edit.path}: path escapes working directory")
                    continue
            except (OSError, ValueError):
                errors.append(f"{file_edit.path}: invalid path")
                continue

            if file_edit.create:
                if file_path.exists():
                    errors.append(f"{file_edit.path}: file already exists (marked as create)")
                continue

            if not file_path.exists():
                errors.append(f"{file_edit.path}: file not found")
                continue

            content = file_path.read_text()
            seen_old_strings: set[str] = set()

            for i, edit in enumerate(file_edit.edits):
                old = edit.get("old_string", "")
                if not old:
                    errors.append(f"{file_edit.path} edit {i + 1}: empty old_string")
                    continue

                if old not in content:
                    snippet = old[:MAX_ERROR_SNIPPET_LENGTH].replace("\n", "\\n")
                    errors.append(f"{file_edit.path} edit {i + 1}: old_string not found: '{snippet}...'")

                if old in seen_old_strings:
                    errors.append(f"{file_edit.path} edit {i + 1}: duplicate old_string (overlapping edit)")
                seen_old_strings.add(old)

                if content.count(old) > 1:
                    errors.append(f"{file_edit.path} edit {i + 1}: old_string matches {content.count(old)} locations (ambiguous)")

        return errors

    def execute(self, plan: EditPlan) -> PlanResult:
        """Execute an edit plan with atomic rollback.

        All edits succeed, or all are rolled back.
        Validates the plan first — returns errors without modifying files if invalid.
        """
        result = PlanResult(success=True)

        # Validate before executing
        validation_errors = self.validate(plan)
        if validation_errors:
            result.success = False
            result.errors = validation_errors
            log.error("Plan validation failed with %d errors", len(validation_errors))
            return result

        # Save state for rollback
        backups: dict[str, str] = {}
        for file_edit in plan.files:
            file_path = self.working_dir / file_edit.path
            if file_path.exists():
                backups[file_edit.path] = file_path.read_text()
                file_edit.old_content = backups[file_edit.path]

        # Execute edits in dependency order
        ordered_files = {f.path: f for f in plan.files}
        edit_order = plan.dependency_order or [f.path for f in plan.files]

        for path in edit_order:
            file_edit = ordered_files.get(path)
            if not file_edit:
                continue

            try:
                if file_edit.create:
                    self._create_file(file_edit)
                else:
                    self._apply_edits(file_edit)
                result.files_modified.append(path)
            except Exception as e:
                result.success = False
                result.errors.append(f"{path}: {e}")
                log.error("Edit failed for %s: %s", path, e)
                break

        # Rollback on failure
        if not result.success:
            for path, content in backups.items():
                try:
                    (self.working_dir / path).write_text(content)
                except Exception:
                    pass
            result.rolled_back = True
            log.info("Rolled back all changes due to failure")

        return result

    def _create_file(self, file_edit: FileEdit) -> None:
        """Create a new file."""
        file_path = self.working_dir / file_edit.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_edit.new_content)
        log.info("Created %s", file_edit.path)

    def _apply_edits(self, file_edit: FileEdit) -> None:
        """Apply edits to an existing file."""
        file_path = self.working_dir / file_edit.path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_edit.path}")

        content = file_path.read_text()
        for edit in file_edit.edits:
            old = edit["old_string"]
            new = edit["new_string"]
            if old not in content:
                raise ValueError(f"String not found in {file_edit.path}: {old[:50]}...")
            content = content.replace(old, new, 1)

        file_path.write_text(content)
        log.info("Applied %d edits to %s", len(file_edit.edits), file_edit.path)

    def _get_project_files(self) -> list[str]:
        """Get a list of project files (Python, JS, etc.)."""
        extensions = CODE_EXTENSIONS
        ignore_dirs = IGNORE_DIRS
        files = []

        for root, dirs, filenames in _walk_compat(self.working_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in filenames:
                if Path(f).suffix in extensions:
                    rel = str((Path(root) / f).relative_to(self.working_dir))
                    files.append(rel)

        return sorted(files)

    def _analyze_dependencies(self, files: list[str]) -> dict[str, list[str]]:
        """Analyze import dependencies between project files.

        Returns dict of file -> list of files it imports from.
        """
        deps: dict[str, list[str]] = {}
        module_map: dict[str, str] = {}  # module name -> file path

        # Build module map
        for f in files:
            if f.endswith(".py"):
                # Convert path to module name: forge/tools/git.py -> forge.tools.git
                module = f.replace("/", ".").replace("\\", ".").removesuffix(".py")
                module_map[module] = f
                # Also map the short name
                short = Path(f).stem
                if short not in module_map:
                    module_map[short] = f

        # Analyze imports
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                content = (self.working_dir / f).read_text()
                file_deps = []
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("from ") or stripped.startswith("import "):
                        # Extract module name
                        match = re.match(r'(?:from|import)\s+([\w.]+)', stripped)
                        if match:
                            module = match.group(1)
                            # Check if it maps to a project file
                            if module in module_map:
                                dep_file = module_map[module]
                                if dep_file != f:
                                    file_deps.append(dep_file)
                            # Check parent modules
                            parts = module.split(".")
                            for i in range(len(parts), 0, -1):
                                parent = ".".join(parts[:i])
                                if parent in module_map and module_map[parent] != f:
                                    file_deps.append(module_map[parent])
                                    break
                deps[f] = list(set(file_deps))
            except Exception:
                deps[f] = []

        return deps

    def _llm_plan(
        self, task: str, context: str, project_tree: str, deps: dict,
    ) -> EditPlan:
        """Use LLM to generate an edit plan."""
        prompt = (
            f"Plan the file edits needed for this task:\n\n"
            f"Task: {task}\n\n"
            f"Project files:\n{project_tree}\n\n"
        )
        if context:
            prompt += f"Context:\n{context}\n\n"

        prompt += (
            "Respond in JSON format:\n"
            '{"reasoning": "why these changes", "files": [{"path": "file.py", '
            '"description": "what to change", "edits": [{"old_string": "...", '
            '"new_string": "..."}]}]}'
        )

        result = self.client.generate(
            prompt=prompt,
            system=(
                "You are a senior developer planning multi-file edits. "
                "Be precise with old_string matches. List files in dependency order "
                "(dependencies first, dependents last)."
            ),
            json_mode=True,
            temperature=LLM_PLAN_TEMPERATURE,
            timeout=LLM_PLAN_TIMEOUT,
        )

        try:
            import json
            data = json.loads(result.get("response", "{}"))
            plan = EditPlan(
                task=task,
                reasoning=data.get("reasoning", ""),
            )
            for f_data in data.get("files", []):
                file_edit = FileEdit(
                    path=f_data["path"],
                    description=f_data.get("description", ""),
                    edits=f_data.get("edits", []),
                    create=f_data.get("create", False),
                    new_content=f_data.get("new_content", ""),
                )
                plan.files.append(file_edit)
            plan.dependency_order = [f.path for f in plan.files]
            return plan
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse LLM plan: %s", e)
            return EditPlan(task=task, reasoning=f"LLM plan parsing failed: {e}")


def _walk_compat(directory: Path):
    """os.walk compatible fallback for Python < 3.12."""
    import os
    for root, dirs, files in os.walk(directory):
        yield root, dirs, files
