"""Auto-fix loop: run linters/tests after edits and feed errors back to the agent.

When an agent edits a file, the auto-fix loop automatically:
1. Runs configured checks (linters, type checkers, tests)
2. If any check fails, feeds the error back to the agent
3. The agent attempts to fix the error
4. Repeat until all checks pass or max retries reached

This closes the feedback loop — agents produce working code, not just
plausible code. Inspired by Aider's lint-and-fix and Claude Code's
auto-correction patterns.

Usage:
    fixer = AutoFixer(client, working_dir="/path/to/project")

    # Configure checks
    fixer.add_check("python lint", "ruff check {file}")
    fixer.add_check("python type", "mypy {file}")
    fixer.add_check("tests", "python -m pytest tests/ -x -q")

    # Run after an edit
    result = fixer.check_and_fix(
        files_changed=["src/main.py"],
        agent=my_agent,
    )
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("agents.autofix")

# Maximum fix attempts before giving up
MAX_FIX_ATTEMPTS = 3


@dataclass
class Check:
    """A check to run after file edits."""

    name: str
    command: str  # use {file} as placeholder for the edited file path
    run_per_file: bool = True  # True = run once per file, False = run once total


@dataclass
class CheckResult:
    """Result of running a check."""

    check_name: str
    passed: bool
    output: str
    file: str = ""


@dataclass
class AutoFixResult:
    """Result of the auto-fix loop."""

    all_passed: bool
    checks_run: int
    fixes_attempted: int
    final_errors: list[str] = field(default_factory=list)


# Default checks for common languages
DEFAULT_CHECKS: dict[str, list[Check]] = {
    ".py": [
        Check("python-syntax", "python -c \"import py_compile; py_compile.compile('{file}', doraise=True)\""),
        Check("python-lint", "ruff check {file} --no-fix", run_per_file=True),
    ],
    ".js": [
        Check("js-syntax", "node --check {file}", run_per_file=True),
    ],
    ".ts": [
        Check("ts-syntax", "npx tsc --noEmit {file}", run_per_file=True),
    ],
}


class AutoFixer:
    """Runs checks after edits and uses the agent to fix errors.

    The fixer maintains a list of checks that run after any file edit.
    If a check fails, the error output is fed back to the agent as a
    message, asking it to fix the issue. This loops until all checks
    pass or max retries is reached.
    """

    def __init__(
        self,
        working_dir: str = ".",
        max_attempts: int = MAX_FIX_ATTEMPTS,
        auto_detect: bool = True,
    ):
        self.working_dir = Path(working_dir)
        self.max_attempts = max_attempts
        self.checks: list[Check] = []
        self.project_checks: list[Check] = []  # run once, not per-file

        if auto_detect:
            self._detect_project_checks()

    def add_check(self, name: str, command: str, run_per_file: bool = True) -> None:
        """Add a check to run after edits."""
        check = Check(name=name, command=command, run_per_file=run_per_file)
        if run_per_file:
            self.checks.append(check)
        else:
            self.project_checks.append(check)

    def run_checks(self, files_changed: list[str]) -> list[CheckResult]:
        """Run all applicable checks on the changed files.

        Returns a list of CheckResult objects.
        """
        results = []

        for filepath in files_changed:
            path = Path(filepath)
            suffix = path.suffix

            # Get checks for this file type
            applicable = [c for c in self.checks if c.run_per_file]

            # Add default checks for the file's language
            if suffix in DEFAULT_CHECKS:
                for default_check in DEFAULT_CHECKS[suffix]:
                    if not any(c.name == default_check.name for c in applicable):
                        applicable.append(default_check)

            # Run per-file checks
            for check in applicable:
                result = self._run_check(check, filepath)
                results.append(result)

        # Run project-level checks (once, not per file)
        for check in self.project_checks:
            result = self._run_check(check)
            results.append(result)

        return results

    def check_and_fix(
        self,
        files_changed: list[str],
        fix_callback: Any = None,
    ) -> AutoFixResult:
        """Run checks and attempt to fix errors.

        Args:
            files_changed: List of file paths that were edited.
            fix_callback: Optional callable(error_message) -> str that
                attempts to fix the error. If not provided, errors are
                just reported without auto-fix.

        Returns:
            AutoFixResult with details of what happened.
        """
        total_checks = 0
        total_fixes = 0
        final_errors = []

        for attempt in range(self.max_attempts):
            results = self.run_checks(files_changed)
            total_checks += len(results)

            failures = [r for r in results if not r.passed]
            if not failures:
                log.info("All checks passed (attempt %d)", attempt + 1)
                return AutoFixResult(
                    all_passed=True,
                    checks_run=total_checks,
                    fixes_attempted=total_fixes,
                )

            if not fix_callback:
                # No auto-fix available — just report errors
                final_errors = [f"{r.check_name}: {r.output[:200]}" for r in failures]
                break

            # Attempt to fix each failure
            for failure in failures:
                error_msg = (
                    f"Check '{failure.check_name}' failed"
                    f"{' for ' + failure.file if failure.file else ''}:\n"
                    f"{failure.output[:500]}\n\n"
                    f"Please fix this error."
                )
                log.info("Auto-fix attempt %d: %s", attempt + 1, failure.check_name)
                fix_callback(error_msg)
                total_fixes += 1

        # Final check after all fix attempts
        results = self.run_checks(files_changed)
        failures = [r for r in results if not r.passed]
        final_errors = [f"{r.check_name}: {r.output[:200]}" for r in failures]

        return AutoFixResult(
            all_passed=len(failures) == 0,
            checks_run=total_checks + len(results),
            fixes_attempted=total_fixes,
            final_errors=final_errors,
        )

    def _run_check(self, check: Check, filepath: str = "") -> CheckResult:
        """Run a single check command."""
        command = check.command
        if "{file}" in command and filepath:
            command = command.replace("{file}", filepath)
        elif "{file}" in command:
            # Skip per-file checks when no file is specified
            return CheckResult(check_name=check.name, passed=True, output="skipped")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.working_dir),
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            if not passed:
                log.debug("Check failed: %s\n%s", check.name, output[:300])

            return CheckResult(
                check_name=check.name,
                passed=passed,
                output=output.strip(),
                file=filepath,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                check_name=check.name,
                passed=False,
                output=f"Check timed out: {command}",
                file=filepath,
            )
        except Exception as e:
            return CheckResult(
                check_name=check.name,
                passed=False,
                output=f"Check error: {e}",
                file=filepath,
            )

    def _detect_project_checks(self) -> None:
        """Auto-detect project checks based on config files present."""
        # pytest
        if (self.working_dir / "pytest.ini").exists() or \
           (self.working_dir / "pyproject.toml").exists() or \
           (self.working_dir / "tests").is_dir():
            self.project_checks.append(
                Check("pytest", "python -m pytest tests/ -x -q --tb=short", run_per_file=False)
            )

        # ruff config
        if (self.working_dir / "ruff.toml").exists() or \
           (self.working_dir / ".ruff.toml").exists():
            # Already in DEFAULT_CHECKS for .py files
            pass

        # package.json (JS/TS projects)
        if (self.working_dir / "package.json").exists():
            self.project_checks.append(
                Check("npm-test", "npm test --if-present", run_per_file=False)
            )
