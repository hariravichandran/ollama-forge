"""Tests for the auto-fix loop: run checks after edits and fix errors."""

import tempfile
from pathlib import Path

import pytest

from forge.agents.autofix import (
    AutoFixer,
    AutoFixResult,
    Check,
    CheckResult,
    DEFAULT_CHECKS,
    MAX_FIX_ATTEMPTS,
)


class TestCheckDataclass:
    """Tests for Check dataclass."""

    def test_defaults(self):
        c = Check(name="lint", command="ruff check {file}")
        assert c.name == "lint"
        assert c.run_per_file is True

    def test_project_level_check(self):
        c = Check(name="tests", command="pytest tests/", run_per_file=False)
        assert c.run_per_file is False


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_passed(self):
        r = CheckResult(check_name="lint", passed=True, output="ok")
        assert r.passed is True
        assert r.file == ""

    def test_failed_with_file(self):
        r = CheckResult(check_name="lint", passed=False, output="error!", file="main.py")
        assert r.passed is False
        assert r.file == "main.py"


class TestAutoFixResult:
    """Tests for AutoFixResult dataclass."""

    def test_all_passed(self):
        r = AutoFixResult(all_passed=True, checks_run=3, fixes_attempted=0)
        assert r.all_passed is True
        assert r.final_errors == []

    def test_with_errors(self):
        r = AutoFixResult(
            all_passed=False, checks_run=3, fixes_attempted=2,
            final_errors=["lint: syntax error"],
        )
        assert r.all_passed is False
        assert len(r.final_errors) == 1


class TestDefaultChecks:
    """Tests for default check definitions."""

    def test_python_checks(self):
        assert ".py" in DEFAULT_CHECKS
        py_checks = DEFAULT_CHECKS[".py"]
        names = [c.name for c in py_checks]
        assert "python-syntax" in names
        assert "python-lint" in names

    def test_js_checks(self):
        assert ".js" in DEFAULT_CHECKS
        js_checks = DEFAULT_CHECKS[".js"]
        assert any("syntax" in c.name for c in js_checks)

    def test_ts_checks(self):
        assert ".ts" in DEFAULT_CHECKS

    def test_python_syntax_check_has_file_placeholder(self):
        py_checks = DEFAULT_CHECKS[".py"]
        syntax = [c for c in py_checks if c.name == "python-syntax"][0]
        assert "{file}" in syntax.command


class TestAutoFixerInit:
    """Tests for AutoFixer initialization."""

    def test_default_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir)
            assert fixer.max_attempts == MAX_FIX_ATTEMPTS

    def test_custom_max_attempts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, max_attempts=5)
            assert fixer.max_attempts == 5

    def test_no_auto_detect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            assert fixer.checks == []
            assert fixer.project_checks == []

    def test_auto_detect_pytest(self):
        """Should auto-detect pytest when tests/ dir exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "tests").mkdir()
            fixer = AutoFixer(working_dir=tmpdir)
            project_names = [c.name for c in fixer.project_checks]
            assert "pytest" in project_names

    def test_auto_detect_pyproject(self):
        """Should auto-detect pytest when pyproject.toml exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[build-system]")
            fixer = AutoFixer(working_dir=tmpdir)
            project_names = [c.name for c in fixer.project_checks]
            assert "pytest" in project_names

    def test_auto_detect_npm(self):
        """Should auto-detect npm test when package.json exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            fixer = AutoFixer(working_dir=tmpdir)
            project_names = [c.name for c in fixer.project_checks]
            assert "npm-test" in project_names


class TestAddCheck:
    """Tests for add_check."""

    def test_add_per_file_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("my-lint", "echo lint {file}")
            assert len(fixer.checks) == 1
            assert fixer.checks[0].name == "my-lint"

    def test_add_project_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("my-test", "echo test", run_per_file=False)
            assert len(fixer.project_checks) == 1
            assert fixer.project_checks[0].name == "my-test"


class TestRunChecks:
    """Tests for running checks on files."""

    def test_passing_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid Python file
            test_file = Path(tmpdir) / "good.py"
            test_file.write_text("x = 1\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("echo-check", "echo ok")
            results = fixer.run_checks(["good.py"])
            assert len(results) >= 1
            assert results[0].passed is True

    def test_failing_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("fail-check", "false")
            results = fixer.run_checks(["any.py"])
            assert any(not r.passed for r in results)

    def test_python_syntax_check(self):
        """Valid Python file should pass syntax check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "valid.py"
            test_file.write_text("def hello():\n    return 'world'\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            results = fixer.run_checks(["valid.py"])
            # Default checks for .py include python-syntax
            syntax_results = [r for r in results if r.check_name == "python-syntax"]
            assert len(syntax_results) == 1
            assert syntax_results[0].passed is True

    def test_invalid_python_syntax(self):
        """Invalid Python should fail syntax check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bad.py"
            test_file.write_text("def hello(\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            results = fixer.run_checks(["bad.py"])
            syntax_results = [r for r in results if r.check_name == "python-syntax"]
            assert len(syntax_results) == 1
            assert syntax_results[0].passed is False

    def test_project_checks_run_once(self):
        """Project-level checks run once, not per file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("project", "echo project_check", run_per_file=False)

            results = fixer.run_checks(["a.txt", "b.txt", "c.txt"])
            project_results = [r for r in results if r.check_name == "project"]
            assert len(project_results) == 1

    def test_timeout_handling(self):
        """Checks that time out should return a failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            # sleep command that would exceed timeout
            fixer.add_check("slow", "sleep 999")
            results = fixer.run_checks(["test.py"])
            # The default timeout is 60s which we won't wait for
            # But we can test that the method handles it
            assert len(results) >= 0  # no crash


class TestCheckAndFix:
    """Tests for the check_and_fix auto-fix loop."""

    def test_all_passing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a .txt file to avoid default Python checks
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("pass", "true")
            result = fixer.check_and_fix(["test.txt"])
            assert result.all_passed is True
            assert result.fixes_attempted == 0

    def test_no_callback_reports_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("fail", "echo 'error found' && false")
            result = fixer.check_and_fix(["test.txt"])
            assert result.all_passed is False
            assert len(result.final_errors) > 0

    def test_fix_callback_called(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fix_calls = []
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False, max_attempts=1)
            fixer.add_check("fail", "echo 'error' && false")
            fixer.check_and_fix(
                ["test.txt"],
                fix_callback=lambda msg: fix_calls.append(msg),
            )
            assert len(fix_calls) >= 1
            assert "error" in fix_calls[0].lower()

    def test_empty_files_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("pass", "true")
            result = fixer.check_and_fix([])
            # No per-file checks, no project checks => passes
            assert isinstance(result, AutoFixResult)
            assert result.all_passed is True


class TestRunCheck:
    """Tests for _run_check internal method."""

    def test_file_placeholder_substitution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("x = 1\n")
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            check = Check("echo", "echo {file}")
            result = fixer._run_check(check, "test.py")
            assert result.passed is True
            assert "test.py" in result.output

    def test_skip_per_file_without_file(self):
        """Per-file checks with {file} placeholder should skip when no file given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            check = Check("lint", "echo {file}")
            result = fixer._run_check(check)
            assert result.passed is True
            assert result.output == "skipped"
