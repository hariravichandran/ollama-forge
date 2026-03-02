"""Tests for project rules: loading .forge-rules files."""

import tempfile
from pathlib import Path

import pytest

from forge.agents.rules import (
    RULES_FILENAMES,
    load_project_rules,
    create_rules_template,
    _find_project_rules,
    _find_rules_in_dir,
    _read_rules_file,
)


class TestRulesFilenames:
    """Tests for the RULES_FILENAMES constant."""

    def test_has_forge_rules(self):
        assert ".forge-rules" in RULES_FILENAMES

    def test_has_forge_md(self):
        assert "FORGE.md" in RULES_FILENAMES

    def test_has_claude_md(self):
        assert "CLAUDE.md" in RULES_FILENAMES

    def test_forge_rules_first(self):
        """forge-rules should be checked first (highest priority)."""
        assert RULES_FILENAMES[0] == ".forge-rules"


class TestReadRulesFile:
    """Tests for _read_rules_file."""

    def test_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_path = Path(tmpdir) / ".forge-rules"
            rules_path.write_text("Use Python 3.10+")
            result = _read_rules_file(rules_path)
            assert result == "Use Python 3.10+"

    def test_returns_empty_for_missing(self):
        result = _read_rules_file(Path("/tmp/nonexistent_forge_rules_file"))
        assert result == ""

    def test_returns_empty_for_blank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_path = Path(tmpdir) / ".forge-rules"
            rules_path.write_text("   \n  \n  ")
            result = _read_rules_file(rules_path)
            assert result == ""

    def test_returns_empty_for_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _read_rules_file(Path(tmpdir))
            assert result == ""


class TestFindRulesInDir:
    """Tests for _find_rules_in_dir."""

    def test_finds_forge_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_path = Path(tmpdir) / ".forge-rules"
            rules_path.write_text("Rule: always test")
            result = _find_rules_in_dir(Path(tmpdir))
            assert result == "Rule: always test"

    def test_finds_forge_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "FORGE.md").write_text("# Forge Rules")
            result = _find_rules_in_dir(Path(tmpdir))
            assert result == "# Forge Rules"

    def test_prefers_forge_rules_over_forge_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("from .forge-rules")
            (Path(tmpdir) / "FORGE.md").write_text("from FORGE.md")
            result = _find_rules_in_dir(Path(tmpdir))
            assert result == "from .forge-rules"

    def test_returns_empty_for_no_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_rules_in_dir(Path(tmpdir))
            assert result == ""


class TestFindProjectRules:
    """Tests for _find_project_rules (walks up directories)."""

    def test_finds_in_current_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("Project rules here")
            rules, project_dir = _find_project_rules(tmpdir)
            assert rules == "Project rules here"
            assert project_dir == Path(tmpdir).resolve()

    def test_finds_in_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            child = parent / "src" / "lib"
            child.mkdir(parents=True)
            (parent / ".forge-rules").write_text("Root level rules")
            rules, project_dir = _find_project_rules(str(child))
            assert rules == "Root level rules"

    def test_returns_empty_when_no_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules, project_dir = _find_project_rules(tmpdir)
            assert rules == ""
            assert project_dir is None


class TestLoadProjectRules:
    """Tests for load_project_rules (the main entry point)."""

    def test_loads_project_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("Always use pytest")
            result = load_project_rules(tmpdir)
            assert "Always use pytest" in result
            assert "Project Rules" in result

    def test_empty_when_no_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_project_rules(tmpdir)
            assert result == ""

    def test_nearest_rules_wins(self):
        """When both child and parent have rules, the nearest (child) is found as project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            child = parent / "src"
            child.mkdir()
            (parent / ".forge-rules").write_text("Parent level rules")
            (child / ".forge-rules").write_text("Child level rules")
            # _find_project_rules starts at child and finds its rules first
            result = load_project_rules(str(child))
            assert "Child level rules" in result


class TestCreateRulesTemplate:
    """Tests for create_rules_template."""

    def test_creates_template_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_rules_template(tmpdir)
            assert "Created" in result
            rules_path = Path(tmpdir) / ".forge-rules"
            assert rules_path.exists()
            content = rules_path.read_text()
            assert "# .forge-rules" in content
            assert "Coding Style" in content
            assert "Testing" in content

    def test_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("Existing rules")
            result = create_rules_template(tmpdir)
            assert "already exists" in result
            # Original content preserved
            content = (Path(tmpdir) / ".forge-rules").read_text()
            assert content == "Existing rules"
