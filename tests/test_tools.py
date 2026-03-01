"""Tests for built-in tools."""

import tempfile
from pathlib import Path

import pytest

from forge.tools.filesystem import FilesystemTool
from forge.tools.shell import ShellTool
from forge.tools.git import GitTool


class TestFilesystemTool:
    """Tests for the filesystem tool."""

    def test_tool_definitions(self):
        tool = FilesystemTool()
        defs = tool.get_tool_definitions()
        assert len(defs) >= 4
        names = [d["function"]["name"] for d in defs]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "list_files" in names

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)

            # Write
            result = tool.execute("write_file", {"path": "test.txt", "content": "Hello world"})
            assert "Written" in result

            # Read
            result = tool.execute("read_file", {"path": "test.txt"})
            assert "Hello world" in result

    def test_edit_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)

            # Create file
            (Path(tmpdir) / "test.py").write_text("x = 1\ny = 2\n")

            # Edit
            result = tool.execute("edit_file", {
                "path": "test.py",
                "old_string": "x = 1",
                "new_string": "x = 42",
            })
            assert "Replaced" in result

            # Verify
            content = (Path(tmpdir) / "test.py").read_text()
            assert "x = 42" in content

    def test_edit_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool.execute("edit_file", {
                "path": "nonexistent.txt",
                "old_string": "x",
                "new_string": "y",
            })
            assert "not found" in result.lower()

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("# a")
            (Path(tmpdir) / "b.py").write_text("# b")
            (Path(tmpdir) / "c.txt").write_text("c")

            tool = FilesystemTool(working_dir=tmpdir)
            result = tool.execute("list_files", {"pattern": "*.py"})
            assert "a.py" in result
            assert "b.py" in result
            assert "c.txt" not in result

    def test_search_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello():\n    return 42\n")

            tool = FilesystemTool(working_dir=tmpdir)
            result = tool.execute("search_files", {"pattern": "def hello"})
            assert "test.py" in result

    def test_directory_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool.execute("read_file", {"path": "../../../etc/passwd"})
            assert "Error" in result or "escapes" in result.lower()


class TestShellTool:
    """Tests for the shell tool."""

    def test_run_command(self):
        tool = ShellTool()
        result = tool.execute("run_command", {"command": "echo hello"})
        assert "hello" in result

    def test_dangerous_command_blocked(self):
        tool = ShellTool()
        result = tool.execute("run_command", {"command": "rm -rf /"})
        assert "Blocked" in result

    def test_timeout(self):
        tool = ShellTool()
        result = tool.execute("run_command", {"command": "sleep 10", "timeout": 1})
        assert "timed out" in result.lower()

    def test_nonexistent_command(self):
        tool = ShellTool()
        result = tool.execute("run_command", {"command": "nonexistent_command_xyz"})
        # Should return error info, not crash
        assert isinstance(result, str)
