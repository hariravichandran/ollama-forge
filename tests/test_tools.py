"""Tests for built-in tools."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from forge.tools.filesystem import FilesystemTool
from forge.tools.shell import ShellTool
from forge.tools.git import GitTool
from forge.tools.web import WebTool
from forge.tools import BUILTIN_TOOLS


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

    def test_sudo_blocked(self):
        """sudo rm should be blocked."""
        tool = ShellTool()
        assert tool._is_dangerous("sudo rm -rf /tmp/test") is True

    def test_fork_bomb_blocked(self):
        """Fork bomb should be blocked."""
        tool = ShellTool()
        assert tool._is_dangerous(":(){ :|:& };:") is True

    def test_safe_command_not_blocked(self):
        """Normal commands should not be blocked."""
        tool = ShellTool()
        assert tool._is_dangerous("ls -la") is False
        assert tool._is_dangerous("python3 test.py") is False
        assert tool._is_dangerous("git status") is False

    def test_interactive_vim_blocked(self):
        """Interactive editors should be blocked."""
        tool = ShellTool()
        assert tool._is_interactive("vim test.py") is True
        assert tool._is_interactive("nano config.yaml") is True

    def test_interactive_rebase_blocked(self):
        """Interactive git rebase should be blocked."""
        tool = ShellTool()
        assert tool._is_interactive("git rebase -i HEAD~3") is True
        assert tool._is_interactive("git add -i") is True

    def test_interactive_top_blocked(self):
        """Bare top/htop should be blocked."""
        tool = ShellTool()
        assert tool._is_interactive("top") is True
        assert tool._is_interactive("htop") is True

    def test_noninteractive_commands_pass(self):
        """Non-interactive commands should not be blocked."""
        tool = ShellTool()
        assert tool._is_interactive("git status") is False
        assert tool._is_interactive("python3 script.py") is False
        assert tool._is_interactive("ls -la") is False
        assert tool._is_interactive("grep pattern file.txt") is False

    def test_interactive_command_returns_message(self):
        """Interactive commands should return a helpful error."""
        tool = ShellTool()
        result = tool.execute("run_command", {"command": "vim test.py"})
        assert "Blocked" in result
        assert "interactive" in result.lower()

    def test_empty_command(self):
        """Empty command should return error."""
        tool = ShellTool()
        result = tool.execute("run_command", {"command": ""})
        assert "Error" in result or "empty" in result.lower()


class TestGitTool:
    """Tests for the git tool."""

    def test_tool_definitions(self):
        tool = GitTool()
        defs = tool.get_tool_definitions()
        assert len(defs) >= 5
        names = [d["function"]["name"] for d in defs]
        assert "git_status" in names
        assert "git_diff" in names
        assert "git_log" in names
        assert "git_commit" in names
        assert "git_undo" in names

    def test_check_conflicts_clean(self):
        """No conflicts should return empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            tool = GitTool(working_dir=tmpdir)
            result = tool._check_conflicts()
            assert result == ""

    def test_has_uncommitted_changes_empty(self):
        """Fresh repo should have no uncommitted changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            tool = GitTool(working_dir=tmpdir)
            assert tool.has_uncommitted_changes() is False

    def test_has_uncommitted_changes_with_file(self):
        """Repo with new file should have uncommitted changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            (Path(tmpdir) / "test.txt").write_text("hello")
            tool = GitTool(working_dir=tmpdir)
            assert tool.has_uncommitted_changes() is True

    def test_get_current_branch(self):
        """Should return branch name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess
            subprocess.run(["git", "init", "-b", "main"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            (Path(tmpdir) / "init.txt").write_text("init")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            tool = GitTool(working_dir=tmpdir)
            branch = tool.get_current_branch()
            assert branch == "main"

    def test_agent_commit_tag(self):
        """Commits should be tagged with [forge]."""
        from forge.tools.git import AGENT_COMMIT_TAG
        assert AGENT_COMMIT_TAG == "[forge]"


class TestWebTool:
    """Tests for the web tool."""

    def test_tool_definitions(self):
        """Web tool should define search and fetch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            defs = tool.get_tool_definitions()
            assert len(defs) == 2
            names = [d["function"]["name"] for d in defs]
            assert "web_search" in names
            assert "web_fetch" in names

    def test_empty_search_returns_error(self):
        """Empty search query should return error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            result = tool.execute("web_search", {"query": ""})
            assert "error" in result.lower() or "empty" in result.lower()

    def test_empty_fetch_returns_error(self):
        """Empty URL should return error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            result = tool.execute("web_fetch", {"url": ""})
            assert "error" in result.lower() or "empty" in result.lower()

    def test_unknown_function(self):
        """Unknown function name should return error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            result = tool.execute("nonexistent", {})
            assert "Unknown" in result

    def test_cache_ttl(self):
        """Cache TTL should be 6 hours."""
        assert WebTool.CACHE_TTL == 6 * 3600

    def test_cache_set_and_get(self):
        """Cache should store and retrieve values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            tool._set_cached("test_key", "test_value")
            assert tool._get_cached("test_key") == "test_value"

    def test_cache_expired_returns_none(self):
        """Expired cache entries should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            # Set cache with old timestamp
            tool._cache["old_key"] = {"data": "old_value", "ts": time.time() - 999999}
            assert tool._get_cached("old_key") is None

    def test_cache_persists_to_disk(self):
        """Cache should be saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            tool._set_cached("persist_key", "persist_value")

            # Verify file exists
            assert tool.cache_file.exists()

            # Load fresh and check
            data = json.loads(tool.cache_file.read_text())
            assert "persist_key" in data

    def test_html_to_text(self):
        """HTML should be converted to plain text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            html = "<html><body><p>Hello <b>World</b></p><script>evil()</script></body></html>"
            text = tool._html_to_text(html)
            assert "Hello" in text
            assert "World" in text
            assert "evil" not in text
            assert "<" not in text


class TestToolsRegistry:
    """Tests for the tools registry."""

    def test_builtin_tools_complete(self):
        """All built-in tools should be registered."""
        assert "filesystem" in BUILTIN_TOOLS
        assert "shell" in BUILTIN_TOOLS
        assert "git" in BUILTIN_TOOLS
        assert "web" in BUILTIN_TOOLS
        assert "sandbox" in BUILTIN_TOOLS

    def test_all_tools_instantiate(self):
        """All built-in tools should be constructable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, tool_class in BUILTIN_TOOLS.items():
                # SandboxTool uses project_dir, others use working_dir
                if name == "sandbox":
                    tool = tool_class(project_dir=tmpdir)
                else:
                    tool = tool_class(working_dir=tmpdir)
                assert tool is not None
                assert hasattr(tool, "get_tool_definitions")
                assert hasattr(tool, "execute")

    def test_all_tools_have_definitions(self):
        """All tools should return at least one definition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, tool_class in BUILTIN_TOOLS.items():
                if name == "sandbox":
                    tool = tool_class(project_dir=tmpdir)
                else:
                    tool = tool_class(working_dir=tmpdir)
                defs = tool.get_tool_definitions()
                assert len(defs) >= 1, f"{name} has no tool definitions"
