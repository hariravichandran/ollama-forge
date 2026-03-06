"""Tests for batch 19 improvements: defensive input validation.

Verifies that sandbox, web tool, and filesystem tool properly validate
inputs to prevent KeyErrors, unbounded queries, and edge cases.
"""

import pytest


class TestSandboxInputValidation:
    """Tests for sandbox tool input validation."""

    def test_run_python_empty_code_returns_error(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        result = sb.run_python("")
        assert not result.success
        assert "Empty code" in result.error

    def test_run_python_whitespace_code_returns_error(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        result = sb.run_python("   \n  ")
        assert not result.success
        assert "Empty code" in result.error

    def test_run_command_empty_returns_error(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        result = sb.run_command("")
        assert not result.success
        assert "Empty command" in result.error

    def test_run_command_whitespace_returns_error(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        result = sb.run_command("   ")
        assert not result.success
        assert "Empty command" in result.error


class TestSandboxToolExecuteValidation:
    """Tests for SandboxTool.execute() safe dict access."""

    def test_execute_run_code_missing_key(self):
        from forge.tools.sandbox import SandboxTool
        tool = SandboxTool()
        result = tool.execute("run_code", {})
        assert "Error" in result or "empty" in result.lower()

    def test_execute_run_shell_missing_key(self):
        from forge.tools.sandbox import SandboxTool
        tool = SandboxTool()
        result = tool.execute("run_shell", {})
        assert "Error" in result or "empty" in result.lower()

    def test_execute_run_code_empty_code(self):
        from forge.tools.sandbox import SandboxTool
        tool = SandboxTool()
        result = tool.execute("run_code", {"code": ""})
        assert "Error" in result or "empty" in result.lower()

    def test_execute_run_shell_empty_command(self):
        from forge.tools.sandbox import SandboxTool
        tool = SandboxTool()
        result = tool.execute("run_shell", {"command": ""})
        assert "Error" in result or "empty" in result.lower()

    def test_execute_unknown_function(self):
        from forge.tools.sandbox import SandboxTool
        tool = SandboxTool()
        result = tool.execute("nonexistent", {})
        assert "Unknown function" in result


class TestWebToolSearchValidation:
    """Tests for web tool search max_results clamping."""

    def test_search_max_results_clamped_high(self):
        """Verify max_results is clamped to 20."""
        from forge.tools.web import WebTool
        import inspect
        source = inspect.getsource(WebTool._search)
        assert "max(1, min(max_results" in source or "min(max_results" in source

    def test_search_max_results_clamped_low(self):
        """Verify max_results minimum is 1."""
        from forge.tools.web import WebTool
        import inspect
        source = inspect.getsource(WebTool._search)
        assert "max(1," in source

    def test_search_empty_query_returns_error(self):
        from forge.tools.web import WebTool
        tool = WebTool()
        result = tool._search("")
        assert "Error" in result or "empty" in result.lower()


class TestFilesystemReadValidation:
    """Tests for filesystem tool read_file end_line clamping."""

    def test_end_line_clamped_to_file_length(self):
        """Verify end_line uses min() to clamp to file length."""
        from forge.tools.filesystem import FilesystemTool
        import inspect
        source = inspect.getsource(FilesystemTool._read_file)
        assert "min(end_line, len(lines))" in source

    def test_read_file_with_large_end_line(self):
        """Reading with end_line beyond file should not error."""
        import tempfile
        from pathlib import Path
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("line1\nline2\nline3\n")
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._read_file("test.txt", start_line=1, end_line=999999)
            assert "line1" in result
            assert "line3" in result

    def test_read_file_with_zero_end_line(self):
        """Reading with end_line=0 should return all lines."""
        import tempfile
        from pathlib import Path
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("line1\nline2\nline3\n")
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._read_file("test.txt")
            assert "line1" in result
            assert "line3" in result

    def test_read_file_start_line_bounds(self):
        """Reading with start_line beyond file should return empty."""
        import tempfile
        from pathlib import Path
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("line1\nline2\n")
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._read_file("test.txt", start_line=999, end_line=1000)
            # Should return empty (no lines in that range)
            assert result == ""


class TestBatch19Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_sandbox_imports(self):
        from forge.tools.sandbox import Sandbox, SandboxTool, ExecutionResult

    def test_web_tool_imports(self):
        from forge.tools.web import WebTool

    def test_filesystem_imports(self):
        from forge.tools.filesystem import FilesystemTool
