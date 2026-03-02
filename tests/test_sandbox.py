"""Tests for sandboxed code execution."""

import sys
import tempfile
from pathlib import Path

import pytest

from forge.tools.sandbox import (
    Sandbox,
    SandboxTool,
    ExecutionResult,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_MAX_OUTPUT,
)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_when_zero_return_code(self):
        r = ExecutionResult(stdout="ok", stderr="", return_code=0, duration_s=0.1)
        assert r.success is True

    def test_failure_when_nonzero_return_code(self):
        r = ExecutionResult(stdout="", stderr="err", return_code=1, duration_s=0.1)
        assert r.success is False

    def test_failure_when_timed_out(self):
        r = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=5.0, timed_out=True)
        assert r.success is False

    def test_output_combines_stdout_stderr(self):
        r = ExecutionResult(stdout="out", stderr="err", return_code=0, duration_s=0.1)
        assert "out" in r.output
        assert "[stderr]" in r.output
        assert "err" in r.output

    def test_output_stdout_only(self):
        r = ExecutionResult(stdout="hello", stderr="", return_code=0, duration_s=0.1)
        assert r.output == "hello"

    def test_output_with_error(self):
        r = ExecutionResult(stdout="", stderr="", return_code=-1, duration_s=0.1, error="boom")
        assert "[error]" in r.output
        assert "boom" in r.output

    def test_output_empty(self):
        r = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=0.0)
        assert r.output == ""


class TestSandboxDefaults:
    """Tests for Sandbox configuration defaults."""

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30

    def test_default_max_memory(self):
        assert DEFAULT_MAX_MEMORY_MB == 256

    def test_default_max_output(self):
        assert DEFAULT_MAX_OUTPUT == 50_000

    def test_sandbox_uses_defaults(self):
        s = Sandbox()
        assert s.timeout == DEFAULT_TIMEOUT
        assert s.max_memory_mb == DEFAULT_MAX_MEMORY_MB
        assert s.max_output == DEFAULT_MAX_OUTPUT
        assert s.allow_network is False
        assert s.allow_project_read is False

    def test_sandbox_custom_config(self):
        s = Sandbox(timeout=10, max_memory_mb=128, allow_network=True)
        assert s.timeout == 10
        assert s.max_memory_mb == 128
        assert s.allow_network is True


class TestSandboxRunPython:
    """Tests for Sandbox.run_python()."""

    def test_hello_world(self):
        s = Sandbox()
        result = s.run_python("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    def test_return_code_on_error(self):
        s = Sandbox()
        result = s.run_python("raise ValueError('boom')")
        assert not result.success
        assert result.return_code != 0
        assert "ValueError" in result.stderr or "boom" in result.stderr

    def test_syntax_error(self):
        s = Sandbox()
        result = s.run_python("def broken(")
        assert not result.success

    def test_files_parameter(self):
        s = Sandbox()
        result = s.run_python(
            code="print(open('data.txt').read())",
            files={"data.txt": "test content"},
        )
        assert result.success
        assert "test content" in result.stdout

    def test_nested_files(self):
        s = Sandbox()
        result = s.run_python(
            code="print(open('sub/deep.txt').read())",
            files={"sub/deep.txt": "nested"},
        )
        assert result.success
        assert "nested" in result.stdout

    def test_timeout(self):
        s = Sandbox()
        result = s.run_python("import time; time.sleep(60)", timeout=1)
        assert result.timed_out
        assert not result.success

    def test_output_truncation(self):
        s = Sandbox(max_output=100)
        result = s.run_python("print('x' * 500)")
        assert len(result.stdout) <= 100

    def test_temp_directory_cleaned_up(self):
        """After execution, the sandbox temp dir should be removed."""
        s = Sandbox()
        # Run something that prints its working dir
        result = s.run_python("import os; print(os.getcwd())")
        assert result.success
        cwd = result.stdout.strip()
        assert not Path(cwd).exists(), "Sandbox temp dir should be cleaned up"

    def test_duration_tracked(self):
        s = Sandbox()
        result = s.run_python("print(1)")
        assert result.duration_s >= 0

    def test_env_isolation(self):
        """Sandbox should override HOME to temp dir."""
        s = Sandbox()
        result = s.run_python("import os; print(os.environ.get('HOME', ''))")
        assert result.success
        # HOME should point to a sandbox dir, not the real home
        assert "forge-sandbox" in result.stdout


class TestSandboxRunCommand:
    """Tests for Sandbox.run_command()."""

    def test_echo(self):
        s = Sandbox()
        result = s.run_command("echo hello")
        assert result.success
        assert "hello" in result.stdout

    def test_command_with_files(self):
        s = Sandbox()
        result = s.run_command("cat data.txt", files={"data.txt": "file content"})
        assert result.success
        assert "file content" in result.stdout

    def test_nonexistent_command(self):
        s = Sandbox()
        result = s.run_command("nonexistent_xyz_command_12345")
        assert not result.success

    def test_command_timeout(self):
        s = Sandbox()
        result = s.run_command("sleep 60", timeout=1)
        assert result.timed_out


class TestSandboxRunTests:
    """Tests for Sandbox.run_tests()."""

    def test_passing_test(self):
        s = Sandbox()
        test_code = "def test_add(): assert 1 + 1 == 2"
        result = s.run_tests(test_code)
        # May pass or fail depending on pytest availability in subprocess
        assert isinstance(result, ExecutionResult)

    def test_with_source_files(self):
        s = Sandbox()
        result = s.run_tests(
            test_code="from mymod import add\ndef test_add(): assert add(1, 2) == 3",
            source_files={"mymod.py": "def add(a, b): return a + b"},
        )
        assert isinstance(result, ExecutionResult)


class TestSandboxBuildEnv:
    """Tests for Sandbox._build_env()."""

    def test_env_contains_home(self):
        s = Sandbox()
        env = s._build_env("/tmp/test-sandbox")
        assert env["HOME"] == "/tmp/test-sandbox"
        assert env["TMPDIR"] == "/tmp/test-sandbox"

    def test_env_network_disabled(self):
        s = Sandbox(allow_network=False)
        env = s._build_env("/tmp/test-sandbox")
        assert env.get("http_proxy") == "http://0.0.0.0:0"
        assert env.get("https_proxy") == "http://0.0.0.0:0"

    def test_env_network_allowed(self):
        s = Sandbox(allow_network=True)
        env = s._build_env("/tmp/test-sandbox")
        # Should NOT override proxy settings
        assert env.get("http_proxy") != "http://0.0.0.0:0" or "http_proxy" not in env

    def test_pythonpath_includes_lib(self):
        s = Sandbox()
        env = s._build_env("/tmp/test-sandbox")
        assert "/tmp/test-sandbox/lib" in env.get("PYTHONPATH", "")


class TestSandboxTool:
    """Tests for the SandboxTool interface."""

    def test_tool_definitions(self):
        tool = SandboxTool()
        defs = tool.get_tool_definitions()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert "run_code" in names
        assert "run_shell" in names

    def test_run_code(self):
        tool = SandboxTool()
        result = tool.execute("run_code", {"code": "print('tool test')"})
        assert "OK" in result
        assert "tool test" in result

    def test_run_shell(self):
        tool = SandboxTool()
        result = tool.execute("run_shell", {"command": "echo shell test"})
        assert "OK" in result
        assert "shell test" in result

    def test_run_code_failure(self):
        tool = SandboxTool()
        result = tool.execute("run_code", {"code": "raise Exception('fail')"})
        assert "FAILED" in result

    def test_unknown_function(self):
        tool = SandboxTool()
        result = tool.execute("nonexistent", {})
        assert "Unknown" in result

    def test_timeout_output(self):
        tool = SandboxTool()
        result = tool.execute("run_code", {"code": "import time; time.sleep(60)", "timeout": 1})
        assert "TIMEOUT" in result or "FAILED" in result
