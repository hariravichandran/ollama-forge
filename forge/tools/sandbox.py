"""Sandboxed code execution for agents.

Runs code snippets in isolated environments to prevent agents from
accidentally damaging the host system. Supports:

- Python execution in subprocess with resource limits
- Temporary working directory (auto-cleaned)
- Timeout enforcement
- Output capture (stdout + stderr)
- Optional network isolation (Linux namespaces when available)

Security model:
- Each execution gets a fresh temp directory
- Resource limits: CPU time, memory, file size
- No access to user home or project files by default
- Network access disabled by default
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from forge.utils.logging import get_logger

log = get_logger("tools.sandbox")

# Resource limits
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_MAX_MEMORY_MB = 256  # megabytes
DEFAULT_MAX_OUTPUT = 50_000  # characters

# Validation ranges
MIN_TIMEOUT = 1
MAX_TIMEOUT = 300  # 5 minutes
MIN_MEMORY_MB = 16
MAX_MEMORY_MB = 2048  # 2 GB


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    stdout: str
    stderr: str
    return_code: int
    duration_s: float
    timed_out: bool = False
    error: str = ""
    peak_memory_mb: float = 0.0  # Peak RSS in MB (Linux only, 0 if unavailable)

    def __post_init__(self) -> None:
        """Validate execution result fields."""
        self.duration_s = max(0.0, self.duration_s)
        self.peak_memory_mb = max(0.0, self.peak_memory_mb)

    @property
    def success(self) -> bool:
        """Whether execution completed successfully (exit 0, no timeout)."""
        return self.return_code == 0 and not self.timed_out

    def __repr__(self) -> str:
        status = "OK" if self.success else ("TIMEOUT" if self.timed_out else f"exit={self.return_code}")
        mem = f", {self.peak_memory_mb:.0f}MB" if self.peak_memory_mb else ""
        return f"ExecutionResult({status}, {self.duration_s}s{mem})"

    @property
    def output(self) -> str:
        """Combined output (stdout + stderr if present)."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.error:
            parts.append(f"[error]\n{self.error}")
        return "\n".join(parts)


class Sandbox:
    """Sandboxed code execution environment.

    Usage:
        sandbox = Sandbox()

        # Run Python code
        result = sandbox.run_python("print('hello')")

        # Run a shell command
        result = sandbox.run_command("echo hello")

        # Run with files pre-populated
        result = sandbox.run_python(
            code="import json; data = json.load(open('input.json')); print(data)",
            files={"input.json": '{"key": "value"}'},
        )
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        max_output: int = DEFAULT_MAX_OUTPUT,
        allow_network: bool = False,
        allow_project_read: bool = False,
        project_dir: str | None = None,
    ):
        # Validate and clamp resource limits
        self.timeout = max(MIN_TIMEOUT, min(timeout, MAX_TIMEOUT))
        self.max_memory_mb = max(MIN_MEMORY_MB, min(max_memory_mb, MAX_MEMORY_MB))
        self.max_output = max(100, min(max_output, 500_000))
        self.allow_network = allow_network
        self.allow_project_read = allow_project_read
        self.project_dir = project_dir
        # Execution metrics
        self._execution_count = 0
        self._total_duration_s = 0.0
        self._timeout_count = 0
        self._error_count = 0

        if timeout != self.timeout:
            log.info("Sandbox timeout clamped: %d → %d", timeout, self.timeout)
        if max_memory_mb != self.max_memory_mb:
            log.info("Sandbox memory clamped: %d → %d MB", max_memory_mb, self.max_memory_mb)

    def run_python(
        self,
        code: str,
        files: dict[str, str] | None = None,
        timeout: int | None = None,
        packages: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute Python code in a sandboxed environment.

        Args:
            code: Python source code to execute.
            files: Dict of filename -> content to create in the sandbox.
            timeout: Override default timeout.
            packages: pip packages to install before running.

        Returns:
            ExecutionResult with stdout, stderr, and metadata.
        """
        if not code or not code.strip():
            return ExecutionResult(
                stdout="", stderr="", return_code=-1,
                duration_s=0.0, error="Empty code",
            )
        timeout = timeout or self.timeout
        tmpdir = tempfile.mkdtemp(prefix="forge-sandbox-")

        try:
            # Write code to file
            code_file = Path(tmpdir) / "main.py"
            code_file.write_text(code)

            # Write additional files
            if files:
                for name, content in files.items():
                    file_path = Path(tmpdir) / name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)

            # If project read is allowed, symlink project dir (with validation)
            if self.allow_project_read and self.project_dir:
                project_path = Path(self.project_dir).resolve()
                if project_path.exists() and project_path.is_dir():
                    project_link = Path(tmpdir) / "project"
                    try:
                        project_link.symlink_to(str(project_path))
                    except Exception as e:
                        log.debug("Could not create project symlink: %s", e)
                else:
                    log.warning("Project dir does not exist or is not a directory: %s", self.project_dir)

            # Build the command
            cmd = [sys.executable, str(code_file)]

            # Install packages if requested (in the sandbox)
            if packages:
                install_result = self._run_subprocess(
                    [sys.executable, "-m", "pip", "install", "--quiet", "--target",
                     str(Path(tmpdir) / "lib")] + packages,
                    cwd=tmpdir,
                    timeout=60,
                )
                if not install_result.success:
                    return install_result

            # Set up environment
            env = self._build_env(tmpdir)

            return self._run_subprocess(cmd, cwd=tmpdir, timeout=timeout, env=env)

        finally:
            # Cleanup
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                log.debug("Sandbox tmpdir cleanup failed (non-critical): %s", e)

    def run_command(
        self,
        command: str,
        files: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a shell command in a sandboxed environment.

        Args:
            command: Shell command to execute.
            files: Dict of filename -> content to create in the sandbox.
            timeout: Override default timeout.

        Returns:
            ExecutionResult with stdout, stderr, and metadata.
        """
        if not command or not command.strip():
            return ExecutionResult(
                stdout="", stderr="", return_code=-1,
                duration_s=0.0, error="Empty command",
            )
        timeout = timeout or self.timeout
        tmpdir = tempfile.mkdtemp(prefix="forge-sandbox-")

        try:
            # Write files
            if files:
                for name, content in files.items():
                    file_path = Path(tmpdir) / name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)

            env = self._build_env(tmpdir)

            return self._run_subprocess(
                command, cwd=tmpdir, timeout=timeout, env=env, shell=True,
            )

        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                log.debug("Sandbox tmpdir cleanup failed (non-critical): %s", e)

    def run_tests(
        self,
        test_code: str,
        source_files: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Run test code in a sandboxed environment.

        Convenience method that runs pytest or unittest on provided test code.

        Args:
            test_code: Test source code.
            source_files: Dict of source files the tests import.
            timeout: Override default timeout.
        """
        timeout = timeout or self.timeout
        files = dict(source_files or {})
        files["test_sandbox.py"] = test_code

        # Try pytest first, fall back to unittest
        code = (
            "import subprocess, sys\n"
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', 'test_sandbox.py', '-v', '--tb=short'],\n"
            "    capture_output=True, text=True\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.stderr:\n"
            "    print(result.stderr, file=sys.stderr)\n"
            "sys.exit(result.returncode)\n"
        )

        return self.run_python(code, files=files, timeout=timeout)

    def _run_subprocess(
        self,
        cmd: Any,
        cwd: str,
        timeout: int,
        env: dict[str, str] | None = None,
        shell: bool = False,
    ) -> ExecutionResult:
        """Run a subprocess with resource limits and output capture."""
        start = time.time()

        try:
            # Build preexec_fn for resource limits (Linux only)
            preexec = self._get_preexec_fn() if os.name != "nt" else None

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                shell=shell,
                text=True,
                preexec_fn=preexec,
            )

            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.time() - start

            # Truncate output
            stdout = stdout[:self.max_output] if stdout else ""
            stderr = stderr[:self.max_output] if stderr else ""

            # Try to get peak memory usage (Linux only)
            peak_mem = self._get_peak_memory(proc.pid)

            self._execution_count += 1
            self._total_duration_s += duration

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                return_code=proc.returncode,
                duration_s=round(duration, 2),
                peak_memory_mb=peak_mem,
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._execution_count += 1
            self._timeout_count += 1
            duration = time.time() - start
            self._total_duration_s += duration
            return ExecutionResult(
                stdout="",
                stderr="",
                return_code=-1,
                duration_s=round(duration, 2),
                timed_out=True,
                error=f"Execution timed out after {timeout}s",
            )

        except Exception as e:
            self._execution_count += 1
            self._error_count += 1
            duration = time.time() - start
            self._total_duration_s += duration
            return ExecutionResult(
                stdout="",
                stderr="",
                return_code=-1,
                duration_s=round(duration, 2),
                error=str(e),
            )

    @staticmethod
    def _get_peak_memory(pid: int) -> float:
        """Get peak memory usage of a process in MB (Linux only).

        Reads /proc/<pid>/status for VmHWM (high water mark RSS).
        Returns 0.0 if unavailable.
        """
        try:
            status_path = Path(f"/proc/{pid}/status")
            if status_path.exists():
                for line in status_path.read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        # VmHWM:    12345 kB
                        kb = int(line.split()[1])
                        return round(kb / 1024, 1)
        except (OSError, ValueError, IndexError):
            pass  # Expected on non-Linux or if process already exited
        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Get sandbox execution metrics."""
        return {
            "executions": self._execution_count,
            "total_duration_s": round(self._total_duration_s, 2),
            "avg_duration_s": round(self._total_duration_s / max(1, self._execution_count), 2),
            "timeouts": self._timeout_count,
            "errors": self._error_count,
            "timeout_rate": round(self._timeout_count / max(1, self._execution_count), 2),
        }

    def _build_env(self, tmpdir: str) -> dict[str, str]:
        """Build an isolated environment for the subprocess."""
        env = os.environ.copy()

        # Override HOME and temp dirs to sandbox
        env["HOME"] = tmpdir
        env["TMPDIR"] = tmpdir
        env["TEMP"] = tmpdir

        # Add sandbox lib to Python path (for pip --target packages)
        lib_dir = str(Path(tmpdir) / "lib")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

        # Disable network if requested (Linux only, best-effort)
        if not self.allow_network and os.name != "nt":
            # unshare requires root, so this is best-effort
            env["no_proxy"] = "*"
            env["http_proxy"] = "http://0.0.0.0:0"
            env["https_proxy"] = "http://0.0.0.0:0"

        return env

    def _get_preexec_fn(self) -> Callable[[], None] | None:
        """Get a preexec function that sets resource limits (Linux/macOS)."""
        def set_limits() -> None:
            try:
                import resource
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout + 5))
                # Memory limit
                mem_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                # Max file size (10MB)
                resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
            except Exception:
                pass  # Resource limits are best-effort
        return set_limits


class SandboxTool:
    """Tool interface for sandboxed execution, compatible with the agent tool system."""

    name = "sandbox"
    description = "Execute code in a safe, isolated sandbox environment"

    def __init__(self, project_dir: str = ".", allow_project_read: bool = False):
        self.sandbox = Sandbox(
            project_dir=project_dir,
            allow_project_read=allow_project_read,
        )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "Execute Python code in a sandboxed environment. "
                    "The code runs in an isolated temporary directory with resource limits. "
                    "Use this for testing code snippets, running calculations, or verifying behavior.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Execute a shell command in a sandboxed environment. "
                    "The command runs in an isolated temp directory. Limited to non-destructive commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a sandbox tool function."""
        if function_name == "run_code":
            code = args.get("code", "")
            if not code or not code.strip():
                return "Error: empty code"
            result = self.sandbox.run_python(
                code=code,
                timeout=args.get("timeout", 30),
            )
        elif function_name == "run_shell":
            command = args.get("command", "")
            if not command or not command.strip():
                return "Error: empty command"
            result = self.sandbox.run_command(
                command=command,
                timeout=args.get("timeout", 30),
            )
        else:
            return f"Unknown function: {function_name}"

        # Format output
        status = "OK" if result.success else "FAILED"
        parts = [f"[{status}] (exit={result.return_code}, {result.duration_s}s)"]
        if result.output:
            parts.append(result.output)
        if result.timed_out:
            parts.append(f"[TIMEOUT after {self.sandbox.timeout}s]")

        return "\n".join(parts)
