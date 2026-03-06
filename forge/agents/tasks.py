"""Background task execution: run long-running tasks asynchronously.

Enables agents to kick off tasks (test suites, builds, deployments, research)
that run in the background while the user continues chatting. When a task
completes, the agent reports results.

This eliminates blocking on slow operations. The agent can run tests,
wait for results, fix failures, and re-run — all without holding up
the conversation.

Usage:
    tasks = TaskManager()

    # Submit a background task
    task_id = tasks.submit("Run test suite", "python -m pytest tests/ -v")

    # Check status
    status = tasks.get_status(task_id)

    # Get results when done
    if status.done:
        print(status.output)
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from forge.utils.logging import get_logger

log = get_logger("agents.tasks")

# Task limits
DEFAULT_TASK_TIMEOUT = 600  # seconds
MIN_TASK_TIMEOUT = 1
MAX_TASK_TIMEOUT = 7200  # 2 hours
DEFAULT_MAX_CONCURRENT = 5
MIN_MAX_CONCURRENT = 1
MAX_MAX_CONCURRENT = 50


class TaskStatus(Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a background task."""

    task_id: str
    name: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    return_code: int = -1

    @property
    def done(self) -> bool:
        """Whether the task has finished (completed, failed, or cancelled)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds (live if running, final if done)."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0

    def __repr__(self) -> str:
        elapsed = f", {self.elapsed_s:.1f}s" if self.started_at else ""
        return f"TaskResult({self.task_id}, {self.status.value}{elapsed})"


class TaskManager:
    """Manages background tasks for agents.

    Tasks run in separate threads and can be checked/cancelled.

    Usage:
        tm = TaskManager()

        # Submit a shell command
        task_id = tm.submit("tests", "python -m pytest tests/")

        # Submit a Python callable
        task_id = tm.submit_callable("research", my_research_fn, args=("topic",))

        # Check status
        result = tm.get_status(task_id)

        # List active tasks
        active = tm.list_tasks(status=TaskStatus.RUNNING)

        # Cancel a task
        tm.cancel(task_id)
    """

    def __init__(self, working_dir: str = ".", max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self.working_dir = working_dir
        self.max_concurrent = min(max(max_concurrent, MIN_MAX_CONCURRENT), MAX_MAX_CONCURRENT)
        self._tasks: dict[str, TaskResult] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        name: str,
        command: str,
        timeout: int = DEFAULT_TASK_TIMEOUT,
        callback: Callable[[TaskResult], None] | None = None,
    ) -> str:
        """Submit a shell command as a background task.

        Returns the task_id.
        """
        if not command or not command.strip():
            task_id = f"task-{uuid.uuid4().hex[:8]}"
            result = TaskResult(task_id=task_id, name=name, status=TaskStatus.FAILED)
            result.error = "Empty command"
            with self._lock:
                self._tasks[task_id] = result
            return task_id

        timeout = min(max(timeout, MIN_TASK_TIMEOUT), MAX_TASK_TIMEOUT)
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        result = TaskResult(
            task_id=task_id,
            name=name,
            status=TaskStatus.PENDING,
        )

        with self._lock:
            # Check concurrent limit
            active = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
            if active >= self.max_concurrent:
                result.status = TaskStatus.FAILED
                result.error = f"Too many concurrent tasks ({active}/{self.max_concurrent})"
                self._tasks[task_id] = result
                return task_id

            self._tasks[task_id] = result

        # Run in background thread
        thread = threading.Thread(
            target=self._run_command,
            args=(task_id, command, timeout, callback),
            daemon=True,
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()

        log.info("Submitted task %s: %s", task_id, name)
        return task_id

    def submit_callable(
        self,
        name: str,
        fn: Callable[..., str],
        args: tuple = (),
        kwargs: dict | None = None,
        callback: Callable[[TaskResult], None] | None = None,
    ) -> str:
        """Submit a Python callable as a background task.

        The callable should return a string (the output).
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        result = TaskResult(
            task_id=task_id,
            name=name,
            status=TaskStatus.PENDING,
        )

        with self._lock:
            self._tasks[task_id] = result

        thread = threading.Thread(
            target=self._run_callable,
            args=(task_id, fn, args, kwargs or {}, callback),
            daemon=True,
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()

        return task_id

    def get_status(self, task_id: str) -> TaskResult | None:
        """Get the current status of a task."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskResult]:
        """List tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.started_at or 0, reverse=True)

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status != TaskStatus.RUNNING:
                return False
            proc = self._processes.get(task_id)

        # Kill the subprocess if it exists (outside lock to avoid holding lock during I/O)
        if proc:
            try:
                proc.kill()
            except Exception as e:
                log.debug("Could not kill process for task %s: %s", task_id, e)

        with self._lock:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
        log.info("Cancelled task %s", task_id)
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shut down all running tasks and join threads.

        Cancels running tasks and waits for threads to finish.

        Args:
            timeout: Maximum seconds to wait for each thread to join.
        """
        # Cancel all running tasks
        with self._lock:
            running = [tid for tid, t in self._tasks.items() if t.status == TaskStatus.RUNNING]

        for tid in running:
            self.cancel(tid)

        # Join all threads
        for tid, thread in list(self._threads.items()):
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    log.warning("Thread for task %s did not finish within %.0fs", tid, timeout)

        log.info("TaskManager shutdown complete (%d tasks processed)", len(self._tasks))

    def cleanup(self) -> int:
        """Remove completed/failed/cancelled tasks. Returns count removed."""
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.done
            ]
            for tid in to_remove:
                del self._tasks[tid]
                self._threads.pop(tid, None)
                self._processes.pop(tid, None)
            return len(to_remove)

    def _run_command(
        self,
        task_id: str,
        command: str,
        timeout: int,
        callback: Callable | None,
    ) -> None:
        """Execute a shell command in the background."""
        with self._lock:
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
            )
            with self._lock:
                self._processes[task_id] = proc

            stdout, _ = proc.communicate(timeout=timeout)
            with self._lock:
                task.output = stdout or ""
                task.return_code = proc.returncode
                task.status = TaskStatus.COMPLETED if proc.returncode == 0 else TaskStatus.FAILED
                if proc.returncode != 0:
                    task.error = f"Exit code: {proc.returncode}"

        except subprocess.TimeoutExpired:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = f"Timed out after {timeout}s"
                proc = self._processes.get(task_id)
            if proc:
                proc.kill()

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)

        finally:
            with self._lock:
                task.completed_at = time.time()
                self._processes.pop(task_id, None)
            log.info("Task %s %s (%.1fs)", task_id, task.status.value, task.elapsed_s)

            if callback:
                try:
                    callback(task)
                except Exception as e:
                    log.error("Task callback error: %s", e)

    def _run_callable(
        self,
        task_id: str,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        callback: Callable | None,
    ) -> None:
        """Execute a Python callable in the background."""
        with self._lock:
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                task.output = str(result) if result else ""
                task.return_code = 0
                task.status = TaskStatus.COMPLETED

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)

        finally:
            with self._lock:
                task.completed_at = time.time()
            log.info("Task %s %s (%.1fs)", task_id, task.status.value, task.elapsed_s)

            if callback:
                try:
                    callback(task)
                except Exception as e:
                    log.error("Task callback error: %s", e)
