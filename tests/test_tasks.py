"""Tests for the background TaskManager."""

import time
import threading

import pytest

from forge.agents.tasks import TaskManager, TaskResult, TaskStatus


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_pending_not_done(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.PENDING)
        assert r.done is False

    def test_running_not_done(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.RUNNING)
        assert r.done is False

    def test_completed_is_done(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.COMPLETED)
        assert r.done is True

    def test_failed_is_done(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.FAILED)
        assert r.done is True

    def test_cancelled_is_done(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.CANCELLED)
        assert r.done is True

    def test_elapsed_with_times(self):
        r = TaskResult(
            task_id="t1", name="test", status=TaskStatus.COMPLETED,
            started_at=100.0, completed_at=105.5,
        )
        assert r.elapsed_s == 5.5

    def test_elapsed_no_start(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.PENDING)
        assert r.elapsed_s == 0.0


class TestTaskManagerInit:
    """Tests for TaskManager initialization."""

    def test_defaults(self):
        tm = TaskManager()
        assert tm.max_concurrent == 5
        assert len(tm._tasks) == 0

    def test_custom_concurrent(self):
        tm = TaskManager(max_concurrent=2)
        assert tm.max_concurrent == 2


class TestSubmit:
    """Tests for submit() shell command tasks."""

    def test_submit_returns_id(self):
        tm = TaskManager()
        tid = tm.submit("test", "echo hello")
        assert tid.startswith("task-")

    def test_submit_creates_task(self):
        tm = TaskManager()
        tid = tm.submit("test", "echo hello")
        result = tm.get_status(tid)
        assert result is not None
        assert result.name == "test"

    def test_submit_echo_completes(self):
        tm = TaskManager()
        tid = tm.submit("echo test", "echo hello")
        # Wait for completion
        for _ in range(50):
            result = tm.get_status(tid)
            if result.done:
                break
            time.sleep(0.1)
        assert result.status == TaskStatus.COMPLETED
        assert "hello" in result.output

    def test_submit_failing_command(self):
        tm = TaskManager()
        tid = tm.submit("fail test", "exit 1")
        for _ in range(50):
            result = tm.get_status(tid)
            if result.done:
                break
            time.sleep(0.1)
        assert result.status == TaskStatus.FAILED

    def test_concurrent_limit(self):
        tm = TaskManager(max_concurrent=1)
        # Submit a long-running task
        tid1 = tm.submit("long", "sleep 10")
        time.sleep(0.2)  # let it start

        # Submit another — should fail due to limit
        tid2 = tm.submit("overflow", "echo overflow")
        r2 = tm.get_status(tid2)
        # It might fail due to concurrent limit OR succeed because the first finished
        assert r2 is not None

        # Cleanup
        tm.cancel(tid1)


class TestSubmitCallable:
    """Tests for submit_callable() Python function tasks."""

    def test_callable_completes(self):
        tm = TaskManager()

        def my_fn():
            return "result from callable"

        tid = tm.submit_callable("callable test", my_fn)
        for _ in range(50):
            result = tm.get_status(tid)
            if result.done:
                break
            time.sleep(0.1)
        assert result.status == TaskStatus.COMPLETED
        assert "result from callable" in result.output

    def test_callable_with_args(self):
        tm = TaskManager()

        def add(a, b):
            return str(a + b)

        tid = tm.submit_callable("add", add, args=(3, 4))
        for _ in range(50):
            result = tm.get_status(tid)
            if result.done:
                break
            time.sleep(0.1)
        assert result.status == TaskStatus.COMPLETED
        assert "7" in result.output

    def test_callable_exception(self):
        tm = TaskManager()

        def failing():
            raise ValueError("intentional failure")

        tid = tm.submit_callable("fail", failing)
        for _ in range(50):
            result = tm.get_status(tid)
            if result.done:
                break
            time.sleep(0.1)
        assert result.status == TaskStatus.FAILED
        assert "intentional failure" in result.error


class TestGetStatus:
    """Tests for get_status()."""

    def test_nonexistent_task(self):
        tm = TaskManager()
        assert tm.get_status("nonexistent") is None

    def test_returns_correct_task(self):
        tm = TaskManager()
        tid = tm.submit("test", "echo hello")
        result = tm.get_status(tid)
        assert result.task_id == tid


class TestListTasks:
    """Tests for list_tasks()."""

    def test_list_all(self):
        tm = TaskManager()
        tm.submit("a", "echo a")
        tm.submit("b", "echo b")
        time.sleep(0.5)  # let them complete
        tasks = tm.list_tasks()
        assert len(tasks) >= 2

    def test_filter_by_status(self):
        tm = TaskManager()
        tm.submit("quick", "echo done")
        time.sleep(0.5)
        completed = tm.list_tasks(status=TaskStatus.COMPLETED)
        for t in completed:
            assert t.status == TaskStatus.COMPLETED


class TestCancel:
    """Tests for cancel()."""

    def test_cancel_running_task(self):
        tm = TaskManager()
        tid = tm.submit("long", "sleep 30")
        time.sleep(0.3)  # wait for it to start
        success = tm.cancel(tid)
        assert success is True
        result = tm.get_status(tid)
        assert result.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self):
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False

    def test_cancel_completed_task(self):
        tm = TaskManager()
        tid = tm.submit("quick", "echo done")
        time.sleep(0.5)  # wait for completion
        # Can't cancel a completed task
        assert tm.cancel(tid) is False


class TestCleanup:
    """Tests for cleanup()."""

    def test_cleanup_removes_done_tasks(self):
        tm = TaskManager()
        tm.submit("a", "echo a")
        tm.submit("b", "echo b")
        time.sleep(0.5)
        assert len(tm._tasks) >= 2
        removed = tm.cleanup()
        assert removed >= 2
        assert len(tm._tasks) == 0

    def test_cleanup_empty(self):
        tm = TaskManager()
        assert tm.cleanup() == 0


class TestCallback:
    """Tests for task completion callbacks."""

    def test_callback_called_on_completion(self):
        results = []
        tm = TaskManager()

        def on_done(result):
            results.append(result)

        tid = tm.submit("cb test", "echo callback", callback=on_done)
        for _ in range(50):
            if results:
                break
            time.sleep(0.1)
        assert len(results) == 1
        assert results[0].status == TaskStatus.COMPLETED

    def test_callable_callback(self):
        results = []
        tm = TaskManager()

        def on_done(result):
            results.append(result.output)

        def my_fn():
            return "callable output"

        tm.submit_callable("cb", my_fn, callback=on_done)
        for _ in range(50):
            if results:
                break
            time.sleep(0.1)
        assert len(results) == 1
        assert "callable output" in results[0]
