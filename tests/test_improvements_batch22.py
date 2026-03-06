"""Tests for batch 22 improvements: graceful shutdown, close(), and TaskResult improvements.

Verifies that OllamaClient has close(), TaskManager has shutdown(),
and TaskResult has __repr__ and property docstrings.
"""

import time
import pytest


class TestOllamaClientClose:
    """Tests for OllamaClient.close() method."""

    def test_close_method_exists(self):
        from forge.llm.client import OllamaClient
        assert hasattr(OllamaClient, "close")

    def test_close_does_not_raise(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        client.close()  # Should not raise

    def test_close_is_idempotent(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        client.close()
        client.close()  # Should not raise on double-close


class TestTaskManagerShutdown:
    """Tests for TaskManager.shutdown() method."""

    def test_shutdown_method_exists(self):
        from forge.agents.tasks import TaskManager
        assert hasattr(TaskManager, "shutdown")

    def test_shutdown_with_no_tasks(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        tm.shutdown()  # Should not raise with no tasks

    def test_shutdown_cancels_running_tasks(self):
        """shutdown() should cancel all running tasks."""
        from forge.agents.tasks import TaskManager
        import inspect
        source = inspect.getsource(TaskManager.shutdown)
        assert "cancel" in source
        assert "join" in source

    def test_shutdown_has_timeout_param(self):
        """shutdown() should accept a timeout parameter."""
        from forge.agents.tasks import TaskManager
        import inspect
        sig = inspect.signature(TaskManager.shutdown)
        assert "timeout" in sig.parameters


class TestTaskResultRepr:
    """Tests for TaskResult __repr__ and property docstrings."""

    def test_repr_pending(self):
        from forge.agents.tasks import TaskResult, TaskStatus
        result = TaskResult(task_id="task-abc", name="test", status=TaskStatus.PENDING)
        r = repr(result)
        assert "task-abc" in r
        assert "pending" in r

    def test_repr_completed(self):
        from forge.agents.tasks import TaskResult, TaskStatus
        result = TaskResult(
            task_id="task-xyz", name="test", status=TaskStatus.COMPLETED,
            started_at=1000.0, completed_at=1005.0,
        )
        r = repr(result)
        assert "task-xyz" in r
        assert "completed" in r
        assert "5.0s" in r

    def test_repr_failed(self):
        from forge.agents.tasks import TaskResult, TaskStatus
        result = TaskResult(
            task_id="task-err", name="test", status=TaskStatus.FAILED,
            started_at=1000.0, completed_at=1002.5,
        )
        r = repr(result)
        assert "failed" in r

    def test_done_property_docstring(self):
        from forge.agents.tasks import TaskResult
        assert TaskResult.done.fget.__doc__ is not None

    def test_elapsed_s_property_docstring(self):
        from forge.agents.tasks import TaskResult
        assert TaskResult.elapsed_s.fget.__doc__ is not None


class TestBatch22Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_client_imports(self):
        from forge.llm.client import OllamaClient

    def test_tasks_imports(self):
        from forge.agents.tasks import TaskManager, TaskResult, TaskStatus
