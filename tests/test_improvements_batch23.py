"""Tests for batch 23 improvements: thread safety and concurrency.

Verifies that locks are present in OllamaClient, TaskManager, WebTool,
and BaseAgent for protecting shared mutable state.
"""

import inspect
import threading

import pytest


class TestOllamaClientThreadSafety:
    """Tests for OllamaClient thread safety locks."""

    def test_stats_lock_exists(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert hasattr(client, "_stats_lock")
        assert isinstance(client._stats_lock, type(threading.Lock()))

    def test_cache_lock_exists(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert hasattr(client, "_cache_lock")
        assert isinstance(client._cache_lock, type(threading.Lock()))

    def test_session_lock_exists(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert hasattr(client, "_session_lock")
        assert isinstance(client._session_lock, type(threading.Lock()))

    def test_get_session_uses_lock(self):
        """_get_session should use _session_lock."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient._get_session)
        assert "_session_lock" in source

    def test_close_uses_lock(self):
        """close() should use _session_lock."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.close)
        assert "_session_lock" in source

    def test_list_models_uses_lock(self):
        """list_models() should use _cache_lock."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.list_models)
        assert "_cache_lock" in source

    def test_generate_uses_stats_lock(self):
        """generate() should use _stats_lock for stats updates."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.generate)
        assert "_stats_lock" in source

    def test_chat_uses_stats_lock(self):
        """chat() should use _stats_lock for stats updates."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.chat)
        assert "_stats_lock" in source

    def test_stream_chat_uses_stats_lock(self):
        """stream_chat() should use _stats_lock for stats updates."""
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.stream_chat)
        assert "_stats_lock" in source

    def test_threading_imported(self):
        """client.py should import threading."""
        import forge.llm.client as mod
        source = inspect.getsource(mod)
        assert "import threading" in source


class TestTaskManagerThreadSafety:
    """Tests for TaskManager lock usage in all methods."""

    def test_lock_exists(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        assert hasattr(tm, "_lock")
        assert isinstance(tm._lock, type(threading.Lock()))

    def test_submit_callable_uses_lock(self):
        """submit_callable() should use _lock for task registration."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.submit_callable)
        assert "self._lock" in source

    def test_get_status_uses_lock(self):
        """get_status() should use _lock."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.get_status)
        assert "self._lock" in source

    def test_list_tasks_uses_lock(self):
        """list_tasks() should use _lock."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.list_tasks)
        assert "self._lock" in source

    def test_cancel_uses_lock(self):
        """cancel() should use _lock."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.cancel)
        assert "self._lock" in source

    def test_run_command_uses_lock(self):
        """_run_command() should use _lock for task status updates."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager._run_command)
        assert "self._lock" in source

    def test_run_callable_uses_lock(self):
        """_run_callable() should use _lock for task status updates."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager._run_callable)
        assert "self._lock" in source

    def test_submit_empty_command_uses_lock(self):
        """submit() with empty command should use _lock."""
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.submit)
        assert source.count("self._lock") >= 2  # at least 2 lock usages

    def test_concurrent_submit_callable(self):
        """submit_callable() should be safe under concurrent access."""
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        results = []
        errors = []

        def submit_fn():
            try:
                tid = tm.submit_callable("test", lambda: "ok")
                results.append(tid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_fn) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Errors during concurrent submit: {errors}"
        assert len(results) == 10
        assert len(set(results)) == 10  # all unique task IDs


class TestWebToolThreadSafety:
    """Tests for WebTool thread safety locks."""

    def test_cache_lock_exists(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert hasattr(tool, "_cache_lock")
            assert isinstance(tool._cache_lock, type(threading.Lock()))

    def test_session_lock_exists(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert hasattr(tool, "_session_lock")
            assert isinstance(tool._session_lock, type(threading.Lock()))

    def test_rate_limit_lock_exists(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert hasattr(tool, "_rate_limit_lock")
            assert isinstance(tool._rate_limit_lock, type(threading.Lock()))

    def test_get_cached_uses_lock(self):
        """_get_cached() should use _cache_lock."""
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool._get_cached)
        assert "_cache_lock" in source

    def test_set_cached_uses_lock(self):
        """_set_cached() should use _cache_lock."""
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool._set_cached)
        assert "_cache_lock" in source

    def test_apply_rate_limit_uses_lock(self):
        """_apply_rate_limit() should use _rate_limit_lock."""
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool._apply_rate_limit)
        assert "_rate_limit_lock" in source

    def test_close_uses_lock(self):
        """close() should use _session_lock."""
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool.close)
        assert "_session_lock" in source

    def test_fetch_uses_session_lock(self):
        """_fetch() should use _session_lock for lazy session init."""
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool._fetch)
        assert "_session_lock" in source

    def test_threading_imported(self):
        """web.py should import threading."""
        import forge.tools.web as mod
        source = inspect.getsource(mod)
        assert "import threading" in source


class TestBaseAgentThreadSafety:
    """Tests for BaseAgent circuit breaker lock."""

    def test_circuit_breaker_lock_exists(self):
        """BaseAgent should have _circuit_breaker_lock."""
        from forge.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent.__init__)
        assert "_circuit_breaker_lock" in source

    def test_execute_tool_uses_lock(self):
        """_execute_tool() should use _circuit_breaker_lock."""
        from forge.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent._execute_tool)
        assert "_circuit_breaker_lock" in source

    def test_reset_circuit_breaker_uses_lock(self):
        """reset_circuit_breaker() should use _circuit_breaker_lock."""
        from forge.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent.reset_circuit_breaker)
        assert "_circuit_breaker_lock" in source

    def test_threading_imported(self):
        """base.py should import threading."""
        import forge.agents.base as mod
        source = inspect.getsource(mod)
        assert "import threading" in source


class TestBatch23Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_client_imports(self):
        from forge.llm.client import OllamaClient

    def test_tasks_imports(self):
        from forge.agents.tasks import TaskManager, TaskResult, TaskStatus

    def test_web_imports(self):
        from forge.tools.web import WebTool

    def test_base_imports(self):
        from forge.agents.base import BaseAgent
