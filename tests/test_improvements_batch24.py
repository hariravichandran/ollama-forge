"""Tests for batch 24 improvements: context manager support (__enter__/__exit__).

Verifies that OllamaClient, WebTool, TaskManager, and BaseAgent
support the `with` statement for automatic resource cleanup.
"""

import inspect

import pytest


class TestOllamaClientContextManager:
    """Tests for OllamaClient context manager support."""

    def test_enter_returns_self(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert client.__enter__() is client

    def test_exit_returns_false(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert client.__exit__(None, None, None) is False

    def test_with_statement(self):
        from forge.llm.client import OllamaClient
        with OllamaClient() as client:
            assert isinstance(client, OllamaClient)

    def test_exit_calls_close(self):
        from forge.llm.client import OllamaClient
        source = inspect.getsource(OllamaClient.__exit__)
        assert "self.close()" in source

    def test_does_not_suppress_exceptions(self):
        from forge.llm.client import OllamaClient
        with pytest.raises(ValueError):
            with OllamaClient() as client:
                raise ValueError("test error")


class TestWebToolContextManager:
    """Tests for WebTool context manager support."""

    def test_enter_returns_self(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert tool.__enter__() is tool

    def test_exit_returns_false(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert tool.__exit__(None, None, None) is False

    def test_with_statement(self):
        import tempfile
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with WebTool(working_dir=tmpdir) as tool:
                assert isinstance(tool, WebTool)

    def test_exit_calls_close(self):
        from forge.tools.web import WebTool
        source = inspect.getsource(WebTool.__exit__)
        assert "self.close()" in source

    def test_session_cleaned_after_exit(self):
        import tempfile
        import requests
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            tool._http_session = requests.Session()
            tool.__exit__(None, None, None)
            assert tool._http_session is None


class TestTaskManagerContextManager:
    """Tests for TaskManager context manager support."""

    def test_enter_returns_self(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        assert tm.__enter__() is tm

    def test_exit_returns_false(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        assert tm.__exit__(None, None, None) is False

    def test_with_statement(self):
        from forge.agents.tasks import TaskManager
        with TaskManager() as tm:
            assert isinstance(tm, TaskManager)

    def test_exit_calls_shutdown(self):
        from forge.agents.tasks import TaskManager
        source = inspect.getsource(TaskManager.__exit__)
        assert "self.shutdown" in source

    def test_does_not_suppress_exceptions(self):
        from forge.agents.tasks import TaskManager
        with pytest.raises(RuntimeError):
            with TaskManager() as tm:
                raise RuntimeError("test error")


class TestBaseAgentContextManager:
    """Tests for BaseAgent context manager support."""

    def test_has_enter(self):
        from forge.agents.base import BaseAgent
        assert hasattr(BaseAgent, "__enter__")

    def test_has_exit(self):
        from forge.agents.base import BaseAgent
        assert hasattr(BaseAgent, "__exit__")

    def test_has_close(self):
        from forge.agents.base import BaseAgent
        assert hasattr(BaseAgent, "close")

    def test_exit_calls_close(self):
        from forge.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent.__exit__)
        assert "self.close()" in source

    def test_close_iterates_tools(self):
        """close() should iterate over tools and call close() on each."""
        from forge.agents.base import BaseAgent
        source = inspect.getsource(BaseAgent.close)
        assert "self._tools" in source
        assert "close" in source


class TestBatch24Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_client_imports(self):
        from forge.llm.client import OllamaClient

    def test_web_imports(self):
        from forge.tools.web import WebTool

    def test_tasks_imports(self):
        from forge.agents.tasks import TaskManager, TaskResult, TaskStatus

    def test_base_imports(self):
        from forge.agents.base import BaseAgent
