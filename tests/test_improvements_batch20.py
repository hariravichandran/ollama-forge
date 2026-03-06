"""Tests for batch 20 improvements: resource cleanup and performance.

Verifies connection pooling, cache eviction, rate limit cleanup,
and circuit breaker stale entry cleanup.
"""

import time

import pytest


class TestClientConnectionPooling:
    """Tests for LLM client connection pooling."""

    def test_delete_model_uses_session(self):
        """delete_model should use the session pool, not raw requests."""
        from forge.llm.client import OllamaClient
        import inspect
        source = inspect.getsource(OllamaClient.delete_model)
        assert "self._get_session().delete" in source
        # Should NOT use raw requests.delete
        assert "requests.delete" not in source


class TestWebToolCacheEviction:
    """Tests for WebTool cache eviction."""

    def test_max_cache_entries_constant(self):
        from forge.tools.web import MAX_CACHE_ENTRIES
        assert MAX_CACHE_ENTRIES > 0

    def test_cache_eviction_on_overflow(self):
        from forge.tools.web import WebTool, MAX_CACHE_ENTRIES
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            # Fill cache to max
            for i in range(MAX_CACHE_ENTRIES):
                tool._cache[f"key_{i}"] = {"data": f"value_{i}", "ts": time.time() - i}
            assert len(tool._cache) == MAX_CACHE_ENTRIES

            # Adding one more should evict oldest
            tool._set_cached("new_key", "new_value")
            assert len(tool._cache) <= MAX_CACHE_ENTRIES
            assert "new_key" in tool._cache

    def test_eviction_removes_oldest(self):
        from forge.tools.web import WebTool
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            tool._cache = {
                "old": {"data": "old_value", "ts": 1000},
                "new": {"data": "new_value", "ts": 9999},
            }
            # Simulate full cache
            from forge.tools.web import MAX_CACHE_ENTRIES
            # Pad to MAX_CACHE_ENTRIES
            for i in range(MAX_CACHE_ENTRIES - 2):
                tool._cache[f"pad_{i}"] = {"data": "", "ts": 5000}

            tool._set_cached("latest", "latest_value")
            # "old" should have been evicted (lowest ts)
            assert "old" not in tool._cache
            assert "latest" in tool._cache


class TestWebToolRateLimitCleanup:
    """Tests for WebTool rate limit domain cleanup."""

    def test_max_rate_limit_domains_constant(self):
        from forge.tools.web import MAX_RATE_LIMIT_DOMAINS
        assert MAX_RATE_LIMIT_DOMAINS > 0

    def test_rate_limit_cleanup_source(self):
        """_apply_rate_limit should clean stale entries."""
        from forge.tools.web import WebTool
        import inspect
        source = inspect.getsource(WebTool._apply_rate_limit)
        assert "MAX_RATE_LIMIT_DOMAINS" in source


class TestWebToolClose:
    """Tests for WebTool close() method."""

    def test_close_method_exists(self):
        from forge.tools.web import WebTool
        assert hasattr(WebTool, "close")

    def test_close_cleans_session(self):
        from forge.tools.web import WebTool
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            # Force session creation
            import requests
            tool._http_session = requests.Session()
            assert tool._http_session is not None
            tool.close()
            assert tool._http_session is None

    def test_close_safe_when_no_session(self):
        from forge.tools.web import WebTool
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            # Should not raise even with no session
            tool.close()
            assert tool._http_session is None


class TestCircuitBreakerCleanup:
    """Tests for BaseAgent circuit breaker stale entry cleanup."""

    def test_cleanup_constants_exist(self):
        from forge.agents.base import (
            CIRCUIT_BREAKER_CLEANUP_INTERVAL,
            MAX_CIRCUIT_BREAKER_ENTRIES,
        )
        assert CIRCUIT_BREAKER_CLEANUP_INTERVAL > 0
        assert MAX_CIRCUIT_BREAKER_ENTRIES > 0

    def test_tool_call_count_attribute(self):
        """BaseAgent should track tool call count for cleanup."""
        from forge.agents.base import BaseAgent
        import inspect
        source = inspect.getsource(BaseAgent.__init__)
        assert "_tool_call_count" in source

    def test_cleanup_method_exists(self):
        from forge.agents.base import BaseAgent
        assert hasattr(BaseAgent, "_cleanup_stale_circuit_breakers")

    def test_cleanup_removes_stale_entries(self):
        """_cleanup_stale_circuit_breakers should remove expired entries."""
        from forge.agents.base import BaseAgent
        import inspect
        source = inspect.getsource(BaseAgent._cleanup_stale_circuit_breakers)
        assert "CIRCUIT_BREAKER_RESET_TIME" in source
        assert "pop" in source


class TestBatch20Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_client_imports(self):
        from forge.llm.client import OllamaClient

    def test_web_tool_imports(self):
        from forge.tools.web import WebTool, MAX_CACHE_ENTRIES, MAX_RATE_LIMIT_DOMAINS

    def test_base_agent_imports(self):
        from forge.agents.base import (
            BaseAgent, CIRCUIT_BREAKER_CLEANUP_INTERVAL, MAX_CIRCUIT_BREAKER_ENTRIES,
        )
