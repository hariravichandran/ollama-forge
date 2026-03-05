"""Tests for batch 13 improvements: autofix limits, web search validation, cascade bounds, ideas validation, API constants."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# === AutoFix: Constants and Validation ===

class TestAutoFixConstants:
    """Tests for autofix constants."""

    def test_max_fix_attempts(self):
        from forge.agents.autofix import MAX_FIX_ATTEMPTS
        assert MAX_FIX_ATTEMPTS > 0

    def test_min_fix_attempts(self):
        from forge.agents.autofix import MIN_FIX_ATTEMPTS
        assert MIN_FIX_ATTEMPTS >= 1

    def test_max_fix_attempts_limit(self):
        from forge.agents.autofix import MAX_FIX_ATTEMPTS_LIMIT
        assert MAX_FIX_ATTEMPTS_LIMIT > MAX_FIX_ATTEMPTS_LIMIT // 2  # reasonable

    def test_check_timeout(self):
        from forge.agents.autofix import CHECK_TIMEOUT
        assert CHECK_TIMEOUT > 0

    def test_max_output_length(self):
        from forge.agents.autofix import MAX_OUTPUT_LENGTH
        assert MAX_OUTPUT_LENGTH > 0

    def test_error_preview_length(self):
        from forge.agents.autofix import ERROR_PREVIEW_LENGTH
        assert ERROR_PREVIEW_LENGTH > 0

    def test_error_context_length(self):
        from forge.agents.autofix import ERROR_CONTEXT_LENGTH, ERROR_PREVIEW_LENGTH
        assert ERROR_CONTEXT_LENGTH > ERROR_PREVIEW_LENGTH


class TestAutoFixValidation:
    """Tests for autofix input validation."""

    def test_max_attempts_clamped_high(self):
        from forge.agents.autofix import AutoFixer, MAX_FIX_ATTEMPTS_LIMIT
        fixer = AutoFixer(max_attempts=999, auto_detect=False)
        assert fixer.max_attempts == MAX_FIX_ATTEMPTS_LIMIT

    def test_max_attempts_clamped_low(self):
        from forge.agents.autofix import AutoFixer, MIN_FIX_ATTEMPTS
        fixer = AutoFixer(max_attempts=0, auto_detect=False)
        assert fixer.max_attempts == MIN_FIX_ATTEMPTS

    def test_normal_max_attempts(self):
        from forge.agents.autofix import AutoFixer
        fixer = AutoFixer(max_attempts=3, auto_detect=False)
        assert fixer.max_attempts == 3


class TestAutoFixCheck:
    """Tests for Check dataclass."""

    def test_check_creation(self):
        from forge.agents.autofix import Check
        c = Check(name="test", command="echo hello")
        assert c.name == "test"
        assert c.run_per_file is True

    def test_check_result(self):
        from forge.agents.autofix import CheckResult
        r = CheckResult(check_name="test", passed=True, output="ok")
        assert r.passed

    def test_auto_fix_result(self):
        from forge.agents.autofix import AutoFixResult
        r = AutoFixResult(all_passed=True, checks_run=3, fixes_attempted=0)
        assert r.all_passed
        assert r.final_errors == []


class TestAutoFixRunCheck:
    """Tests for running checks."""

    def test_run_check_success(self):
        from forge.agents.autofix import AutoFixer
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("echo-test", "echo hello", run_per_file=False)
            results = fixer.run_checks([])
            assert len(results) == 1
            assert results[0].passed

    def test_run_check_failure(self):
        from forge.agents.autofix import AutoFixer
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("false-test", "false", run_per_file=False)
            results = fixer.run_checks([])
            assert len(results) == 1
            assert not results[0].passed


# === Web Search: Query Validation, Cache Limits ===

class TestWebSearchConstants:
    """Tests for web search constants."""

    def test_max_query_length(self):
        from forge.mcp.web_search import MAX_QUERY_LENGTH
        assert MAX_QUERY_LENGTH > 0

    def test_max_search_results(self):
        from forge.mcp.web_search import MAX_SEARCH_RESULTS
        assert MAX_SEARCH_RESULTS > 0
        assert MAX_SEARCH_RESULTS <= 100

    def test_cache_ttl_bounds(self):
        from forge.mcp.web_search import MIN_CACHE_TTL, MAX_CACHE_TTL
        assert MIN_CACHE_TTL > 0
        assert MAX_CACHE_TTL > MIN_CACHE_TTL

    def test_max_cache_entries(self):
        from forge.mcp.web_search import MAX_CACHE_ENTRIES
        assert MAX_CACHE_ENTRIES > 0

    def test_build_context_limits(self):
        from forge.mcp.web_search import MAX_BUILD_CONTEXT_QUERIES, MAX_BUILD_CONTEXT_RESULTS
        assert MAX_BUILD_CONTEXT_QUERIES > 0
        assert MAX_BUILD_CONTEXT_RESULTS > 0


class TestWebSearchValidation:
    """Tests for web search input validation."""

    def test_empty_query_returns_empty(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            assert mcp.search("") == []

    def test_whitespace_query_returns_empty(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            assert mcp.search("   ") == []

    def test_disabled_returns_empty(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            mcp.enabled = False
            assert mcp.search("test query") == []


class TestWebSearchCacheBounds:
    """Tests for cache TTL clamping."""

    def test_cache_ttl_clamped_low(self):
        from forge.mcp.web_search import WebSearchMCP, MIN_CACHE_TTL
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_ttl=1, cache_dir=tmpdir)
            assert mcp.cache_ttl == MIN_CACHE_TTL

    def test_cache_ttl_clamped_high(self):
        from forge.mcp.web_search import WebSearchMCP, MAX_CACHE_TTL
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_ttl=999999, cache_dir=tmpdir)
            assert mcp.cache_ttl == MAX_CACHE_TTL

    def test_max_results_clamped(self):
        from forge.mcp.web_search import WebSearchMCP, MAX_SEARCH_RESULTS
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(max_results=999, cache_dir=tmpdir)
            assert mcp.max_results == MAX_SEARCH_RESULTS


class TestWebSearchCache:
    """Tests for cache operations."""

    def test_cache_set_and_get(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            mcp._set_cached("test:5", [{"title": "Test"}])
            result = mcp._get_cached("test:5")
            assert result is not None
            assert result[0]["title"] == "Test"

    def test_cache_expired_returns_none(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_ttl=60, cache_dir=tmpdir)
            mcp._cache["old:5"] = {"data": [{"title": "Old"}], "ts": time.time() - 120}
            assert mcp._get_cached("old:5") is None

    def test_build_context_empty_queries(self):
        from forge.mcp.web_search import WebSearchMCP
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp = WebSearchMCP(cache_dir=tmpdir)
            assert mcp.build_context([]) == ""


# === Cascade: Threshold Validation, Counter Bounds ===

class TestCascadeConstants:
    """Tests for cascade constants."""

    def test_escalation_threshold(self):
        from forge.agents.cascade import ESCALATION_THRESHOLD
        assert ESCALATION_THRESHOLD > 0

    def test_min_escalation_threshold(self):
        from forge.agents.cascade import MIN_ESCALATION_THRESHOLD
        assert MIN_ESCALATION_THRESHOLD >= 1

    def test_max_escalation_threshold(self):
        from forge.agents.cascade import MAX_ESCALATION_THRESHOLD, ESCALATION_THRESHOLD
        assert MAX_ESCALATION_THRESHOLD > ESCALATION_THRESHOLD

    def test_min_useful_length(self):
        from forge.agents.cascade import MIN_USEFUL_LENGTH
        assert MIN_USEFUL_LENGTH > 0

    def test_max_consecutive_poor(self):
        from forge.agents.cascade import MAX_CONSECUTIVE_POOR
        assert MAX_CONSECUTIVE_POOR > 0


class TestCascadeValidation:
    """Tests for cascade threshold validation."""

    def test_threshold_clamped_high(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig, MAX_ESCALATION_THRESHOLD
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        config = CascadeConfig(escalation_threshold=999)
        agent = CascadeAgent(client=client, cascade_config=config)
        assert agent.cascade.escalation_threshold == MAX_ESCALATION_THRESHOLD

    def test_threshold_clamped_low(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig, MIN_ESCALATION_THRESHOLD
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        config = CascadeConfig(escalation_threshold=0)
        agent = CascadeAgent(client=client, cascade_config=config)
        assert agent.cascade.escalation_threshold == MIN_ESCALATION_THRESHOLD

    def test_normal_threshold(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        config = CascadeConfig(escalation_threshold=5)
        agent = CascadeAgent(client=client, cascade_config=config)
        assert agent.cascade.escalation_threshold == 5


class TestCascadePoorResponse:
    """Tests for poor response detection."""

    def test_empty_response_is_poor(self):
        from forge.agents.cascade import CascadeAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("") is True

    def test_short_response_is_poor(self):
        from forge.agents.cascade import CascadeAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("short") is True

    def test_stuck_pattern_is_poor(self):
        from forge.agents.cascade import CascadeAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("I'm not sure about this topic but I'll try to help") is True

    def test_good_response_is_not_poor(self):
        from forge.agents.cascade import CascadeAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = CascadeAgent(client=client)
        good = "Here is a detailed explanation of the topic that covers all the key points you asked about."
        assert agent._is_poor_response(good) is False


class TestCascadeStats:
    """Tests for cascade stats."""

    def test_stats_include_cascade(self):
        from forge.agents.cascade import CascadeAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = CascadeAgent(client=client)
        stats = agent.get_stats()
        assert "cascade" in stats
        assert "escalation_count" in stats["cascade"]
        assert "escalation_success_rate" in stats["cascade"]


# === Ideas: Status Validation, Display Constants ===

class TestIdeasConstants:
    """Tests for ideas constants."""

    def test_fuzzy_length_ratio_min(self):
        from forge.community.ideas import FUZZY_LENGTH_RATIO_MIN
        assert FUZZY_LENGTH_RATIO_MIN > 0

    def test_fuzzy_length_ratio_max(self):
        from forge.community.ideas import FUZZY_LENGTH_RATIO_MAX
        assert FUZZY_LENGTH_RATIO_MAX > 1.0

    def test_max_description_display(self):
        from forge.community.ideas import MAX_DESCRIPTION_DISPLAY
        assert MAX_DESCRIPTION_DISPLAY > 0

    def test_valid_statuses(self):
        from forge.community.ideas import VALID_STATUSES
        assert "new" in VALID_STATUSES
        assert "implemented" in VALID_STATUSES


class TestIdeasStatusValidation:
    """Tests for status validation in update_status."""

    def test_invalid_status_rejected(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.update_status("abc", "invalid_status")
            assert "Invalid status" in result

    def test_valid_status_accepted(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            # Submit an idea first
            collector.submit("Test Idea Title Here", "A test description", "feature")
            ideas = collector.list_ideas()
            if ideas:
                result = collector.update_status(ideas[0].id, "evaluated")
                assert "updated" in result.lower()


class TestIdeasSubmission:
    """Tests for idea submission validation."""

    def test_submit_valid_idea(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.submit("Add dark mode", "Support dark theme", "feature")
            assert "submitted" in result.lower() or "ID" in result

    def test_submit_short_title_rejected(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.submit("Hi", "description", "feature")
            assert "Invalid" in result

    def test_submit_disabled(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir, enabled=False)
            result = collector.submit("Test Idea Title", "Description", "feature")
            assert "disabled" in result.lower()

    def test_submit_invalid_category(self):
        from forge.community.ideas import IdeaCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.submit("Test Idea Title", "Description", "invalid_cat")
            assert "Invalid" in result


# === OpenAI Compat: API Constants ===

class TestOpenAICompatConstants:
    """Tests for OpenAI-compatible API constants."""

    def test_max_messages(self):
        from forge.api.openai_compat import MAX_MESSAGES
        assert MAX_MESSAGES > 0

    def test_max_max_tokens(self):
        from forge.api.openai_compat import MAX_MAX_TOKENS
        assert MAX_MAX_TOKENS > 0

    def test_temperature_bounds(self):
        from forge.api.openai_compat import MIN_TEMPERATURE, MAX_TEMPERATURE
        assert MIN_TEMPERATURE >= 0.0
        assert MAX_TEMPERATURE <= 2.0

    def test_fim_timeout(self):
        from forge.api.openai_compat import FIM_TIMEOUT
        assert FIM_TIMEOUT > 0

    def test_fim_default_max_tokens(self):
        from forge.api.openai_compat import FIM_DEFAULT_MAX_TOKENS
        assert FIM_DEFAULT_MAX_TOKENS > 0

    def test_max_n_completions(self):
        from forge.api.openai_compat import MAX_N_COMPLETIONS
        assert MAX_N_COMPLETIONS >= 1


class TestOpenAICompatRunServer:
    """Tests for run_api_server function."""

    def test_run_api_server_callable(self):
        from forge.api.openai_compat import run_api_server
        assert callable(run_api_server)


# === Integration Tests ===

class TestBatch13Integration:
    """Integration tests across batch 13 improvements."""

    def test_autofix_constants_importable(self):
        from forge.agents.autofix import (
            MAX_FIX_ATTEMPTS, MIN_FIX_ATTEMPTS, MAX_FIX_ATTEMPTS_LIMIT,
            CHECK_TIMEOUT, MAX_OUTPUT_LENGTH,
            ERROR_PREVIEW_LENGTH, ERROR_CONTEXT_LENGTH,
        )
        assert all(v > 0 for v in [
            MAX_FIX_ATTEMPTS, MIN_FIX_ATTEMPTS, MAX_FIX_ATTEMPTS_LIMIT,
            CHECK_TIMEOUT, MAX_OUTPUT_LENGTH,
        ])

    def test_web_search_constants_importable(self):
        from forge.mcp.web_search import (
            MAX_QUERY_LENGTH, MAX_SEARCH_RESULTS,
            MIN_CACHE_TTL, MAX_CACHE_TTL, MAX_CACHE_ENTRIES,
        )
        assert all(v > 0 for v in [
            MAX_QUERY_LENGTH, MAX_SEARCH_RESULTS,
            MIN_CACHE_TTL, MAX_CACHE_TTL, MAX_CACHE_ENTRIES,
        ])

    def test_cascade_constants_importable(self):
        from forge.agents.cascade import (
            ESCALATION_THRESHOLD, MIN_ESCALATION_THRESHOLD,
            MAX_ESCALATION_THRESHOLD, MIN_USEFUL_LENGTH, MAX_CONSECUTIVE_POOR,
        )
        assert MIN_ESCALATION_THRESHOLD <= ESCALATION_THRESHOLD <= MAX_ESCALATION_THRESHOLD

    def test_ideas_constants_importable(self):
        from forge.community.ideas import (
            FUZZY_DEDUP_THRESHOLD, FUZZY_LENGTH_RATIO_MIN,
            FUZZY_LENGTH_RATIO_MAX, VALID_STATUSES,
            MAX_DESCRIPTION_DISPLAY,
        )
        assert FUZZY_DEDUP_THRESHOLD > 0
        assert len(VALID_STATUSES) >= 3

    def test_api_constants_importable(self):
        from forge.api.openai_compat import (
            MAX_MESSAGES, MAX_MAX_TOKENS,
            MIN_TEMPERATURE, MAX_TEMPERATURE,
            FIM_TIMEOUT,
        )
        assert MAX_MESSAGES > 0
        assert FIM_TIMEOUT > 0
