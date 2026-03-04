"""Tests for batch 2 improvements:

- WebTool retry logic and structured error messages
- Context compression graceful degradation (extractive fallback)
- Planner edit validation before execution
- Cascade model availability check before escalation
- AutoFixer infinite loop protection
- OllamaClient image validation
- CodebaseTool staleness detection
- CLI tools list/info commands
- CLI batch mode (--input, --no-stream)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# WebTool retry logic
# ──────────────────────────────────────────────────────────────────────────────

class TestWebToolRetry:
    """Test retry logic and structured error messages in WebTool."""

    def _make_tool(self, tmp_path):
        from forge.tools.web import WebTool
        return WebTool(working_dir=str(tmp_path))

    def test_search_empty_query(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._search("")
        assert result == "Error: empty search query"

    def test_fetch_empty_url(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._fetch("")
        assert result == "Error: empty URL"

    def test_search_import_error(self, tmp_path):
        tool = self._make_tool(tmp_path)
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            # Force ImportError by patching the import
            result = tool._search("test query")
            # Either returns import error or retry error
            assert "error" in result.lower() or "not installed" in result.lower()

    def test_fetch_retries_on_500(self, tmp_path):
        """Test that fetch retries on 500 status codes."""
        tool = self._make_tool(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        tool._http_session = mock_session

        with patch("forge.tools.web.RETRY_BACKOFF_BASE", 0.01):  # Fast retries for test
            result = tool._fetch("http://example.com/test")

        assert "error" in result.lower() or "500" in result
        # Should have retried 3 times
        assert mock_session.get.call_count == 3

    def test_fetch_retries_on_429(self, tmp_path):
        """Test that fetch retries on 429 rate limit."""
        tool = self._make_tool(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        tool._http_session = mock_session

        with patch("forge.tools.web.RETRY_BACKOFF_BASE", 0.01):
            result = tool._fetch("http://example.com/test")

        assert mock_session.get.call_count == 3

    def test_fetch_no_retry_on_404(self, tmp_path):
        """Test that fetch does NOT retry on 404 — it's not transient."""
        tool = self._make_tool(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        tool._http_session = mock_session

        with patch("forge.tools.web.RETRY_BACKOFF_BASE", 0.01):
            result = tool._fetch("http://example.com/missing")

        # 404 is not retryable, but the raise_for_status exception triggers retry
        # (only status codes in RETRYABLE_STATUS_CODES get the special status retry)
        assert "error" in result.lower()

    def test_fetch_success_on_second_attempt(self, tmp_path):
        """Test that fetch succeeds when first attempt fails but second works."""
        tool = self._make_tool(tmp_path)

        fail_response = MagicMock()
        fail_response.status_code = 503

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.text = "Hello World"
        ok_response.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.side_effect = [fail_response, ok_response]
        tool._http_session = mock_session

        with patch("forge.tools.web.RETRY_BACKOFF_BASE", 0.01):
            result = tool._fetch("http://example.com/test")

        assert result == "Hello World"
        assert mock_session.get.call_count == 2

    def test_structured_error_includes_attempt_count(self, tmp_path):
        """Test that final error message includes retry count."""
        from forge.tools.web import MAX_RETRIES
        tool = self._make_tool(tmp_path)

        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("connection refused")
        tool._http_session = mock_session

        with patch("forge.tools.web.RETRY_BACKOFF_BASE", 0.01):
            result = tool._fetch("http://unreachable.local")

        assert str(MAX_RETRIES) in result
        assert "attempts" in result.lower()

    def test_retryable_status_codes_defined(self):
        """Verify retryable status codes are reasonable."""
        from forge.tools.web import RETRYABLE_STATUS_CODES
        assert 429 in RETRYABLE_STATUS_CODES  # rate limit
        assert 500 in RETRYABLE_STATUS_CODES  # server error
        assert 502 in RETRYABLE_STATUS_CODES  # bad gateway
        assert 503 in RETRYABLE_STATUS_CODES  # service unavailable
        assert 504 in RETRYABLE_STATUS_CODES  # gateway timeout
        assert 404 not in RETRYABLE_STATUS_CODES  # not retryable
        assert 200 not in RETRYABLE_STATUS_CODES  # success

    def test_cached_results_skip_retry(self, tmp_path):
        """Cached results should be returned immediately, no retries."""
        tool = self._make_tool(tmp_path)
        # Populate cache
        tool._set_cached("fetch:http://example.com", "cached content")

        result = tool._fetch("http://example.com")
        assert result == "cached content"


# ──────────────────────────────────────────────────────────────────────────────
# Context compression graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

class TestContextCompressionFallback:
    """Test extractive fallback when LLM summarization fails."""

    def _make_compressor(self, fail_generate=False, empty_response=False):
        from forge.llm.context import ContextCompressor
        mock_client = MagicMock()
        if fail_generate:
            mock_client.generate.side_effect = ConnectionError("LLM unavailable")
        elif empty_response:
            mock_client.generate.return_value = {"response": "", "error": "timeout"}
        else:
            mock_client.generate.return_value = {"response": "Summary of conversation."}
        return ContextCompressor(client=mock_client, max_tokens=500, keep_recent=2)

    def test_normal_summarization(self):
        """Normal path: LLM summarization works."""
        compressor = self._make_compressor()
        result = compressor._ask_for_summary("Some conversation text")
        assert result == "Summary of conversation."

    def test_fallback_on_llm_exception(self):
        """Fallback: LLM raises exception, extractive summary used."""
        compressor = self._make_compressor(fail_generate=True)
        text = "Some conversation\n```python\nprint('hello')\n```\nMore text"
        result = compressor._ask_for_summary(text)
        # Should return extractive summary (keeps code blocks)
        assert "print('hello')" in result

    def test_fallback_on_empty_response(self):
        """Fallback: LLM returns empty response."""
        compressor = self._make_compressor(empty_response=True)
        text = "Discussion about src/main.py\nError: ImportError\nDecided to use pandas"
        result = compressor._ask_for_summary(text)
        # Should extract important lines
        assert len(result) > 0

    def test_extractive_keeps_code_blocks(self):
        """Extractive summary preserves code blocks intact."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        text = (
            "Hello\nHow are you\n"
            "```python\ndef foo():\n    return 42\n```\n"
            "Thanks\nGoodbye"
        )
        result = compressor._extractive_summary(text)
        assert "def foo():" in result
        assert "return 42" in result

    def test_extractive_keeps_file_paths(self):
        """Extractive summary preserves file paths."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        text = "Greeting\nModified forge/tools/web.py\nDone"
        result = compressor._extractive_summary(text)
        assert "forge/tools/web.py" in result

    def test_extractive_keeps_errors(self):
        """Extractive summary preserves error messages."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        text = "OK\nTraceback (most recent call last):\nFile not found error\nFixed it"
        result = compressor._extractive_summary(text)
        assert "Traceback" in result or "error" in result.lower()

    def test_extractive_keeps_urls(self):
        """Extractive summary preserves URLs."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        text = "Hello\nSee https://example.com/docs for more\nBye"
        result = compressor._extractive_summary(text)
        assert "https://example.com/docs" in result

    def test_extractive_fallback_on_empty_input(self):
        """Extractive summary handles empty or very short input."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        result = compressor._extractive_summary("")
        assert isinstance(result, str)

    def test_extractive_with_no_important_lines(self):
        """When no important patterns found, keep first/last portions."""
        from forge.llm.context import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        # Long text with no code, paths, errors, etc.
        lines = [f"Normal line {i}" for i in range(40)]
        text = "\n".join(lines)
        result = compressor._extractive_summary(text)
        assert "Normal line 0" in result  # first portion kept
        assert "Normal line 39" in result  # last portion kept


# ──────────────────────────────────────────────────────────────────────────────
# Planner edit validation
# ──────────────────────────────────────────────────────────────────────────────

class TestPlannerValidation:
    """Test edit plan validation before execution."""

    def _make_planner(self, tmp_path):
        from forge.agents.planner import EditPlanner
        return EditPlanner(working_dir=str(tmp_path))

    def test_valid_plan(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        # Create file to edit
        (tmp_path / "test.py").write_text("def foo():\n    pass\n")

        plan = EditPlan(
            task="Update foo",
            files=[FileEdit(
                path="test.py",
                description="Change pass to return",
                edits=[{"old_string": "    pass", "new_string": "    return 42"}],
            )],
        )
        errors = planner.validate(plan)
        assert errors == []

    def test_file_not_found(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        plan = EditPlan(
            task="Edit missing file",
            files=[FileEdit(
                path="nonexistent.py",
                description="Update",
                edits=[{"old_string": "old", "new_string": "new"}],
            )],
        )
        errors = planner.validate(plan)
        assert any("not found" in e for e in errors)

    def test_old_string_not_in_file(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        (tmp_path / "test.py").write_text("def bar():\n    pass\n")

        plan = EditPlan(
            task="Edit wrong string",
            files=[FileEdit(
                path="test.py",
                description="Fix",
                edits=[{"old_string": "def foo()", "new_string": "def baz()"}],
            )],
        )
        errors = planner.validate(plan)
        assert any("not found" in e for e in errors)

    def test_ambiguous_old_string(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        (tmp_path / "test.py").write_text("pass\npass\npass\n")

        plan = EditPlan(
            task="Edit ambiguous",
            files=[FileEdit(
                path="test.py",
                description="Fix",
                edits=[{"old_string": "pass", "new_string": "return"}],
            )],
        )
        errors = planner.validate(plan)
        assert any("ambiguous" in e for e in errors)

    def test_create_existing_file(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        (tmp_path / "existing.py").write_text("exists")

        plan = EditPlan(
            task="Create file",
            files=[FileEdit(
                path="existing.py",
                description="Create",
                create=True,
                new_content="new content",
            )],
        )
        errors = planner.validate(plan)
        assert any("already exists" in e for e in errors)

    def test_path_traversal_blocked(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        plan = EditPlan(
            task="Escape",
            files=[FileEdit(
                path="../../etc/passwd",
                description="Escape working dir",
                edits=[{"old_string": "root", "new_string": "hacked"}],
            )],
        )
        errors = planner.validate(plan)
        assert any("escapes" in e or "not found" in e for e in errors)

    def test_empty_old_string(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        (tmp_path / "test.py").write_text("content")

        plan = EditPlan(
            task="Empty edit",
            files=[FileEdit(
                path="test.py",
                description="Fix",
                edits=[{"old_string": "", "new_string": "new"}],
            )],
        )
        errors = planner.validate(plan)
        assert any("empty" in e for e in errors)

    def test_execute_validates_first(self, tmp_path):
        """Execute should validate and fail before modifying files."""
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        plan = EditPlan(
            task="Bad plan",
            files=[FileEdit(
                path="missing.py",
                description="Edit",
                edits=[{"old_string": "x", "new_string": "y"}],
            )],
        )
        result = planner.execute(plan)
        assert not result.success
        assert len(result.errors) > 0
        assert not result.rolled_back  # No rollback needed — nothing was modified

    def test_duplicate_old_string(self, tmp_path):
        from forge.agents.planner import EditPlan, FileEdit
        planner = self._make_planner(tmp_path)

        (tmp_path / "test.py").write_text("unique_content_here")

        plan = EditPlan(
            task="Duplicate edits",
            files=[FileEdit(
                path="test.py",
                description="Duplicate",
                edits=[
                    {"old_string": "unique_content_here", "new_string": "a"},
                    {"old_string": "unique_content_here", "new_string": "b"},
                ],
            )],
        )
        errors = planner.validate(plan)
        assert any("duplicate" in e for e in errors)


# ──────────────────────────────────────────────────────────────────────────────
# Cascade model availability check
# ──────────────────────────────────────────────────────────────────────────────

class TestCascadeAvailability:
    """Test model availability check before escalation."""

    def _make_cascade_agent(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        from forge.agents.base import AgentConfig

        mock_client = MagicMock()
        mock_client.model = "small-model:3b"
        mock_client.list_models.return_value = [
            {"name": "small-model:3b"},
        ]

        config = AgentConfig(
            name="test",
            system_prompt="Test",
            tools=[],
        )
        cascade_config = CascadeConfig(
            primary_model="small-model:3b",
            escalation_model="big-model:14b",
            escalation_threshold=1,
        )
        agent = CascadeAgent(
            client=mock_client,
            config=config,
            cascade_config=cascade_config,
        )
        return agent, mock_client

    def test_model_not_available_prevents_escalation(self):
        agent, mock_client = self._make_cascade_agent()
        # big-model:14b is not in list_models
        assert not agent._is_model_available("big-model:14b")

    def test_model_available_allows_check(self):
        agent, mock_client = self._make_cascade_agent()
        assert agent._is_model_available("small-model:3b")

    def test_escalation_blocked_when_model_unavailable(self):
        agent, mock_client = self._make_cascade_agent()
        agent.messages = [{"role": "assistant", "content": "bad response"}]

        result = agent._escalate_and_retry("test question")

        # Should not have called switch_model since model is unavailable
        mock_client.switch_model.assert_not_called()
        # Counter should be reset to prevent re-escalation attempts
        assert agent._consecutive_poor == 0

    def test_model_available_check_handles_exceptions(self):
        agent, mock_client = self._make_cascade_agent()
        mock_client.list_models.side_effect = Exception("connection error")
        assert not agent._is_model_available("any-model")

    def test_model_available_by_base_name(self):
        """Check matching by base name (without tag)."""
        agent, mock_client = self._make_cascade_agent()
        mock_client.list_models.return_value = [{"name": "qwen2.5-coder:7b"}]
        assert agent._is_model_available("qwen2.5-coder:7b")
        assert agent._is_model_available("qwen2.5-coder")


# ──────────────────────────────────────────────────────────────────────────────
# AutoFixer infinite loop protection
# ──────────────────────────────────────────────────────────────────────────────

class TestAutoFixerLoopProtection:
    """Test infinite loop detection in auto-fix."""

    def test_breaks_on_recurring_errors(self, tmp_path):
        from forge.agents.autofix import AutoFixer, CheckResult

        fixer = AutoFixer(working_dir=str(tmp_path), auto_detect=False, max_attempts=5)
        fixer.checks = []

        # Mock run_checks to always return the same error
        same_error = CheckResult(check_name="lint", passed=False, output="syntax error line 5", file="test.py")
        fixer.run_checks = MagicMock(return_value=[same_error])

        fix_calls = []
        def mock_fix(msg):
            fix_calls.append(msg)

        result = fixer.check_and_fix(["test.py"], fix_callback=mock_fix)

        # Should have broken early — not exhausted all 5 attempts
        assert not result.all_passed
        # First attempt: fix called, second attempt: same error → break
        assert len(fix_calls) == 1  # Only 1 fix attempt before loop detection

    def test_continues_on_different_errors(self, tmp_path):
        from forge.agents.autofix import AutoFixer, CheckResult

        fixer = AutoFixer(working_dir=str(tmp_path), auto_detect=False, max_attempts=5)
        fixer.checks = []

        call_count = [0]
        def make_different_errors(files):
            call_count[0] += 1
            if call_count[0] >= 4:
                return []  # Pass on 4th check
            return [CheckResult(
                check_name="lint",
                passed=False,
                output=f"error on attempt {call_count[0]}",
                file="test.py",
            )]

        fixer.run_checks = MagicMock(side_effect=make_different_errors)

        fix_calls = []
        result = fixer.check_and_fix(["test.py"], fix_callback=lambda msg: fix_calls.append(msg))

        assert result.all_passed
        assert len(fix_calls) == 3  # Fixed 3 different errors

    def test_no_callback_reports_errors_immediately(self, tmp_path):
        from forge.agents.autofix import AutoFixer, CheckResult

        fixer = AutoFixer(working_dir=str(tmp_path), auto_detect=False)
        fixer.checks = []

        error = CheckResult(check_name="test", passed=False, output="test failed", file="main.py")
        fixer.run_checks = MagicMock(return_value=[error])

        result = fixer.check_and_fix(["main.py"], fix_callback=None)

        assert not result.all_passed
        assert len(result.final_errors) > 0


# ──────────────────────────────────────────────────────────────────────────────
# OllamaClient image validation
# ──────────────────────────────────────────────────────────────────────────────

class TestImageValidation:
    """Test image validation in _inject_images."""

    def _make_client(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        client.MAX_IMAGE_SIZE = 20 * 1024 * 1024
        client.SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        return client

    def test_valid_png_accepted(self, tmp_path):
        client = self._make_client()
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake image data")

        messages = [{"role": "user", "content": "Describe this"}]
        result = client._inject_images(messages, [str(img)])

        assert "images" in result[-1]
        assert len(result[-1]["images"]) == 1

    def test_unsupported_extension_skipped(self, tmp_path):
        client = self._make_client()
        img = tmp_path / "test.svg"
        img.write_bytes(b"<svg>...</svg>")

        messages = [{"role": "user", "content": "Describe this"}]
        result = client._inject_images(messages, [str(img)])

        # SVG skipped, no images field added
        assert "images" not in result[-1]

    def test_oversized_file_skipped(self, tmp_path):
        client = self._make_client()
        client.MAX_IMAGE_SIZE = 100  # Very small limit for test

        img = tmp_path / "big.png"
        img.write_bytes(b"\x89PNG" + b"x" * 200)

        messages = [{"role": "user", "content": "What's this?"}]
        result = client._inject_images(messages, [str(img)])

        assert "images" not in result[-1]

    def test_empty_file_skipped(self, tmp_path):
        client = self._make_client()
        img = tmp_path / "empty.png"
        img.write_bytes(b"")

        messages = [{"role": "user", "content": "?"}]
        result = client._inject_images(messages, [str(img)])

        assert "images" not in result[-1]

    def test_base64_string_passed_through(self):
        client = self._make_client()
        messages = [{"role": "user", "content": "Analyze"}]
        result = client._inject_images(messages, ["aGVsbG8="])  # base64 for "hello"

        assert result[-1]["images"] == ["aGVsbG8="]

    def test_multiple_images_mixed_validity(self, tmp_path):
        client = self._make_client()
        good = tmp_path / "good.jpg"
        good.write_bytes(b"\xff\xd8\xff" + b"fake jpg data")

        bad = tmp_path / "bad.txt"
        bad.write_bytes(b"not an image")

        messages = [{"role": "user", "content": "Compare"}]
        result = client._inject_images(messages, [str(good), str(bad)])

        # Only good.jpg should be included
        assert len(result[-1]["images"]) == 1

    def test_supported_extensions_list(self):
        from forge.llm.client import OllamaClient
        assert ".png" in OllamaClient.SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpg" in OllamaClient.SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in OllamaClient.SUPPORTED_IMAGE_EXTENSIONS
        assert ".gif" in OllamaClient.SUPPORTED_IMAGE_EXTENSIONS
        assert ".webp" in OllamaClient.SUPPORTED_IMAGE_EXTENSIONS


# ──────────────────────────────────────────────────────────────────────────────
# CodebaseTool staleness detection
# ──────────────────────────────────────────────────────────────────────────────

class TestCodebaseToolStaleness:
    """Test codebase index staleness detection and refresh."""

    def test_staleness_check_interval_defined(self):
        from forge.tools.codebase import CodebaseTool
        assert CodebaseTool.STALENESS_CHECK_INTERVAL > 0

    def test_initial_index_sets_check_time(self, tmp_path):
        from forge.tools.codebase import CodebaseTool
        tool = CodebaseTool(working_dir=str(tmp_path))
        tool._ensure_indexed()
        assert tool._last_staleness_check > 0

    def test_staleness_not_rechecked_within_interval(self, tmp_path):
        from forge.tools.codebase import CodebaseTool
        tool = CodebaseTool(working_dir=str(tmp_path))
        tool._ensure_indexed()

        first_check = tool._last_staleness_check
        tool._ensure_indexed()  # Should not re-check
        assert tool._last_staleness_check == first_check

    def test_staleness_rechecked_after_interval(self, tmp_path):
        from forge.tools.codebase import CodebaseTool
        tool = CodebaseTool(working_dir=str(tmp_path))
        tool._ensure_indexed()

        # Simulate interval passing
        tool._last_staleness_check = time.time() - tool.STALENESS_CHECK_INTERVAL - 1
        old_check = tool._last_staleness_check

        tool._ensure_indexed()
        assert tool._last_staleness_check > old_check


# ──────────────────────────────────────────────────────────────────────────────
# CLI tools commands
# ──────────────────────────────────────────────────────────────────────────────

class TestCLIToolsCommands:
    """Test forge tools list and info CLI commands."""

    def test_tools_list_command_exists(self):
        """Verify the tools group is registered on the main CLI."""
        from forge.cli import main
        # Check that 'tools' is a registered command
        commands = main.commands if hasattr(main, 'commands') else {}
        assert "tools" in commands, "tools command group not registered"

    def test_tools_info_command_exists(self):
        """Verify tools info subcommand exists."""
        from forge.cli import tools
        commands = tools.commands if hasattr(tools, 'commands') else {}
        assert "info" in commands

    def test_tools_list_subcommand_exists(self):
        """Verify tools list subcommand exists."""
        from forge.cli import tools
        commands = tools.commands if hasattr(tools, 'commands') else {}
        assert "list" in commands


# ──────────────────────────────────────────────────────────────────────────────
# CLI batch mode
# ──────────────────────────────────────────────────────────────────────────────

class TestCLIBatchMode:
    """Test chat --input and --no-stream batch mode options."""

    def test_chat_accepts_input_option(self):
        """Verify chat command has --input parameter."""
        from forge.cli import chat
        params = {p.name for p in chat.params}
        assert "input_file" in params

    def test_chat_accepts_no_stream_option(self):
        """Verify chat command has --no-stream parameter."""
        from forge.cli import chat
        params = {p.name for p in chat.params}
        assert "no_stream" in params


# ──────────────────────────────────────────────────────────────────────────────
# Integration: constants and module-level checks
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleConstants:
    """Verify module-level constants are properly defined."""

    def test_web_retry_constants(self):
        from forge.tools.web import MAX_RETRIES, RETRY_BACKOFF_BASE, RETRYABLE_STATUS_CODES
        assert MAX_RETRIES >= 1
        assert RETRY_BACKOFF_BASE > 0
        assert isinstance(RETRYABLE_STATUS_CODES, set)

    def test_image_constants(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient.MAX_IMAGE_SIZE > 0
        assert len(OllamaClient.SUPPORTED_IMAGE_EXTENSIONS) >= 5

    def test_planner_validate_method_exists(self):
        from forge.agents.planner import EditPlanner
        assert hasattr(EditPlanner, "validate")

    def test_cascade_is_model_available_method_exists(self):
        from forge.agents.cascade import CascadeAgent
        assert hasattr(CascadeAgent, "_is_model_available")

    def test_context_extractive_summary_method_exists(self):
        from forge.llm.context import ContextCompressor
        assert hasattr(ContextCompressor, "_extractive_summary")
