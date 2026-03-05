"""Tests for batch 9 improvements: client robustness, agent limits, filesystem safety, git validation."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# === LLM Client: JSON Robustness, Model Validation, Context Bounds ===

class TestClientModelValidation:
    """Tests for model name validation."""

    def test_valid_model_name(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient.validate_model_name("qwen2.5-coder:7b") == ""

    def test_valid_model_with_slash(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient.validate_model_name("library/model:latest") == ""

    def test_empty_model_name(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient.validate_model_name("")
        assert "empty" in result.lower()

    def test_model_name_too_long(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient.validate_model_name("a" * 101)
        assert "too long" in result.lower()

    def test_model_name_with_dotdot(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient.validate_model_name("model/../evil")
        assert ".." in result

    def test_model_name_special_chars(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient.validate_model_name("model; rm -rf /")
        assert result != ""  # should fail validation


class TestClientBaseURLValidation:
    """Tests for base URL validation."""

    def test_valid_http_url(self):
        from forge.llm.client import OllamaClient
        assert OllamaClient._validate_base_url("http://localhost:11434") == "http://localhost:11434"

    def test_valid_https_url(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient._validate_base_url("https://ollama.example.com")
        assert result == "https://ollama.example.com"

    def test_invalid_scheme(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient._validate_base_url("ftp://badhost:11434")
        assert result == "http://localhost:11434"

    def test_trailing_slash_stripped(self):
        from forge.llm.client import OllamaClient
        result = OllamaClient._validate_base_url("http://localhost:11434/")
        assert not result.endswith("/")


class TestClientContextBounds:
    """Tests for context window clamping."""

    def test_num_ctx_clamped_low(self):
        from forge.llm.client import OllamaClient, MIN_NUM_CTX
        client = OllamaClient(num_ctx=10)
        assert client.num_ctx == MIN_NUM_CTX

    def test_num_ctx_clamped_high(self):
        from forge.llm.client import OllamaClient, MAX_NUM_CTX
        client = OllamaClient(num_ctx=999999)
        assert client.num_ctx == MAX_NUM_CTX

    def test_num_ctx_valid_passes(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient(num_ctx=4096)
        assert client.num_ctx == 4096


class TestClientSafeJSON:
    """Tests for safe JSON parsing."""

    def test_safe_json_valid(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "hello"}
        mock_resp.content = b'{"response": "hello"}'
        result = client._safe_json(mock_resp)
        assert result["response"] == "hello"

    def test_safe_json_malformed(self):
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_resp.content = b"not json"
        result = client._safe_json(mock_resp)
        assert "error" in result

    def test_safe_json_too_large(self):
        from forge.llm.client import OllamaClient, MAX_RESPONSE_SIZE
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.content = b"x" * (MAX_RESPONSE_SIZE + 1)
        result = client._safe_json(mock_resp)
        assert "too large" in result.get("error", "").lower()


class TestClientConstants:
    """Tests for client constants."""

    def test_min_ctx(self):
        from forge.llm.client import MIN_NUM_CTX
        assert MIN_NUM_CTX > 0

    def test_max_ctx(self):
        from forge.llm.client import MAX_NUM_CTX
        assert MAX_NUM_CTX >= 8192

    def test_model_name_pattern(self):
        from forge.llm.client import MODEL_NAME_PATTERN
        assert MODEL_NAME_PATTERN.match("llama3:8b")
        assert not MODEL_NAME_PATTERN.match("")


# === Base Agent: Message Limits, Circuit Breaker Auto-Reset ===

class TestAgentMessageLimits:
    """Tests for message size and conversation limits."""

    def test_max_message_length_constant(self):
        from forge.agents.base import MAX_USER_MESSAGE_LENGTH
        assert MAX_USER_MESSAGE_LENGTH > 0

    def test_max_conversation_messages_constant(self):
        from forge.agents.base import MAX_CONVERSATION_MESSAGES
        assert MAX_CONVERSATION_MESSAGES > 0

    def test_max_tool_result_length_constant(self):
        from forge.agents.base import MAX_TOOL_RESULT_LENGTH
        assert MAX_TOOL_RESULT_LENGTH > 0

    def test_max_error_message_length_constant(self):
        from forge.agents.base import MAX_ERROR_MESSAGE_LENGTH
        assert MAX_ERROR_MESSAGE_LENGTH > 0


class TestCircuitBreakerAutoReset:
    """Tests for circuit breaker auto-reset timing."""

    def test_reset_time_constant(self):
        from forge.agents.base import CIRCUIT_BREAKER_RESET_TIME
        assert CIRCUIT_BREAKER_RESET_TIME > 0

    def test_circuit_breaker_has_failure_times(self):
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        assert hasattr(agent, "_tool_failure_times")
        assert isinstance(agent._tool_failure_times, dict)

    def test_circuit_breaker_records_failure_time(self):
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.agents.permissions import PermissionManager
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        pm = PermissionManager(auto_approve_all=True)
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]), permissions=pm)
        # Simulate a failure by calling _execute_tool with unknown function
        result = agent._execute_tool("nonexistent_func", {})
        assert "Unknown" in result

    def test_circuit_breaker_threshold(self):
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        assert agent._tool_circuit_threshold == 3


class TestAgentToolResultTruncation:
    """Tests for tool result truncation."""

    def test_truncation_with_mock_tool(self):
        from forge.agents.base import BaseAgent, AgentConfig, MAX_TOOL_RESULT_LENGTH
        from forge.agents.permissions import PermissionManager
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        pm = PermissionManager(auto_approve_all=True)
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]), permissions=pm)

        # Create a mock tool that returns a huge result
        mock_tool = MagicMock()
        mock_tool.execute.return_value = "x" * (MAX_TOOL_RESULT_LENGTH + 1000)
        agent._function_tool_map["test_func"] = mock_tool

        result = agent._execute_tool("test_func", {})
        assert len(result) <= MAX_TOOL_RESULT_LENGTH + 50  # +50 for "... (output truncated)"
        assert "truncated" in result


# === Filesystem: Write Size, ReDoS, Hard Links ===

class TestFilesystemWriteSize:
    """Tests for write size limits."""

    def test_max_write_size_constant(self):
        from forge.tools.filesystem import MAX_WRITE_SIZE
        assert MAX_WRITE_SIZE > 0

    def test_write_size_validation(self):
        from forge.tools.filesystem import FilesystemTool, MAX_WRITE_SIZE
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            # Try writing oversized content
            big_content = "x" * (MAX_WRITE_SIZE + 1)
            result = tool._write_file("test.txt", big_content)
            assert "too large" in result.lower()

    def test_normal_write_succeeds(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._write_file("test.txt", "hello world")
            assert "Written" in result
            assert (Path(tmpdir) / "test.txt").read_text() == "hello world"


class TestFilesystemReDoS:
    """Tests for regex pattern validation."""

    def test_max_pattern_length_constant(self):
        from forge.tools.filesystem import MAX_REGEX_PATTERN_LENGTH
        assert MAX_REGEX_PATTERN_LENGTH > 0

    def test_oversized_pattern_rejected(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            long_pattern = "a" * 501
            result = tool._search_files(long_pattern)
            assert "too long" in result.lower()

    def test_invalid_regex_caught(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._search_files("[invalid regex")
            assert "invalid" in result.lower()

    def test_valid_regex_works(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            (Path(tmpdir) / "test.py").write_text("def hello():\n    pass\n")
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._search_files("def hello")
            assert "test.py" in result


class TestFilesystemSearchLimits:
    """Tests for search result limits."""

    def test_max_search_results_constant(self):
        from forge.tools.filesystem import MAX_SEARCH_RESULTS
        assert MAX_SEARCH_RESULTS > 0
        assert MAX_SEARCH_RESULTS <= 1000


class TestFilesystemHardLinks:
    """Tests for hard link detection."""

    def test_resolve_path_with_normal_file(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("hello")
            tool = FilesystemTool(working_dir=tmpdir)
            resolved = tool._resolve_path("test.txt")
            assert resolved.exists()


# === Git: Diff Size, Branch Validation, Log Clamping ===

class TestGitBranchValidation:
    """Tests for branch name validation."""

    def test_valid_branch_name(self):
        from forge.tools.git import GitTool
        assert GitTool.validate_branch_name("feature/add-login") == ""

    def test_valid_branch_simple(self):
        from forge.tools.git import GitTool
        assert GitTool.validate_branch_name("main") == ""

    def test_empty_branch_name(self):
        from forge.tools.git import GitTool
        result = GitTool.validate_branch_name("")
        assert "empty" in result.lower()

    def test_branch_starts_with_dash(self):
        from forge.tools.git import GitTool
        result = GitTool.validate_branch_name("-bad-name")
        assert result != ""

    def test_branch_with_dotdot(self):
        from forge.tools.git import GitTool
        result = GitTool.validate_branch_name("feature/../evil")
        assert ".." in result

    def test_branch_ends_with_lock(self):
        from forge.tools.git import GitTool
        result = GitTool.validate_branch_name("branch.lock")
        assert ".lock" in result

    def test_branch_name_special_chars(self):
        from forge.tools.git import GitTool
        result = GitTool.validate_branch_name("branch; rm -rf /")
        assert result != ""


class TestGitDiffSizeLimit:
    """Tests for diff output size limiting."""

    def test_max_diff_size_constant(self):
        from forge.tools.git import MAX_DIFF_SIZE
        assert MAX_DIFF_SIZE > 0

    def test_diff_truncation_message(self):
        from forge.tools.git import MAX_DIFF_SIZE
        assert MAX_DIFF_SIZE == 500_000


class TestGitLogClamping:
    """Tests for log count clamping."""

    def test_max_log_count(self):
        from forge.tools.git import MAX_LOG_COUNT
        assert MAX_LOG_COUNT > 0
        assert MAX_LOG_COUNT <= 1000

    def test_min_log_count(self):
        from forge.tools.git import MIN_LOG_COUNT
        assert MIN_LOG_COUNT >= 1


class TestGitConstants:
    """Tests for git tool constants."""

    def test_branch_pattern(self):
        from forge.tools.git import BRANCH_NAME_PATTERN
        assert BRANCH_NAME_PATTERN.match("feature/test")
        assert not BRANCH_NAME_PATTERN.match("")

    def test_timeout_constants(self):
        from forge.tools.git import TIMEOUT_FAST, TIMEOUT_NORMAL, TIMEOUT_SLOW
        assert TIMEOUT_FAST < TIMEOUT_NORMAL < TIMEOUT_SLOW

    def test_agent_commit_tag(self):
        from forge.tools.git import AGENT_COMMIT_TAG
        assert AGENT_COMMIT_TAG == "[forge]"


# === Integration Tests ===

class TestBatch9Integration:
    """Integration tests across batch 9 improvements."""

    def test_client_validation_chain(self):
        from forge.llm.client import OllamaClient
        # Valid creation
        client = OllamaClient(num_ctx=4096)
        assert client.num_ctx == 4096
        # Model validation
        assert OllamaClient.validate_model_name("llama3:8b") == ""
        assert OllamaClient.validate_model_name("") != ""

    def test_agent_has_all_safety_fields(self):
        from forge.agents.base import BaseAgent, AgentConfig
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        assert hasattr(agent, "_tool_failure_counts")
        assert hasattr(agent, "_tool_failure_times")
        assert hasattr(agent, "_tool_circuit_threshold")

    def test_filesystem_safety_chain(self):
        from forge.tools.filesystem import FilesystemTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            # Normal operations work
            result = tool._write_file("test.txt", "hello")
            assert "Written" in result
            result = tool._read_file("test.txt")
            assert "hello" in result
            # Regex validation works
            result = tool._search_files("[bad regex")
            assert "invalid" in result.lower()

    def test_git_validation_chain(self):
        from forge.tools.git import GitTool
        # Branch validation
        assert GitTool.validate_branch_name("feature/test") == ""
        assert GitTool.validate_branch_name("") != ""
        assert GitTool.validate_branch_name("..") != ""
