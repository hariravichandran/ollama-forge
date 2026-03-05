"""Tests for batch 12 improvements: detect constants, codebase validation, permissions hardening, memory bounds, NL input."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# === Hardware Detect: Extracted Constants ===

class TestDetectConstants:
    """Tests for detect.py extracted constants."""

    def test_timeout_constants(self):
        from forge.hardware.detect import TIMEOUT_FAST, TIMEOUT_NORMAL, TIMEOUT_SLOW
        assert TIMEOUT_FAST < TIMEOUT_NORMAL < TIMEOUT_SLOW

    def test_vendor_id_amd(self):
        from forge.hardware.detect import AMD_VENDOR_ID
        assert AMD_VENDOR_ID == "0x1002"

    def test_vendor_id_intel(self):
        from forge.hardware.detect import INTEL_VENDOR_ID
        assert INTEL_VENDOR_ID == "0x8086"

    def test_os_memory_headroom(self):
        from forge.hardware.detect import OS_MEMORY_HEADROOM_GB
        assert OS_MEMORY_HEADROOM_GB > 0
        assert OS_MEMORY_HEADROOM_GB < 16

    def test_igpu_vram_threshold(self):
        from forge.hardware.detect import IGPU_VRAM_THRESHOLD_GB
        assert IGPU_VRAM_THRESHOLD_GB > 0


class TestDetectGPUInfo:
    """Tests for GPUInfo usable_gb using constants."""

    def test_igpu_usable_gb(self):
        from forge.hardware.detect import GPUInfo, OS_MEMORY_HEADROOM_GB
        gpu = GPUInfo(vendor="amd", is_igpu=True, total_gb=16.0)
        assert gpu.usable_gb == 16.0 - OS_MEMORY_HEADROOM_GB

    def test_apple_usable_gb(self):
        from forge.hardware.detect import GPUInfo, OS_MEMORY_HEADROOM_GB
        gpu = GPUInfo(vendor="apple", total_gb=24.0)
        assert gpu.usable_gb == 24.0 - OS_MEMORY_HEADROOM_GB

    def test_dgpu_usable_gb(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(vendor="nvidia", is_igpu=False, total_gb=12.0)
        assert gpu.usable_gb == 12.0

    def test_zero_total_gb(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(vendor="amd", is_igpu=True, total_gb=0.0)
        assert gpu.usable_gb == 0.0  # max(0, ...)


class TestDetectHardwareInfo:
    """Tests for HardwareInfo dataclass."""

    def test_default_values(self):
        from forge.hardware.detect import HardwareInfo
        hw = HardwareInfo()
        assert hw.gpu.vendor == "none"
        assert hw.cpu.threads == 1
        assert hw.ram_gb == 0.0

    def test_summary_output(self):
        from forge.hardware.detect import HardwareInfo, GPUInfo, CPUInfo
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="amd", name="Radeon 680M", total_gb=12.0, driver="rocm"),
            cpu=CPUInfo(model="Ryzen 9 6900HX", threads=16, cores=8),
            ram_gb=32.0,
        )
        summary = hw.summary()
        assert "Radeon 680M" in summary
        assert "Ryzen 9" in summary
        assert "32.0" in summary


# === Codebase: Search Validation, Constants ===

class TestCodebaseConstants:
    """Tests for codebase constants."""

    def test_max_search_results(self):
        from forge.tools.codebase import MAX_SEARCH_RESULTS
        assert MAX_SEARCH_RESULTS > 0
        assert MAX_SEARCH_RESULTS <= 1000

    def test_max_search_query_length(self):
        from forge.tools.codebase import MAX_SEARCH_QUERY_LENGTH
        assert MAX_SEARCH_QUERY_LENGTH > 0

    def test_max_symbol_display(self):
        from forge.tools.codebase import MAX_SYMBOL_DISPLAY
        assert MAX_SYMBOL_DISPLAY > 0

    def test_max_import_display(self):
        from forge.tools.codebase import MAX_IMPORT_DISPLAY
        assert MAX_IMPORT_DISPLAY > 0

    def test_max_overview_files(self):
        from forge.tools.codebase import MAX_OVERVIEW_FILES
        assert MAX_OVERVIEW_FILES > 0

    def test_max_context_preview(self):
        from forge.tools.codebase import MAX_CONTEXT_PREVIEW
        assert MAX_CONTEXT_PREVIEW > 0

    def test_max_signature_length(self):
        from forge.tools.codebase import MAX_SIGNATURE_LENGTH
        assert MAX_SIGNATURE_LENGTH > 0

    def test_max_summary_content(self):
        from forge.tools.codebase import MAX_SUMMARY_CONTENT
        assert MAX_SUMMARY_CONTENT > 0


class TestCodebaseSearchValidation:
    """Tests for codebase search input validation."""

    def test_empty_query_returns_empty(self):
        from forge.tools.codebase import CodebaseIndexer
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            indexer._loaded = True
            results = indexer.search("")
            assert results == []

    def test_whitespace_query_returns_empty(self):
        from forge.tools.codebase import CodebaseIndexer
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            indexer._loaded = True
            results = indexer.search("   ")
            assert results == []

    def test_long_query_truncated(self):
        from forge.tools.codebase import CodebaseIndexer, MAX_SEARCH_QUERY_LENGTH
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            indexer._loaded = True
            long_query = "x" * (MAX_SEARCH_QUERY_LENGTH + 100)
            # Should not crash
            results = indexer.search(long_query)
            assert isinstance(results, list)

    def test_max_results_clamped(self):
        from forge.tools.codebase import CodebaseIndexer, MAX_SEARCH_RESULTS
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            indexer._loaded = True
            # Should not crash with huge max_results
            results = indexer.search("test", max_results=99999)
            assert isinstance(results, list)


class TestCodebaseToolValidation:
    """Tests for CodebaseTool search validation."""

    def test_empty_search_query(self):
        from forge.tools.codebase import CodebaseTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = CodebaseTool(working_dir=tmpdir)
            tool._indexed = True
            tool._indexer._loaded = True
            result = tool._search("")
            assert "empty" in result.lower()

    def test_symbol_lookup(self):
        from forge.tools.codebase import CodebaseTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = CodebaseTool(working_dir=tmpdir)
            tool._indexed = True
            tool._indexer._loaded = True
            result = tool._find_symbol("NonExistentSymbol")
            assert "No symbol found" in result


class TestCodebaseIndexer:
    """Tests for CodebaseIndexer initialization and basic operations."""

    def test_init_creates_project_dir(self):
        from forge.tools.codebase import CodebaseIndexer
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            assert indexer.project_dir.exists()

    def test_detect_language(self):
        from forge.tools.codebase import CodebaseIndexer
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            assert indexer._detect_language(".py") == "python"
            assert indexer._detect_language(".js") == "javascript"
            assert indexer._detect_language(".xyz") == "unknown"

    def test_find_symbol_empty_index(self):
        from forge.tools.codebase import CodebaseIndexer
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(project_dir=tmpdir)
            indexer._loaded = True
            assert indexer.find_symbol("anything") == []


# === Permissions: Action Validation, Constants ===

class TestPermissionsConstants:
    """Tests for permissions constants."""

    def test_rate_limit_window(self):
        from forge.agents.permissions import RATE_LIMIT_WINDOW_S
        assert RATE_LIMIT_WINDOW_S > 0

    def test_max_context_value_display(self):
        from forge.agents.permissions import MAX_CONTEXT_VALUE_DISPLAY
        assert MAX_CONTEXT_VALUE_DISPLAY > 0

    def test_max_context_value_audit(self):
        from forge.agents.permissions import MAX_CONTEXT_VALUE_AUDIT
        assert MAX_CONTEXT_VALUE_AUDIT > 0

    def test_max_action_name_length(self):
        from forge.agents.permissions import MAX_ACTION_NAME_LENGTH
        assert MAX_ACTION_NAME_LENGTH > 0


class TestPermissionsActionValidation:
    """Tests for action name validation."""

    def test_empty_action_denied(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        assert pm.check("") is False

    def test_very_long_action_denied(self):
        from forge.agents.permissions import PermissionManager, MAX_ACTION_NAME_LENGTH
        pm = PermissionManager()
        long_action = "x" * (MAX_ACTION_NAME_LENGTH + 1)
        assert pm.check(long_action) is False

    def test_normal_action_works(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager(auto_approve_all=True)
        assert pm.check("read_file") is True

    def test_auto_approve_all(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager(auto_approve_all=True)
        assert pm.check("run_command") is True


class TestPermissionsDangerDetection:
    """Tests for dangerous command detection."""

    def test_rm_rf_root_detected(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        danger = pm._detect_dangerous("run_command", {"command": "rm -rf /"})
        assert danger != ""

    def test_safe_command_not_flagged(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        danger = pm._detect_dangerous("run_command", {"command": "ls -la"})
        assert danger == ""

    def test_no_context_safe(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        danger = pm._detect_dangerous("read_file", None)
        assert danger == ""


class TestPermissionsRateLimit:
    """Tests for permission rate limiting."""

    def test_rate_limit_check(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        assert pm._is_rate_limited() is False

    def test_rate_limit_after_many_prompts(self):
        from forge.agents.permissions import PermissionManager, MAX_PROMPTS_PER_MINUTE
        pm = PermissionManager()
        pm._prompt_timestamps = [time.time()] * MAX_PROMPTS_PER_MINUTE
        assert pm._is_rate_limited() is True


class TestPermissionsRedaction:
    """Tests for secret redaction."""

    def test_api_key_redacted(self):
        from forge.agents.permissions import PermissionManager
        result = PermissionManager._redact_secrets("api_key=sk_12345_secret")
        assert "***REDACTED***" in result
        assert "sk_12345" not in result

    def test_no_secrets_unchanged(self):
        from forge.agents.permissions import PermissionManager
        result = PermissionManager._redact_secrets("just normal text")
        assert result == "just normal text"


class TestPermissionsAuditLog:
    """Tests for audit logging."""

    def test_audit_log_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.permissions import PermissionManager
            audit_file = Path(tmpdir) / "audit.log"
            pm = PermissionManager(auto_approve_all=True, audit_file=audit_file)
            pm.check("test_action", {"key": "value"})
            assert audit_file.exists()
            content = audit_file.read_text()
            assert "test_action" in content

    def test_audit_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.permissions import PermissionManager
            audit_file = Path(tmpdir) / "audit.log"
            pm = PermissionManager(auto_approve_all=True, audit_file=audit_file)
            pm.check("read_file")
            pm.check("write_file")
            stats = pm.get_audit_stats()
            assert stats["entries"] == 2


# === Memory: Confidence Bounds, Session Constants ===

class TestMemoryConstants:
    """Tests for memory constants."""

    def test_min_confidence(self):
        from forge.agents.memory import MIN_CONFIDENCE
        assert MIN_CONFIDENCE >= 0.0

    def test_max_confidence(self):
        from forge.agents.memory import MAX_CONFIDENCE
        assert MAX_CONFIDENCE <= 1.0

    def test_max_recent_sessions(self):
        from forge.agents.memory import MAX_RECENT_SESSIONS
        assert MAX_RECENT_SESSIONS > 0

    def test_max_summary_length(self):
        from forge.agents.memory import MAX_SUMMARY_LENGTH
        assert MAX_SUMMARY_LENGTH > 0


class TestMemoryConfidenceClamping:
    """Tests for confidence bounds in store_fact."""

    def test_confidence_above_max_clamped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.memory import ConversationMemory
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key1", "value1", confidence=5.0)
            fact = mem._facts["key1"]
            assert fact.confidence <= 1.0

    def test_confidence_below_min_clamped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.memory import ConversationMemory
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key1", "value1", confidence=-1.0)
            fact = mem._facts["key1"]
            assert fact.confidence >= 0.0

    def test_normal_confidence_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.memory import ConversationMemory
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key1", "value1", confidence=0.7)
            assert mem._facts["key1"].confidence == 0.7


class TestMemorySummaryLength:
    """Tests for summary length validation."""

    def test_normal_summary_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.memory import ConversationMemory
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.save_summary("Short summary")
            assert mem.get_summary() == "Short summary"

    def test_oversized_summary_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from forge.agents.memory import ConversationMemory, MAX_SUMMARY_LENGTH
            mem = ConversationMemory(memory_dir=tmpdir)
            long_summary = "x" * (MAX_SUMMARY_LENGTH + 1000)
            mem.save_summary(long_summary)
            result = mem.get_summary()
            assert len(result) <= MAX_SUMMARY_LENGTH


# === Natural Language: Input Validation, Constants ===

class TestNaturalLanguageConstants:
    """Tests for natural language constants."""

    def test_max_request_text_length(self):
        from forge.mcp.natural_language import MAX_REQUEST_TEXT_LENGTH
        assert MAX_REQUEST_TEXT_LENGTH > 0

    def test_max_mcp_list_display(self):
        from forge.mcp.natural_language import MAX_MCP_LIST_DISPLAY
        assert MAX_MCP_LIST_DISPLAY > 0

    def test_mcp_name_column_width(self):
        from forge.mcp.natural_language import MCP_NAME_COLUMN_WIDTH
        assert MCP_NAME_COLUMN_WIDTH > 0


class TestParseRequest:
    """Tests for parse_mcp_request input validation."""

    def test_empty_text_returns_none_action(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("")
        assert result["action"] is None

    def test_whitespace_returns_none_action(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("   ")
        assert result["action"] is None

    def test_add_action_detected(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("add a web search tool")
        assert result["action"] == "add"

    def test_remove_action_detected(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("remove the github integration")
        assert result["action"] == "remove"

    def test_list_action_detected(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("list available MCPs")
        assert result["action"] == "list"

    def test_search_action_detected(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("search for database tools")
        assert result["action"] == "search"

    def test_suggest_action_detected(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("suggest some useful tools")
        assert result["action"] == "suggest"

    def test_long_text_truncated(self):
        from forge.mcp.natural_language import parse_mcp_request, MAX_REQUEST_TEXT_LENGTH
        long_text = "add " + "x" * (MAX_REQUEST_TEXT_LENGTH + 100)
        result = parse_mcp_request(long_text)
        assert result["action"] == "add"

    def test_keyword_matching(self):
        from forge.mcp.natural_language import parse_mcp_request
        result = parse_mcp_request("add web search")
        assert result["mcp_name"] == "web-search"


class TestHandleRequest:
    """Tests for handle_mcp_request input validation."""

    def test_empty_text_returns_message(self):
        from forge.mcp.natural_language import handle_mcp_request
        mock_manager = MagicMock()
        result = handle_mcp_request(mock_manager, "")
        assert "provide" in result.lower() or "please" in result.lower()

    def test_whitespace_returns_message(self):
        from forge.mcp.natural_language import handle_mcp_request
        mock_manager = MagicMock()
        result = handle_mcp_request(mock_manager, "   ")
        assert len(result) > 0


# === Integration Tests ===

class TestBatch12Integration:
    """Integration tests across batch 12 improvements."""

    def test_detect_constants_importable(self):
        from forge.hardware.detect import (
            TIMEOUT_FAST, TIMEOUT_NORMAL, TIMEOUT_SLOW,
            AMD_VENDOR_ID, INTEL_VENDOR_ID,
            OS_MEMORY_HEADROOM_GB, IGPU_VRAM_THRESHOLD_GB,
        )
        assert TIMEOUT_FAST > 0
        assert AMD_VENDOR_ID.startswith("0x")

    def test_codebase_constants_importable(self):
        from forge.tools.codebase import (
            MAX_SEARCH_RESULTS, MAX_SEARCH_QUERY_LENGTH,
            MAX_SYMBOL_DISPLAY, MAX_IMPORT_DISPLAY,
            MAX_CONTEXT_PREVIEW, MAX_SIGNATURE_LENGTH,
        )
        assert all(v > 0 for v in [
            MAX_SEARCH_RESULTS, MAX_SEARCH_QUERY_LENGTH,
            MAX_SYMBOL_DISPLAY, MAX_IMPORT_DISPLAY,
        ])

    def test_permissions_constants_importable(self):
        from forge.agents.permissions import (
            RATE_LIMIT_WINDOW_S, MAX_CONTEXT_VALUE_DISPLAY,
            MAX_CONTEXT_VALUE_AUDIT, MAX_ACTION_NAME_LENGTH,
        )
        assert all(v > 0 for v in [
            RATE_LIMIT_WINDOW_S, MAX_CONTEXT_VALUE_DISPLAY,
            MAX_CONTEXT_VALUE_AUDIT, MAX_ACTION_NAME_LENGTH,
        ])

    def test_memory_constants_importable(self):
        from forge.agents.memory import (
            MIN_CONFIDENCE, MAX_CONFIDENCE,
            MAX_RECENT_SESSIONS, MAX_SUMMARY_LENGTH,
        )
        assert MIN_CONFIDENCE >= 0
        assert MAX_CONFIDENCE <= 1.0
        assert MAX_RECENT_SESSIONS > 0

    def test_nl_constants_importable(self):
        from forge.mcp.natural_language import (
            MAX_REQUEST_TEXT_LENGTH, MAX_MCP_LIST_DISPLAY,
            MCP_NAME_COLUMN_WIDTH,
        )
        assert all(v > 0 for v in [
            MAX_REQUEST_TEXT_LENGTH, MAX_MCP_LIST_DISPLAY,
            MCP_NAME_COLUMN_WIDTH,
        ])
