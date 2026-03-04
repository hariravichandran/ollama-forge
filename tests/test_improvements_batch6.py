"""Tests for batch 6 improvements: web safety, orchestrator hardening, context compression, model management."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# === Web Tool: URL Validation & Safety ===

class TestWebURLValidation:
    """Tests for URL validation in WebTool."""

    def _make_tool(self):
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            return WebTool(working_dir=tmpdir)

    def test_valid_https_url(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("https://example.com/page")
        assert result == ""

    def test_valid_http_url(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://example.com")
        assert result == ""

    def test_reject_javascript_scheme(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("javascript:alert(1)")
        assert "not allowed" in result.lower()

    def test_reject_data_scheme(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("data:text/html,<h1>hi</h1>")
        assert "not allowed" in result.lower()

    def test_reject_file_scheme(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("file:///etc/passwd")
        assert "not allowed" in result.lower()

    def test_reject_ftp_scheme(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("ftp://ftp.example.com/file")
        assert "not allowed" in result.lower()

    def test_reject_localhost(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://localhost:8080/admin")
        assert "internal host" in result.lower()

    def test_reject_127_0_0_1(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://127.0.0.1/secret")
        assert "internal" in result.lower() or "private" in result.lower()

    def test_reject_private_ip_10(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://10.0.0.1/internal")
        assert "private" in result.lower()

    def test_reject_private_ip_192_168(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://192.168.1.1/router")
        assert "private" in result.lower()

    def test_reject_no_hostname(self):
        from forge.tools.web import WebTool
        result = WebTool._validate_url("http://")
        assert "no hostname" in result.lower() or "malformed" in result.lower()

    def test_reject_empty_url_in_fetch(self):
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            result = tool._fetch("")
            assert "error" in result.lower()


class TestWebRateLimiting:
    """Tests for per-domain rate limiting."""

    def test_rate_limit_tracking(self):
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            # First request sets the timestamp
            tool._domain_last_request["example.com"] = time.time()
            assert "example.com" in tool._domain_last_request

    def test_rate_limit_dict_initially_empty(self):
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            assert len(tool._domain_last_request) == 0


class TestWebConstants:
    """Tests for web tool constants."""

    def test_allowed_schemes(self):
        from forge.tools.web import ALLOWED_URL_SCHEMES
        assert "http" in ALLOWED_URL_SCHEMES
        assert "https" in ALLOWED_URL_SCHEMES
        assert "javascript" not in ALLOWED_URL_SCHEMES

    def test_max_redirects(self):
        from forge.tools.web import MAX_REDIRECTS
        assert MAX_REDIRECTS > 0
        assert MAX_REDIRECTS <= 10

    def test_max_response_bytes(self):
        from forge.tools.web import MAX_RESPONSE_BYTES
        assert MAX_RESPONSE_BYTES > 0
        assert MAX_RESPONSE_BYTES <= 50 * 1024 * 1024  # max 50MB

    def test_rate_limit_seconds(self):
        from forge.tools.web import RATE_LIMIT_SECONDS
        assert RATE_LIMIT_SECONDS >= 0.5
        assert RATE_LIMIT_SECONDS <= 60


# === Agent Orchestrator: Name Validation & Atomic Registration ===

class TestOrchestratorReservedNames:
    """Tests for reserved name checking in orchestrator."""

    def test_reserved_names_include_system(self):
        from forge.agents.orchestrator import RESERVED_NAMES
        assert "system" in RESERVED_NAMES
        assert "help" in RESERVED_NAMES
        assert "quit" in RESERVED_NAMES

    def test_builtin_agent_names(self):
        from forge.agents.orchestrator import BUILTIN_AGENT_NAMES
        assert "assistant" in BUILTIN_AGENT_NAMES
        assert "coder" in BUILTIN_AGENT_NAMES
        assert "researcher" in BUILTIN_AGENT_NAMES

    def test_max_agent_name_length(self):
        from forge.agents.orchestrator import MAX_AGENT_NAME_LENGTH
        assert MAX_AGENT_NAME_LENGTH > 0
        assert MAX_AGENT_NAME_LENGTH <= 200


class TestOrchestratorValidation:
    """Tests for agent parameter validation."""

    def test_reject_reserved_name(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "system", "desc", "prompt", None, 0.7
        )
        assert any("reserved" in e for e in errors)

    def test_reject_builtin_name(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "assistant", "desc", "prompt", None, 0.7
        )
        assert any("built-in" in e for e in errors)

    def test_reject_long_name(self):
        from forge.agents.orchestrator import AgentOrchestrator, MAX_AGENT_NAME_LENGTH
        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        errors = AgentOrchestrator._validate_agent_params(
            long_name, "desc", "prompt", None, 0.7
        )
        assert any("at most" in e for e in errors)

    def test_accept_valid_name(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "my-custom-agent", "A custom agent", "You are helpful", None, 0.7
        )
        assert len(errors) == 0

    def test_reject_empty_name(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "", "desc", "prompt", None, 0.7
        )
        assert any("empty" in e for e in errors)

    def test_reject_invalid_chars(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "agent with spaces", "desc", "prompt", None, 0.7
        )
        assert any("alphanumeric" in e for e in errors)

    def test_reject_unknown_tools(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "my-agent", "desc", "prompt", ["filesystem", "nonexistent_tool"], 0.7
        )
        assert any("unknown tools" in e for e in errors)

    def test_reject_bad_temperature(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "my-agent", "desc", "prompt", None, 5.0
        )
        assert any("temperature" in e for e in errors)

    def test_reject_reserved_name_case_insensitive(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "SYSTEM", "desc", "prompt", None, 0.7
        )
        assert any("reserved" in e for e in errors)

    def test_reject_builtin_name_case_insensitive(self):
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "CODER", "desc", "prompt", None, 0.7
        )
        assert any("built-in" in e for e in errors)


# === Context Compression: Edge Cases ===

class TestContextValidation:
    """Tests for message validation in ContextCompressor."""

    def _make_compressor(self):
        from forge.llm.context import ContextCompressor
        client = MagicMock()
        return ContextCompressor(client=client, max_tokens=4096)

    def test_validate_removes_no_role(self):
        comp = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"content": "orphan message"},  # no role
        ]
        cleaned = comp.validate_messages(messages)
        assert len(cleaned) == 1

    def test_validate_fixes_none_content(self):
        comp = self._make_compressor()
        messages = [
            {"role": "user", "content": None},
        ]
        cleaned = comp.validate_messages(messages)
        assert cleaned[0]["content"] == ""

    def test_validate_preserves_valid_messages(self):
        comp = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        cleaned = comp.validate_messages(messages)
        assert len(cleaned) == 2

    def test_validate_skips_non_dict(self):
        comp = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
            "not a dict",
            None,
        ]
        cleaned = comp.validate_messages(messages)
        assert len(cleaned) == 1

    def test_estimate_tokens_with_none_content(self):
        comp = self._make_compressor()
        messages = [{"role": "user", "content": None}]
        # Should not crash
        tokens = comp.estimate_tokens(messages)
        assert tokens >= 0

    def test_format_messages_with_none_content(self):
        comp = self._make_compressor()
        messages = [{"role": "user", "content": None}]
        result = comp._format_messages(messages)
        assert "[user]:" in result

    def test_compress_validates_before_processing(self):
        comp = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"content": "bad message"},  # no role — should be removed
        ]
        result = comp.compress(messages)
        # Should not crash; invalid messages filtered
        assert isinstance(result, list)


# === Model Management: Quantization & Headroom ===

class TestQuantizationMultipliers:
    """Tests for quantization-aware model sizing."""

    def test_q4_k_m_default(self):
        from forge.llm.models import QUANTIZATION_MULTIPLIERS, DEFAULT_QUANTIZATION
        assert DEFAULT_QUANTIZATION in QUANTIZATION_MULTIPLIERS

    def test_multipliers_increase_with_quality(self):
        from forge.llm.models import QUANTIZATION_MULTIPLIERS
        assert QUANTIZATION_MULTIPLIERS["q4_k_m"] < QUANTIZATION_MULTIPLIERS["q8_0"]
        assert QUANTIZATION_MULTIPLIERS["q8_0"] < QUANTIZATION_MULTIPLIERS["f16"]

    def test_f16_is_2x(self):
        from forge.llm.models import QUANTIZATION_MULTIPLIERS
        assert QUANTIZATION_MULTIPLIERS["f16"] == 2.0

    def test_all_multipliers_positive(self):
        from forge.llm.models import QUANTIZATION_MULTIPLIERS
        for quant, mult in QUANTIZATION_MULTIPLIERS.items():
            assert mult > 0, f"Multiplier for {quant} should be positive"


class TestEstimateModelSize:
    """Tests for estimate_model_size with quantization awareness."""

    def test_catalogue_hit(self):
        from forge.llm.models import estimate_model_size
        # Known model in catalogue
        size = estimate_model_size("llama3.1:8b")
        assert size == 5.0

    def test_unknown_model_estimation(self):
        from forge.llm.models import estimate_model_size
        # Unknown model, should estimate from param count
        size = estimate_model_size("somemodel:7b")
        assert size > 0
        assert size < 20

    def test_quantization_override(self):
        from forge.llm.models import estimate_model_size
        size_q4 = estimate_model_size("somemodel:7b", quantization="q4_k_m")
        size_f16 = estimate_model_size("somemodel:7b", quantization="f16")
        assert size_f16 > size_q4

    def test_default_estimate(self):
        from forge.llm.models import estimate_model_size
        # No parameter count in name
        size = estimate_model_size("mystery-model")
        assert size == 5.0

    def test_detects_quantization_in_name(self):
        from forge.llm.models import _detect_quantization
        assert _detect_quantization("model:q8_0") == "q8_0"
        assert _detect_quantization("model:q5_k_m") == "q5_k_m"

    def test_no_quantization_detected(self):
        from forge.llm.models import _detect_quantization
        assert _detect_quantization("llama3:7b") == ""


class TestModelNameValidation:
    """Tests for model name validation."""

    def test_valid_name(self):
        from forge.llm.models import validate_model_name
        assert validate_model_name("llama3:7b") == ""

    def test_valid_namespace_name(self):
        from forge.llm.models import validate_model_name
        assert validate_model_name("0xroyce/plutus") == ""

    def test_empty_name(self):
        from forge.llm.models import validate_model_name
        assert "empty" in validate_model_name("")

    def test_too_long_name(self):
        from forge.llm.models import validate_model_name
        result = validate_model_name("a" * 201)
        assert "too long" in result

    def test_invalid_chars(self):
        from forge.llm.models import validate_model_name
        result = validate_model_name("model with spaces")
        assert "invalid" in result


class TestDynamicHeadroom:
    """Tests for dynamic headroom calculation."""

    def test_small_gpu_headroom(self):
        from forge.llm.models import _calculate_headroom
        headroom = _calculate_headroom(4.0)
        assert headroom > 0
        assert headroom <= 1.0

    def test_medium_gpu_headroom(self):
        from forge.llm.models import _calculate_headroom
        headroom = _calculate_headroom(8.0)
        assert headroom == 1.0

    def test_large_gpu_headroom(self):
        from forge.llm.models import _calculate_headroom
        headroom = _calculate_headroom(24.0)
        assert headroom > 1.0

    def test_headroom_increases_with_size(self):
        from forge.llm.models import _calculate_headroom
        small = _calculate_headroom(4.0)
        large = _calculate_headroom(48.0)
        assert large >= small

    def test_get_models_that_fit_with_custom_headroom(self):
        from forge.llm.models import get_models_that_fit
        models = get_models_that_fit(8.0, headroom_gb=2.0)
        for m in models:
            assert m.size_gb <= 6.0


# === Integration Tests ===

class TestBatch6Integration:
    """Integration tests across batch 6 improvements."""

    def test_web_tool_fetch_validates_url(self):
        from forge.tools.web import WebTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebTool(working_dir=tmpdir)
            result = tool._fetch("javascript:alert(1)")
            assert "not allowed" in result.lower()

    def test_orchestrator_validates_on_create(self):
        """Creating an agent with reserved name should fail."""
        from forge.agents.orchestrator import AgentOrchestrator
        errors = AgentOrchestrator._validate_agent_params(
            "exit", "desc", "prompt", None, 0.7
        )
        assert len(errors) > 0

    def test_context_compressor_handles_mixed_messages(self):
        from forge.llm.context import ContextCompressor
        client = MagicMock()
        comp = ContextCompressor(client=client, max_tokens=4096)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": None},  # None content
            {"content": "no role"},              # missing role
            {"role": "assistant", "content": "response"},
        ]
        cleaned = comp.validate_messages(messages)
        assert len(cleaned) == 3  # system, user (fixed None), assistant

    def test_model_size_quantization_aware(self):
        from forge.llm.models import estimate_model_size
        # Q4 should be roughly 0.65 * 7 ≈ 4.55
        size_q4 = estimate_model_size("custom:7b", quantization="q4_k_m")
        # Q8 should be roughly 1.1 * 7 ≈ 7.7
        size_q8 = estimate_model_size("custom:7b", quantization="q8_0")
        assert size_q8 > size_q4 * 1.3  # q8 should be significantly larger
