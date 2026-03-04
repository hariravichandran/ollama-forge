"""Tests for batch 7 improvements: sandbox metrics, permissions hardening, ROCm validation, CLI export."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# === Sandbox: Input Validation & Execution Metrics ===

class TestSandboxValidation:
    """Tests for sandbox input validation."""

    def test_timeout_clamped_to_min(self):
        from forge.tools.sandbox import Sandbox, MIN_TIMEOUT
        sb = Sandbox(timeout=0)
        assert sb.timeout == MIN_TIMEOUT

    def test_timeout_clamped_to_max(self):
        from forge.tools.sandbox import Sandbox, MAX_TIMEOUT
        sb = Sandbox(timeout=9999)
        assert sb.timeout == MAX_TIMEOUT

    def test_valid_timeout_passes(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(timeout=60)
        assert sb.timeout == 60

    def test_memory_clamped_to_min(self):
        from forge.tools.sandbox import Sandbox, MIN_MEMORY_MB
        sb = Sandbox(max_memory_mb=1)
        assert sb.max_memory_mb == MIN_MEMORY_MB

    def test_memory_clamped_to_max(self):
        from forge.tools.sandbox import Sandbox, MAX_MEMORY_MB
        sb = Sandbox(max_memory_mb=999999)
        assert sb.max_memory_mb == MAX_MEMORY_MB

    def test_valid_memory_passes(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(max_memory_mb=512)
        assert sb.max_memory_mb == 512

    def test_max_output_clamped(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(max_output=10)
        assert sb.max_output == 100  # min 100


class TestSandboxMetrics:
    """Tests for sandbox execution metrics tracking."""

    def test_initial_metrics_zero(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        metrics = sb.get_metrics()
        assert metrics["executions"] == 0
        assert metrics["timeouts"] == 0
        assert metrics["errors"] == 0

    def test_metrics_after_execution(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(timeout=5)
        sb.run_python("print('hello')")
        metrics = sb.get_metrics()
        assert metrics["executions"] == 1
        assert metrics["total_duration_s"] > 0

    def test_timeout_rate(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        # No executions — rate should be 0
        assert sb.get_metrics()["timeout_rate"] == 0.0

    def test_avg_duration(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox()
        sb.run_python("pass")
        metrics = sb.get_metrics()
        assert metrics["avg_duration_s"] > 0


class TestExecutionResultMemory:
    """Tests for peak_memory_mb field."""

    def test_default_peak_memory(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=0.1)
        assert result.peak_memory_mb == 0.0

    def test_peak_memory_field_exists(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=0.1, peak_memory_mb=42.5)
        assert result.peak_memory_mb == 42.5

    def test_get_peak_memory_static(self):
        from forge.tools.sandbox import Sandbox
        # Non-existent PID should return 0
        result = Sandbox._get_peak_memory(999999999)
        assert result == 0.0


class TestSandboxSymlinkValidation:
    """Tests for project symlink validation."""

    def test_nonexistent_project_dir_no_crash(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(allow_project_read=True, project_dir="/nonexistent/path/xyz")
        result = sb.run_python("print('hello')")
        assert result.success


# === Permissions: Audit Log Rotation, Secret Redaction, Rate Limiting ===

class TestSecretRedaction:
    """Tests for secret redaction in audit logs."""

    def test_redact_api_key(self):
        from forge.agents.permissions import PermissionManager
        text = "api_key=sk-1234567890abcdef"
        result = PermissionManager._redact_secrets(text)
        assert "sk-1234567890" not in result
        assert "REDACTED" in result

    def test_redact_password(self):
        from forge.agents.permissions import PermissionManager
        text = "password=mysecretpassword"
        result = PermissionManager._redact_secrets(text)
        assert "mysecretpassword" not in result

    def test_redact_bearer_token(self):
        from forge.agents.permissions import PermissionManager
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.token"
        result = PermissionManager._redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_redact_github_token(self):
        from forge.agents.permissions import PermissionManager
        text = "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz123456"
        result = PermissionManager._redact_secrets(text)
        assert "ghp_abcdefghijklmnop" not in result

    def test_no_redaction_for_normal_text(self):
        from forge.agents.permissions import PermissionManager
        text = "Just a normal command: ls -la /home"
        result = PermissionManager._redact_secrets(text)
        assert result == text

    def test_redact_multiple_secrets(self):
        from forge.agents.permissions import PermissionManager
        text = "api_key=secret1 password=secret2"
        result = PermissionManager._redact_secrets(text)
        assert "secret1" not in result
        assert "secret2" not in result


class TestPermissionRateLimiting:
    """Tests for permission prompt rate limiting."""

    def test_not_rate_limited_initially(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager(prompt_fn=lambda _: True)
        assert pm._is_rate_limited() is False

    def test_rate_limited_after_many_prompts(self):
        from forge.agents.permissions import PermissionManager, MAX_PROMPTS_PER_MINUTE
        pm = PermissionManager(prompt_fn=lambda _: True)
        # Simulate many rapid prompts
        pm._prompt_timestamps = [time.time()] * MAX_PROMPTS_PER_MINUTE
        assert pm._is_rate_limited() is True

    def test_rate_limit_expires(self):
        from forge.agents.permissions import PermissionManager, MAX_PROMPTS_PER_MINUTE
        pm = PermissionManager(prompt_fn=lambda _: True)
        # Simulate old prompts (more than 60 seconds ago)
        old_time = time.time() - 120
        pm._prompt_timestamps = [old_time] * MAX_PROMPTS_PER_MINUTE
        assert pm._is_rate_limited() is False

    def test_rate_limit_constant(self):
        from forge.agents.permissions import MAX_PROMPTS_PER_MINUTE
        assert MAX_PROMPTS_PER_MINUTE > 0
        assert MAX_PROMPTS_PER_MINUTE <= 100


class TestAuditLogRotation:
    """Tests for audit log rotation."""

    def test_audit_stats_empty(self):
        from forge.agents.permissions import PermissionManager
        pm = PermissionManager()
        stats = pm.get_audit_stats()
        assert stats["entries"] == 0

    def test_audit_stats_with_entries(self):
        from forge.agents.permissions import PermissionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.log"
            pm = PermissionManager(audit_file=str(audit_path), auto_approve_all=True)
            pm.check("read_file", {"path": "/tmp/test"})
            pm.check("write_file", {"path": "/tmp/test"})
            stats = pm.get_audit_stats()
            assert stats["entries"] == 2
            assert "auto_approved" in stats["decisions"]

    def test_audit_log_redacts_secrets(self):
        from forge.agents.permissions import PermissionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.log"
            pm = PermissionManager(audit_file=str(audit_path), auto_approve_all=True)
            pm.check("run_command", {"command": "curl -H 'api_key=sk-12345' http://example.com"})
            content = audit_path.read_text()
            assert "sk-12345" not in content
            assert "REDACTED" in content

    def test_rotate_audit_log(self):
        from forge.agents.permissions import PermissionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.log"
            # Create a small log
            entries = [json.dumps({"ts": time.time(), "action": "test", "decision": "auto"}) + "\n"
                       for _ in range(10)]
            audit_path.write_text("".join(entries))
            pm = PermissionManager(audit_file=str(audit_path))
            pm._rotate_audit_log()
            # Below threshold — should not rotate
            assert audit_path.read_text() != ""


class TestAuditLogConstants:
    """Tests for audit log constants."""

    def test_max_entries_reasonable(self):
        from forge.agents.permissions import MAX_AUDIT_LOG_ENTRIES
        assert MAX_AUDIT_LOG_ENTRIES > 0
        assert MAX_AUDIT_LOG_ENTRIES <= 10_000_000

    def test_check_interval(self):
        from forge.agents.permissions import AUDIT_LOG_CHECK_INTERVAL
        assert AUDIT_LOG_CHECK_INTERVAL > 0

    def test_secret_patterns_list(self):
        from forge.agents.permissions import SECRET_PATTERNS
        assert len(SECRET_PATTERNS) > 0


# === ROCm: GFX Override Validation ===

class TestGFXOverrideValidation:
    """Tests for HSA_OVERRIDE_GFX_VERSION validation."""

    def test_valid_override_10_3_0(self):
        from forge.hardware.rocm import validate_gfx_override
        assert validate_gfx_override("10.3.0") == ""

    def test_valid_override_11_0_0(self):
        from forge.hardware.rocm import validate_gfx_override
        assert validate_gfx_override("11.0.0") == ""

    def test_valid_override_12_0_0(self):
        from forge.hardware.rocm import validate_gfx_override
        assert validate_gfx_override("12.0.0") == ""

    def test_invalid_format(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("abc")
        assert "invalid format" in result.lower()

    def test_unknown_target(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("99.99.99")
        assert "unknown" in result.lower()

    def test_empty_string(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("")
        assert "empty" in result.lower()

    def test_partial_format(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("10.3")
        assert "invalid format" in result.lower()


class TestROCmGroups:
    """Tests for ROCm group constants."""

    def test_required_groups(self):
        from forge.hardware.rocm import ROCM_REQUIRED_GROUPS
        assert "render" in ROCM_REQUIRED_GROUPS
        assert "video" in ROCM_REQUIRED_GROUPS

    def test_optional_groups(self):
        from forge.hardware.rocm import ROCM_OPTIONAL_GROUPS
        assert "compute" in ROCM_OPTIONAL_GROUPS


# === CLI: Batch Mode & Export ===

class TestCLIExportCommand:
    """Tests for the /export command handler."""

    def test_export_json(self):
        from forge.cli import _handle_export_command
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                _handle_export_command("/export json", messages, "assistant")
                files = list(Path(tmpdir).glob("conversation_*.json"))
                assert len(files) == 1
                data = json.loads(files[0].read_text())
                assert len(data) == 2
            finally:
                os.chdir(old_cwd)

    def test_export_markdown(self):
        from forge.cli import _handle_export_command
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                _handle_export_command("/export markdown", messages, "assistant")
                files = list(Path(tmpdir).glob("conversation_*.md"))
                assert len(files) == 1
                content = files[0].read_text()
                assert "# Conversation" in content
                assert "hello" in content
            finally:
                os.chdir(old_cwd)

    def test_export_txt(self):
        from forge.cli import _handle_export_command
        messages = [
            {"role": "user", "content": "hello"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                _handle_export_command("/export txt", messages, "assistant")
                files = list(Path(tmpdir).glob("conversation_*.txt"))
                assert len(files) == 1
                content = files[0].read_text()
                assert "[USER]" in content
            finally:
                os.chdir(old_cwd)

    def test_export_default_is_markdown(self):
        from forge.cli import _handle_export_command
        messages = [{"role": "user", "content": "test"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                _handle_export_command("/export", messages, "assistant")
                files = list(Path(tmpdir).glob("conversation_*.md"))
                assert len(files) == 1
            finally:
                os.chdir(old_cwd)

    def test_export_handles_none_content(self):
        from forge.cli import _handle_export_command
        messages = [{"role": "user", "content": None}]
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Should not crash
                _handle_export_command("/export json", messages, "assistant")
                files = list(Path(tmpdir).glob("conversation_*.json"))
                assert len(files) == 1
            finally:
                os.chdir(old_cwd)


# === Integration Tests ===

class TestBatch7Integration:
    """Integration tests across batch 7 improvements."""

    def test_sandbox_metrics_accumulate(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(timeout=5)
        sb.run_python("print(1)")
        sb.run_python("print(2)")
        metrics = sb.get_metrics()
        assert metrics["executions"] == 2

    def test_permissions_redaction_in_audit(self):
        from forge.agents.permissions import PermissionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.log"
            pm = PermissionManager(audit_file=str(audit_path), auto_approve_all=True)
            pm.check("run_command", {"command": "export token=ghp_mysecrettoken123"})
            content = audit_path.read_text()
            assert "ghp_mysecrettoken123" not in content

    def test_rocm_validation_catches_bad_overrides(self):
        from forge.hardware.rocm import validate_gfx_override
        # Common user mistakes
        assert validate_gfx_override("10.3") != ""   # missing patch
        assert validate_gfx_override("gfx1035") != ""  # GFX ID, not version
        assert validate_gfx_override("10.3.0") == ""   # correct

    def test_sandbox_timeout_counted(self):
        from forge.tools.sandbox import Sandbox
        sb = Sandbox(timeout=1)
        # This will timeout
        sb.run_python("import time; time.sleep(10)")
        metrics = sb.get_metrics()
        assert metrics["timeouts"] == 1
        assert metrics["timeout_rate"] > 0
