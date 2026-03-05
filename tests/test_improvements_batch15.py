"""Tests for batch 15 improvements: rocm constants, manager truncation, rules size limit, prompts default, terminal display."""

import tempfile
from pathlib import Path

import pytest


# === ROCm: Constants ===

class TestRocmConstants:
    """Tests for ROCm constants."""

    def test_rocminfo_timeout(self):
        from forge.hardware.rocm import ROCMINFO_TIMEOUT
        assert ROCMINFO_TIMEOUT > 0

    def test_groups_timeout(self):
        from forge.hardware.rocm import GROUPS_TIMEOUT
        assert GROUPS_TIMEOUT > 0

    def test_rocm_version_file(self):
        from forge.hardware.rocm import ROCM_VERSION_FILE
        assert str(ROCM_VERSION_FILE).endswith("version")

    def test_igpu_max_loaded_models(self):
        from forge.hardware.rocm import IGPU_MAX_LOADED_MODELS
        assert IGPU_MAX_LOADED_MODELS == "1"

    def test_dgpu_max_loaded_models(self):
        from forge.hardware.rocm import DGPU_MAX_LOADED_MODELS
        assert DGPU_MAX_LOADED_MODELS == "2"

    def test_gfx_overrides_non_empty(self):
        from forge.hardware.rocm import GFX_OVERRIDES
        assert len(GFX_OVERRIDES) > 0

    def test_rocm_required_groups(self):
        from forge.hardware.rocm import ROCM_REQUIRED_GROUPS
        assert "render" in ROCM_REQUIRED_GROUPS
        assert "video" in ROCM_REQUIRED_GROUPS


class TestRocmValidation:
    """Tests for ROCm GFX validation."""

    def test_validate_empty(self):
        from forge.hardware.rocm import validate_gfx_override
        assert validate_gfx_override("") != ""

    def test_validate_invalid_format(self):
        from forge.hardware.rocm import validate_gfx_override
        assert "Invalid format" in validate_gfx_override("abc")

    def test_validate_valid_known(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("10.3.0")
        assert result == ""

    def test_validate_unknown_target(self):
        from forge.hardware.rocm import validate_gfx_override
        result = validate_gfx_override("99.99.99")
        assert "Unknown" in result


class TestRocmEnvConfig:
    """Tests for ROCm environment configuration."""

    def test_configure_non_amd_returns_empty(self):
        from forge.hardware.detect import GPUInfo
        from forge.hardware.rocm import configure_rocm_env
        gpu = GPUInfo(name="NVIDIA GTX", vendor="nvidia")
        result = configure_rocm_env(gpu)
        assert result == {}

    def test_configure_non_rocm_returns_empty(self):
        from forge.hardware.detect import GPUInfo
        from forge.hardware.rocm import configure_rocm_env
        gpu = GPUInfo(name="AMD GPU", vendor="amd", driver="mesa")
        result = configure_rocm_env(gpu)
        assert result == {}


# === MCP Manager: Constants ===

class TestMCPManagerConstants:
    """Tests for MCP manager constants."""

    def test_install_max_retries(self):
        from forge.mcp.manager import MCP_INSTALL_MAX_RETRIES
        assert MCP_INSTALL_MAX_RETRIES > 0

    def test_install_backoff_base(self):
        from forge.mcp.manager import MCP_INSTALL_BACKOFF_BASE
        assert MCP_INSTALL_BACKOFF_BASE > 0

    def test_install_timeout(self):
        from forge.mcp.manager import MCP_INSTALL_TIMEOUT
        assert MCP_INSTALL_TIMEOUT > 0

    def test_health_check_timeout(self):
        from forge.mcp.manager import MCP_HEALTH_CHECK_TIMEOUT
        assert MCP_HEALTH_CHECK_TIMEOUT > 0

    def test_max_mcp_name_length(self):
        from forge.mcp.manager import MAX_MCP_NAME_LENGTH
        assert MAX_MCP_NAME_LENGTH > 0

    def test_max_install_cmd_display(self):
        from forge.mcp.manager import MAX_INSTALL_CMD_DISPLAY
        assert MAX_INSTALL_CMD_DISPLAY > 0

    def test_max_stderr_display(self):
        from forge.mcp.manager import MAX_STDERR_DISPLAY
        assert MAX_STDERR_DISPLAY > 0


class TestMCPManagerValidation:
    """Tests for MCP manager validation."""

    def test_validate_empty_name(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("")
        assert "empty" in result.lower()

    def test_validate_long_name(self):
        from forge.mcp.manager import MCPManager, MAX_MCP_NAME_LENGTH
        result = MCPManager.validate_mcp_name("a" * (MAX_MCP_NAME_LENGTH + 1))
        assert "too long" in result.lower()

    def test_validate_invalid_name(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("123-invalid")
        assert "Invalid" in result

    def test_validate_valid_name(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager.validate_mcp_name("my-mcp-server")
        assert result == ""

    def test_validate_dangerous_install_cmd(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager._validate_install_cmd("pip install foo && rm -rf /")
        assert "dangerous" in result.lower()

    def test_validate_safe_install_cmd(self):
        from forge.mcp.manager import MCPManager
        result = MCPManager._validate_install_cmd("pip install mcp-server-fetch")
        assert result == ""


# === Rules: Size Limit ===

class TestRulesConstants:
    """Tests for rules constants."""

    def test_rules_filenames(self):
        from forge.agents.rules import RULES_FILENAMES
        assert ".forge-rules" in RULES_FILENAMES
        assert "CLAUDE.md" in RULES_FILENAMES

    def test_max_rules_file_size(self):
        from forge.agents.rules import MAX_RULES_FILE_SIZE
        assert MAX_RULES_FILE_SIZE > 0


class TestRulesLoading:
    """Tests for rules file loading."""

    def test_load_from_directory(self):
        from forge.agents.rules import load_project_rules
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / ".forge-rules"
            rules_file.write_text("# Test rules\nBe concise.")
            rules = load_project_rules(tmpdir)
            assert "Test rules" in rules

    def test_load_nonexistent_directory(self):
        from forge.agents.rules import load_project_rules
        with tempfile.TemporaryDirectory() as tmpdir:
            rules = load_project_rules(tmpdir)
            # No rules files → empty (after global check)
            # Just verify it doesn't crash
            assert isinstance(rules, str)

    def test_oversized_rules_file_skipped(self):
        from forge.agents.rules import _read_rules_file, MAX_RULES_FILE_SIZE
        with tempfile.TemporaryDirectory() as tmpdir:
            big_file = Path(tmpdir) / "rules.md"
            big_file.write_text("x" * (MAX_RULES_FILE_SIZE + 1))
            result = _read_rules_file(big_file)
            assert result == ""

    def test_normal_rules_file_read(self):
        from forge.agents.rules import _read_rules_file
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / "rules.md"
            rules_file.write_text("Be concise and practical.")
            result = _read_rules_file(rules_file)
            assert "concise" in result

    def test_create_rules_template(self):
        from forge.agents.rules import create_rules_template
        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_rules_template(tmpdir)
            assert "Created" in result
            assert (Path(tmpdir) / ".forge-rules").exists()

    def test_create_rules_template_exists(self):
        from forge.agents.rules import create_rules_template
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("existing")
            result = create_rules_template(tmpdir)
            assert "already exists" in result


# === Prompts: Default System Prompt ===

class TestPromptsConstants:
    """Tests for prompts constants."""

    def test_default_system_prompt(self):
        from forge.agents.prompts import DEFAULT_SYSTEM_PROMPT
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert "assistant" in DEFAULT_SYSTEM_PROMPT.lower()


class TestPromptsGetPrompt:
    """Tests for get_prompt function."""

    def test_known_template(self):
        from forge.agents.prompts import get_prompt
        prompt = get_prompt("coder")
        assert "developer" in prompt.lower() or "code" in prompt.lower()

    def test_unknown_template_returns_default(self):
        from forge.agents.prompts import get_prompt, DEFAULT_SYSTEM_PROMPT
        prompt = get_prompt("nonexistent_template_xyz")
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_template_with_variables(self):
        from forge.agents.prompts import get_prompt
        prompt = get_prompt("coder", language="python")
        assert "python" in prompt.lower()

    def test_template_without_variables(self):
        from forge.agents.prompts import get_prompt
        prompt = get_prompt("reviewer")
        assert "review" in prompt.lower()

    def test_list_templates(self):
        from forge.agents.prompts import list_templates
        templates = list_templates()
        assert len(templates) > 0
        names = {t["name"] for t in templates}
        assert "coder" in names
        assert "reviewer" in names


# === Terminal: Display Constants ===

class TestTerminalConstants:
    """Tests for terminal display constants."""

    def test_max_models_display(self):
        from forge.ui.terminal import MAX_MODELS_DISPLAY
        assert MAX_MODELS_DISPLAY > 0


# === Integration Tests ===

class TestBatch15Integration:
    """Integration tests verifying all batch 15 constants are importable."""

    def test_rocm_constants_importable(self):
        from forge.hardware.rocm import (
            ROCMINFO_TIMEOUT, GROUPS_TIMEOUT,
            ROCM_VERSION_FILE, IGPU_MAX_LOADED_MODELS, DGPU_MAX_LOADED_MODELS,
        )

    def test_manager_constants_importable(self):
        from forge.mcp.manager import (
            MCP_INSTALL_MAX_RETRIES, MCP_INSTALL_BACKOFF_BASE,
            MCP_INSTALL_TIMEOUT, MCP_HEALTH_CHECK_TIMEOUT,
            MAX_MCP_NAME_LENGTH, MAX_INSTALL_CMD_DISPLAY, MAX_STDERR_DISPLAY,
        )

    def test_rules_constants_importable(self):
        from forge.agents.rules import (
            RULES_FILENAMES, GLOBAL_RULES_PATH, MAX_RULES_FILE_SIZE,
        )

    def test_prompts_constants_importable(self):
        from forge.agents.prompts import DEFAULT_SYSTEM_PROMPT

    def test_terminal_constants_importable(self):
        from forge.ui.terminal import MAX_MODELS_DISPLAY
