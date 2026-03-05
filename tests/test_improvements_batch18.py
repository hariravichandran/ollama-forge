"""Tests for batch 18 improvements: remaining bare except:pass replaced with logging.

Verifies that hardware detection, ROCm, LLM client, web search, MCP manager,
and web tool modules all use proper logging instead of silent except:pass.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest


class TestDetectLogging:
    """Tests for hardware detect module logging."""

    def test_detect_module_has_logger(self):
        from forge.hardware.detect import log
        assert log is not None
        assert "hardware.detect" in log.name

    def test_detect_apple_gpu_cores_logged(self):
        """The GPU cores parsing exception should be logged, not silenced."""
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_apple_gpu)
        assert "log.debug" in source
        # Should not have bare pass after except
        assert "except (json.JSONDecodeError, KeyError) as e:" in source

    def test_detect_amd_gpu_name_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._read_amd_gpu_name)
        assert "log.debug" in source

    def test_detect_rocm_version_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_rocm_version)
        assert "log.debug" in source

    def test_detect_intel_gpu_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_intel_gpu_linux)
        assert "log.debug" in source

    def test_detect_intel_vram_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_intel_vram_linux)
        assert "log.debug" in source

    def test_detect_gpu_windows_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_gpu_windows)
        assert "log.debug" in source

    def test_detect_ram_macos_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_ram_macos)
        assert "log.debug" in source

    def test_detect_ram_windows_logged(self):
        import forge.hardware.detect as mod
        import inspect
        source = inspect.getsource(mod._detect_ram_windows)
        assert "log.debug" in source


class TestRocmLogging:
    """Tests for ROCm module logging."""

    def test_rocm_module_has_logger(self):
        from forge.hardware.rocm import log
        assert log is not None
        assert "hardware.rocm" in log.name

    def test_get_gfx_from_rocminfo_logged(self):
        import forge.hardware.rocm as mod
        import inspect
        source = inspect.getsource(mod._get_gfx_from_rocminfo)
        assert "log.debug" in source
        assert "except (FileNotFoundError, subprocess.TimeoutExpired) as e:" in source

    def test_get_rocm_status_groups_logged(self):
        import forge.hardware.rocm as mod
        import inspect
        source = inspect.getsource(mod.get_rocm_status)
        assert "log.debug" in source


class TestClientLogging:
    """Tests for LLM client module logging."""

    def test_client_module_has_logger(self):
        from forge.llm.client import log
        assert log is not None
        assert "llm.client" in log.name

    def test_show_model_logged(self):
        from forge.llm.client import OllamaClient
        import inspect
        source = inspect.getsource(OllamaClient.show_model)
        assert "log.debug" in source
        assert "except (requests.ConnectionError, requests.Timeout) as e:" in source


class TestWebSearchLogging:
    """Tests for web search MCP module logging."""

    def test_web_search_module_has_logger(self):
        from forge.mcp.web_search import log
        assert log is not None
        assert "mcp.web_search" in log.name

    def test_load_cache_logged(self):
        from forge.mcp.web_search import WebSearchMCP
        import inspect
        source = inspect.getsource(WebSearchMCP._load_cache)
        assert "log.debug" in source

    def test_save_cache_logged(self):
        from forge.mcp.web_search import WebSearchMCP
        import inspect
        source = inspect.getsource(WebSearchMCP._save_cache)
        assert "log.debug" in source


class TestMCPManagerLogging:
    """Tests for MCP manager module logging."""

    def test_manager_module_has_logger(self):
        from forge.mcp.manager import log
        assert log is not None
        assert "mcp.manager" in log.name

    def test_load_config_logged(self):
        from forge.mcp.manager import MCPManager
        import inspect
        source = inspect.getsource(MCPManager._load_config)
        assert "log.debug" in source


class TestWebToolLogging:
    """Tests for web tool module logging."""

    def test_web_tool_module_has_logger(self):
        from forge.tools.web import log
        assert log is not None
        assert "tools.web" in log.name

    def test_web_load_cache_logged(self):
        from forge.tools.web import WebTool
        import inspect
        source = inspect.getsource(WebTool._load_cache)
        assert "log.debug" in source

    def test_web_save_cache_logged(self):
        from forge.tools.web import WebTool
        import inspect
        source = inspect.getsource(WebTool._save_cache)
        assert "log.debug" in source


class TestNoBareExceptPass:
    """Verify no bare except:pass remains in modified modules."""

    @pytest.mark.parametrize("module_path", [
        "forge/hardware/detect.py",
        "forge/hardware/rocm.py",
        "forge/llm/client.py",
        "forge/mcp/web_search.py",
        "forge/mcp/manager.py",
        "forge/tools/web.py",
    ])
    def test_no_except_pass_without_logging(self, module_path):
        """Each except block should log the exception, not just pass."""
        content = Path(module_path).read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("except") and ":" in stripped:
                # Check next non-empty line
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line == "pass":
                        # This is a bare except:pass — should not exist
                        pytest.fail(
                            f"{module_path}:{j+1}: bare 'except:pass' found "
                            f"(except at line {i+1}: {stripped})"
                        )
                    elif next_line:
                        break  # Non-empty, non-pass line — OK


class TestBatch18Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_detect_imports(self):
        from forge.hardware.detect import detect_hardware, GPUInfo, CPUInfo

    def test_rocm_imports(self):
        from forge.hardware.rocm import configure_rocm_env, get_rocm_status

    def test_client_imports(self):
        from forge.llm.client import OllamaClient

    def test_web_search_imports(self):
        from forge.mcp.web_search import WebSearchMCP

    def test_manager_imports(self):
        from forge.mcp.manager import MCPManager

    def test_web_tool_imports(self):
        from forge.tools.web import WebTool
