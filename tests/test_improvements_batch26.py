"""Tests for batch 26 improvements: lazy imports and caching.

Verifies that yaml imports are lazy in config, orchestrator, and MCP manager,
that re is module-level in models.py, and that difflib is lazy in filesystem.
"""

import inspect

import pytest


class TestConfigLazyYaml:
    """Tests for lazy yaml import in config.py."""

    def test_no_top_level_yaml_import(self):
        """config.py should not import yaml at module level."""
        import forge.config as mod
        source = inspect.getsource(mod)
        # Count top-level 'import yaml' — should not be in the top imports
        lines = source.splitlines()
        top_level_yaml = False
        for line in lines:
            stripped = line.strip()
            if stripped == "import yaml":
                # Check if this is at function level (indented) or module level
                if not line.startswith(" ") and not line.startswith("\t"):
                    top_level_yaml = True
        assert not top_level_yaml, "yaml should not be imported at module level"

    def test_load_config_imports_yaml(self):
        """load_config() should import yaml locally."""
        from forge.config import load_config
        source = inspect.getsource(load_config)
        assert "import yaml" in source

    def test_save_config_imports_yaml(self):
        """save_config() should import yaml locally."""
        from forge.config import save_config
        source = inspect.getsource(save_config)
        assert "import yaml" in source

    def test_config_still_works(self):
        """Config module should still work correctly."""
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig()
        errors = validate_config(config)
        assert isinstance(errors, list)


class TestOrchestratorLazyYaml:
    """Tests for lazy yaml import in orchestrator.py."""

    def test_no_top_level_yaml_import(self):
        """orchestrator.py should not import yaml at module level."""
        import forge.agents.orchestrator as mod
        source = inspect.getsource(mod)
        lines = source.splitlines()
        top_level_yaml = False
        for line in lines:
            stripped = line.strip()
            if stripped == "import yaml":
                if not line.startswith(" ") and not line.startswith("\t"):
                    top_level_yaml = True
        assert not top_level_yaml, "yaml should not be imported at module level"

    def test_create_agent_imports_yaml(self):
        """create_agent() should import yaml locally when saving."""
        from forge.agents.orchestrator import AgentOrchestrator
        source = inspect.getsource(AgentOrchestrator.create_agent)
        assert "import yaml" in source


class TestMCPManagerLazyYaml:
    """Tests for lazy yaml import in MCP manager."""

    def test_no_top_level_yaml_import(self):
        """manager.py should not import yaml at module level."""
        import forge.mcp.manager as mod
        source = inspect.getsource(mod)
        lines = source.splitlines()
        top_level_yaml = False
        for line in lines:
            stripped = line.strip()
            if stripped == "import yaml":
                if not line.startswith(" ") and not line.startswith("\t"):
                    top_level_yaml = True
        assert not top_level_yaml, "yaml should not be imported at module level"

    def test_load_config_imports_yaml(self):
        """_load_config() should import yaml locally."""
        from forge.mcp.manager import MCPManager
        source = inspect.getsource(MCPManager._load_config)
        assert "import yaml" in source

    def test_save_config_imports_yaml(self):
        """_save_config() should import yaml locally."""
        from forge.mcp.manager import MCPManager
        source = inspect.getsource(MCPManager._save_config)
        assert "import yaml" in source


class TestModelsModuleLevelRe:
    """Tests for module-level re import in models.py."""

    def test_re_imported_at_module_level(self):
        """models.py should import re at module level."""
        import forge.llm.models as mod
        source = inspect.getsource(mod)
        # Check that 'import re' appears before any function definitions
        lines = source.splitlines()
        found_module_re = False
        for line in lines:
            if line.strip() == "import re" and not line.startswith(" "):
                found_module_re = True
                break
        assert found_module_re, "re should be imported at module level"

    def test_no_local_import_re(self):
        """Functions should not have local 'import re' statements."""
        from forge.llm.models import estimate_model_size, _detect_quantization, validate_model_name
        for fn in [estimate_model_size, _detect_quantization, validate_model_name]:
            source = inspect.getsource(fn)
            assert "import re" not in source, f"{fn.__name__} should not have local import re"

    def test_functions_still_work(self):
        """Model functions should still work correctly."""
        from forge.llm.models import estimate_model_size, validate_model_name
        size = estimate_model_size("llama3.1:8b")
        assert size > 0
        err = validate_model_name("test-model:7b")
        assert err == ""


class TestFilesystemLazyDifflib:
    """Tests for lazy difflib import in filesystem.py."""

    def test_no_top_level_difflib_import(self):
        """filesystem.py should not import difflib at module level."""
        import forge.tools.filesystem as mod
        source = inspect.getsource(mod)
        lines = source.splitlines()
        top_level_difflib = False
        for line in lines:
            stripped = line.strip()
            if stripped == "import difflib":
                if not line.startswith(" ") and not line.startswith("\t"):
                    top_level_difflib = True
        assert not top_level_difflib, "difflib should not be imported at module level"

    def test_fuzzy_find_imports_difflib(self):
        """_fuzzy_find() should import difflib locally."""
        from forge.tools.filesystem import FilesystemTool
        source = inspect.getsource(FilesystemTool._fuzzy_find)
        assert "import difflib" in source


class TestBatch26Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_config_imports(self):
        from forge.config import ForgeConfig, load_config, save_config

    def test_orchestrator_imports(self):
        from forge.agents.orchestrator import AgentOrchestrator

    def test_manager_imports(self):
        from forge.mcp.manager import MCPManager

    def test_models_imports(self):
        from forge.llm.models import estimate_model_size, validate_model_name

    def test_filesystem_imports(self):
        from forge.tools.filesystem import FilesystemTool
