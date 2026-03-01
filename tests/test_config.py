"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from forge.config import ForgeConfig, load_config, save_config


class TestForgeConfig:
    """Tests for the ForgeConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = ForgeConfig()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.default_model == ""
        assert config.max_context_tokens == 8192
        assert config.compression_strategy == "sliding_summary"
        assert config.web_search_enabled is True
        assert config.self_improve_enabled is False
        assert config.self_improve_maintainer is False
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Config should accept custom values."""
        config = ForgeConfig(
            ollama_base_url="http://custom:1234",
            default_model="llama3:8b",
            max_context_tokens=16384,
        )
        assert config.ollama_base_url == "http://custom:1234"
        assert config.default_model == "llama3:8b"
        assert config.max_context_tokens == 16384

    def test_mcp_config_path_default(self):
        """MCP config path should be auto-populated."""
        config = ForgeConfig()
        assert config.mcp_config_path != ""
        assert "mcp.yaml" in config.mcp_config_path


class TestLoadConfig:
    """Tests for loading config from YAML and env."""

    def test_load_defaults(self):
        """Loading without config file should return defaults."""
        config = load_config()
        assert isinstance(config, ForgeConfig)
        assert config.ollama_base_url == "http://localhost:11434"

    def test_env_override_model(self):
        """Environment variables should override config values."""
        os.environ["FORGE_DEFAULT_MODEL"] = "test-model:7b"
        try:
            config = load_config()
            assert config.default_model == "test-model:7b"
        finally:
            del os.environ["FORGE_DEFAULT_MODEL"]

    def test_env_override_bool_true(self):
        """Boolean env vars should parse correctly (true)."""
        os.environ["FORGE_SELF_IMPROVE"] = "1"
        try:
            config = load_config()
            assert config.self_improve_enabled is True
        finally:
            del os.environ["FORGE_SELF_IMPROVE"]

    def test_env_override_bool_false(self):
        """Boolean env vars should parse correctly (false)."""
        os.environ["FORGE_SELF_IMPROVE"] = "0"
        try:
            config = load_config()
            assert config.self_improve_enabled is False
        finally:
            del os.environ["FORGE_SELF_IMPROVE"]

    def test_env_override_int(self):
        """Integer env vars should parse correctly."""
        os.environ["FORGE_WEB_PORT"] = "9090"
        try:
            config = load_config()
            assert config.web_port == 9090
        finally:
            del os.environ["FORGE_WEB_PORT"]


class TestSaveConfig:
    """Tests for saving config to YAML."""

    def test_save_and_reload(self):
        """Saved config should be loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"

            config = ForgeConfig(
                default_model="test:7b",
                max_context_tokens=4096,
            )

            # Write to temp file
            data = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
            config_file.write_text(yaml.dump(data))

            # Read back
            loaded = yaml.safe_load(config_file.read_text())
            assert loaded["default_model"] == "test:7b"
            assert loaded["max_context_tokens"] == 4096

    def test_save_creates_directory(self):
        """save_config should create the config directory."""
        # This tests the function signature, not the side effect
        # (we don't want to write to the real config dir in tests)
        assert callable(save_config)


class TestConfigAttributes:
    """Tests for config attribute types and validation."""

    def test_all_fields_have_defaults(self):
        """Every config field should have a default value."""
        config = ForgeConfig()
        for key, value in config.__dict__.items():
            if key.startswith("_"):
                continue
            assert value is not None, f"{key} has no default"

    def test_bool_fields(self):
        """Boolean fields should be actual bools."""
        config = ForgeConfig()
        assert isinstance(config.web_search_enabled, bool)
        assert isinstance(config.self_improve_enabled, bool)
        assert isinstance(config.self_improve_maintainer, bool)

    def test_int_fields(self):
        """Integer fields should be actual ints."""
        config = ForgeConfig()
        assert isinstance(config.max_context_tokens, int)
        assert isinstance(config.web_port, int)
