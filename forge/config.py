"""Configuration management for ollama-forge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from forge.utils.env import load_env

# Default config directory
CONFIG_DIR = Path(os.environ.get("FORGE_CONFIG_DIR", "~/.config/ollama-forge")).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class ForgeConfig:
    """Global configuration for ollama-forge."""

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = ""  # auto-detected from hardware if empty

    # Context compression
    max_context_tokens: int = 8192
    compression_strategy: str = "sliding_summary"

    # MCP settings
    web_search_enabled: bool = True
    mcp_config_path: str = ""

    # UI settings
    web_port: int = 8080

    # Logging
    log_level: str = "INFO"

    # Paths
    agents_dir: str = "agents"
    state_dir: str = ".forge_state"

    def __post_init__(self):
        if not self.mcp_config_path:
            self.mcp_config_path = str(CONFIG_DIR / "mcp.yaml")


def load_config() -> ForgeConfig:
    """Load configuration from YAML file, env vars, and defaults."""
    load_env()

    config = ForgeConfig()

    # Load from YAML if exists
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Override with env vars
    env_overrides = {
        "OLLAMA_BASE_URL": "ollama_base_url",
        "FORGE_DEFAULT_MODEL": "default_model",
        "FORGE_WEB_SEARCH": "web_search_enabled",
        "FORGE_WEB_PORT": "web_port",
        "FORGE_LOG_LEVEL": "log_level",
    }
    for env_key, config_key in env_overrides.items():
        value = os.environ.get(env_key)
        if value is not None:
            field_type = type(getattr(config, config_key))
            if field_type is bool:
                setattr(config, config_key, value.lower() not in ("0", "false", "no"))
            elif field_type is int:
                setattr(config, config_key, int(value))
            else:
                setattr(config, config_key, value)

    return config


def save_config(config: ForgeConfig) -> None:
    """Save configuration to YAML file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith("_")
    }
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
