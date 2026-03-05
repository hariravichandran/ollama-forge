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

# Validation bounds
MIN_CONTEXT_TOKENS = 256
MAX_CONTEXT_TOKENS = 1_000_000
MIN_PORT = 1
MAX_PORT = 65535

# Known enum values
VALID_COMPRESSION_STRATEGIES = {"sliding_summary", "truncate", "progressive"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


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

    # Self-improvement agent (OPT-IN: disabled by default)
    self_improve_enabled: bool = False
    self_improve_maintainer: bool = False

    # Logging
    log_level: str = "INFO"

    # Paths
    agents_dir: str = "agents"
    state_dir: str = ".forge_state"

    def __post_init__(self) -> None:
        if not self.mcp_config_path:
            self.mcp_config_path = str(CONFIG_DIR / "mcp.yaml")


def validate_config(config: ForgeConfig) -> list[str]:
    """Validate a ForgeConfig, returning a list of errors (empty = valid).

    Checks types, value ranges, and known enum values.
    """
    errors: list[str] = []

    # URL validation
    if not config.ollama_base_url.startswith(("http://", "https://")):
        errors.append(f"ollama_base_url must start with http:// or https://, got: {config.ollama_base_url}")

    # Integer range checks
    if config.max_context_tokens < MIN_CONTEXT_TOKENS:
        errors.append(f"max_context_tokens must be >= {MIN_CONTEXT_TOKENS}, got: {config.max_context_tokens}")
    if config.max_context_tokens > MAX_CONTEXT_TOKENS:
        errors.append(f"max_context_tokens seems too large: {config.max_context_tokens}")

    if not (MIN_PORT <= config.web_port <= MAX_PORT):
        errors.append(f"web_port must be {MIN_PORT}-{MAX_PORT}, got: {config.web_port}")

    # Known enum values
    if config.compression_strategy not in VALID_COMPRESSION_STRATEGIES:
        errors.append(f"compression_strategy must be one of {VALID_COMPRESSION_STRATEGIES}, got: {config.compression_strategy}")

    if config.log_level.upper() not in VALID_LOG_LEVELS:
        errors.append(f"log_level must be one of {VALID_LOG_LEVELS}, got: {config.log_level}")

    return errors


def load_config() -> ForgeConfig:
    """Load configuration from YAML file, env vars, and defaults.

    Validates the resulting config and logs warnings for any issues.
    """
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
        "FORGE_SELF_IMPROVE": "self_improve_enabled",
        "FORGE_SELF_IMPROVE_MAINTAINER": "self_improve_maintainer",
    }
    for env_key, config_key in env_overrides.items():
        value = os.environ.get(env_key)
        if value is not None:
            field_type = type(getattr(config, config_key))
            if field_type is bool:
                setattr(config, config_key, value.lower() not in ("0", "false", "no"))
            elif field_type is int:
                try:
                    setattr(config, config_key, int(value))
                except ValueError:
                    pass  # Keep default if env var is not a valid int
            else:
                setattr(config, config_key, value)

    # Validate
    import logging
    validation_errors = validate_config(config)
    if validation_errors:
        logger = logging.getLogger("forge.config")
        for err in validation_errors:
            logger.warning("Config issue: %s", err)

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
