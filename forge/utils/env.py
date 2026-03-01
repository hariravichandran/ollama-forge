"""Environment variable loading from .env files."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Uses setdefault so existing env vars are not overwritten.
    Returns a dict of all variables that were loaded.
    """
    if env_path is None:
        # Walk up from cwd looking for .env
        env_path = _find_env_file()
    if env_path is None or not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


def _find_env_file() -> Path | None:
    """Walk up from cwd looking for a .env file."""
    current = Path.cwd()
    for _ in range(10):  # max 10 levels up
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)
