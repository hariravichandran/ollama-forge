"""Tests for batch 17 improvements: return type annotations on public functions.

Verifies that key public functions now have proper return type annotations,
improving IDE support and type checker compatibility.
"""

import inspect
from typing import get_type_hints

import pytest


class TestOpenAICompatAnnotations:
    """Tests for openai_compat return type annotations."""

    def test_create_app_has_return_type(self):
        from forge.api.openai_compat import create_app
        hints = get_type_hints(create_app)
        assert "return" in hints

    def test_run_api_server_has_return_type(self):
        from forge.api.openai_compat import run_api_server
        hints = get_type_hints(run_api_server)
        assert "return" in hints
        assert hints["return"] is type(None)


class TestBaseAgentAnnotations:
    """Tests for base agent return type annotations."""

    def test_stream_chat_has_return_type(self):
        from forge.agents.base import BaseAgent
        hints = get_type_hints(BaseAgent.stream_chat)
        assert "return" in hints


class TestPlannerAnnotations:
    """Tests for planner return type annotations."""

    def test_walk_compat_has_return_type(self):
        from forge.agents.planner import _walk_compat
        hints = get_type_hints(_walk_compat)
        assert "return" in hints


class TestSandboxAnnotations:
    """Tests for sandbox return type annotations."""

    def test_get_preexec_fn_has_return_type(self):
        from forge.tools.sandbox import Sandbox
        hints = get_type_hints(Sandbox._get_preexec_fn)
        assert "return" in hints


class TestConfigAnnotations:
    """Tests for config return type annotations."""

    def test_post_init_has_return_type(self):
        from forge.config import ForgeConfig
        hints = get_type_hints(ForgeConfig.__post_init__)
        assert "return" in hints
        assert hints["return"] is type(None)


class TestReturnTypeConsistency:
    """Verify return types match actual behavior."""

    def test_create_app_returns_object(self):
        from forge.api.openai_compat import create_app
        app = create_app()
        assert app is not None

    def test_stream_chat_is_generator(self):
        """stream_chat should be annotated as returning a Generator."""
        from forge.agents.base import BaseAgent
        import typing
        hints = get_type_hints(BaseAgent.stream_chat)
        return_type = hints.get("return")
        # Should be Generator or similar
        origin = getattr(return_type, "__origin__", None)
        assert origin is not None  # Should be a generic type

    def test_walk_compat_yields(self):
        """_walk_compat should be annotated as a Generator."""
        from forge.agents.planner import _walk_compat
        hints = get_type_hints(_walk_compat)
        return_type = hints.get("return")
        origin = getattr(return_type, "__origin__", None)
        assert origin is not None


class TestBatch17Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_openai_compat_imports(self):
        from forge.api.openai_compat import create_app, run_api_server

    def test_base_agent_imports(self):
        from forge.agents.base import BaseAgent

    def test_planner_imports(self):
        from forge.agents.planner import EditPlanner, _walk_compat

    def test_sandbox_imports(self):
        from forge.tools.sandbox import Sandbox

    def test_config_imports(self):
        from forge.config import ForgeConfig
