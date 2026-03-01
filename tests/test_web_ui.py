"""Tests for the Web UI module."""

from pathlib import Path

import pytest


class TestWebUIModule:
    """Tests for web UI module structure."""

    def test_launch_web_ui_importable(self):
        """launch_web_ui should be importable."""
        from forge.ui.web.app import launch_web_ui
        assert callable(launch_web_ui)

    def test_create_web_app_importable(self):
        """create_web_app should be importable."""
        from forge.ui.web.app import create_web_app
        assert callable(create_web_app)

    def test_create_web_app_returns_fastapi(self):
        """create_web_app should return a FastAPI instance."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        from forge.ui.web.app import create_web_app
        app = create_web_app()
        assert isinstance(app, FastAPI)

    def test_app_has_routes(self):
        """Web app should have all expected routes."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        from forge.ui.web.app import create_web_app
        app = create_web_app()

        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in route_paths
        assert "/api/status" in route_paths
        assert "/api/chat" in route_paths
        assert "/api/chat/stream" in route_paths
        assert "/api/model" in route_paths
        assert "/api/agent" in route_paths
        assert "/api/reset" in route_paths
        assert "/api/stats" in route_paths

    def test_app_metadata(self):
        """App should have correct metadata."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        from forge.ui.web.app import create_web_app
        app = create_web_app()
        assert app.title == "ollama-forge Web UI"


class TestWebUIAssets:
    """Tests for static assets and templates."""

    def test_templates_dir_exists(self):
        """Templates directory should exist."""
        from forge.ui.web.app import UI_DIR
        templates_dir = UI_DIR / "templates"
        assert templates_dir.exists()

    def test_index_template_exists(self):
        """index.html template should exist."""
        from forge.ui.web.app import UI_DIR
        index = UI_DIR / "templates" / "index.html"
        assert index.exists()

    def test_index_has_required_elements(self):
        """index.html should contain required UI elements."""
        from forge.ui.web.app import UI_DIR
        index = UI_DIR / "templates" / "index.html"
        content = index.read_text()

        assert "chat-input" in content
        assert "chat-messages" in content
        assert "model-select" in content
        assert "agent-select" in content
        assert "status-panel" in content
        assert "sendMessage" in content

    def test_static_dir_exists(self):
        """Static directory should exist."""
        from forge.ui.web.app import UI_DIR
        static_dir = UI_DIR / "static"
        assert static_dir.exists()

    def test_stylesheet_exists(self):
        """CSS stylesheet should exist."""
        from forge.ui.web.app import UI_DIR
        css = UI_DIR / "static" / "style.css"
        assert css.exists()

    def test_stylesheet_has_dark_theme(self):
        """Stylesheet should have dark theme variables."""
        from forge.ui.web.app import UI_DIR
        css = UI_DIR / "static" / "style.css"
        content = css.read_text()

        assert "--bg-primary" in content
        assert "--text-primary" in content
        assert "--accent" in content

    def test_stylesheet_responsive(self):
        """Stylesheet should have responsive styles."""
        from forge.ui.web.app import UI_DIR
        css = UI_DIR / "static" / "style.css"
        content = css.read_text()
        assert "@media" in content

    def test_htmx_included(self):
        """index.html should include htmx."""
        from forge.ui.web.app import UI_DIR
        index = UI_DIR / "templates" / "index.html"
        content = index.read_text()
        assert "htmx" in content
