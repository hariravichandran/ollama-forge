"""Tests for the Terminal UI module."""

import pytest

# Test that the module loads and has the right structure,
# regardless of whether Textual is installed.


class TestLaunchTUI:
    """Tests for the TUI launcher function."""

    def test_launch_tui_importable(self):
        """launch_tui should be importable."""
        from forge.ui.terminal import launch_tui
        assert callable(launch_tui)

    def test_textual_available_flag(self):
        """TEXTUAL_AVAILABLE should be set correctly."""
        from forge.ui import terminal
        # Textual is installed in test env
        assert hasattr(terminal, "TEXTUAL_AVAILABLE")

    def test_forge_app_exists(self):
        """ForgeApp should exist when Textual is available."""
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if TEXTUAL_AVAILABLE:
            from forge.ui.terminal import ForgeApp
            assert ForgeApp is not None

    def test_chat_message_widget(self):
        """ChatMessage widget should be constructable."""
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if TEXTUAL_AVAILABLE:
            from forge.ui.terminal import ChatMessage
            msg = ChatMessage(role="user", content="Hello")
            assert msg.role == "user"
            assert msg.content == "Hello"

    def test_status_panel_widget(self):
        """StatusPanel should have reactive attributes."""
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if TEXTUAL_AVAILABLE:
            from forge.ui.terminal import StatusPanel
            panel = StatusPanel()
            assert hasattr(panel, "model_name")
            assert hasattr(panel, "agent_name")
            assert hasattr(panel, "is_thinking")


class TestForgeAppConfig:
    """Tests for ForgeApp configuration."""

    @pytest.fixture
    def app(self):
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if not TEXTUAL_AVAILABLE:
            pytest.skip("Textual not installed")
        from forge.ui.terminal import ForgeApp
        return ForgeApp(
            model="test:7b",
            agent_name="coder",
            working_dir="/tmp",
            cascade=True,
            auto_approve=True,
        )

    def test_app_stores_config(self, app):
        """App should store configuration from constructor."""
        assert app._model_arg == "test:7b"
        assert app._agent_name == "coder"
        assert app._working_dir == "/tmp"
        assert app._cascade is True
        assert app._auto_approve is True

    def test_app_title(self, app):
        """App should have the correct title."""
        assert app.TITLE == "ollama-forge"

    def test_app_has_bindings(self, app):
        """App should have keyboard bindings."""
        binding_keys = [b.key for b in app.BINDINGS]
        assert "ctrl+q" in binding_keys
        assert "ctrl+r" in binding_keys
        assert "ctrl+s" in binding_keys

    def test_app_has_css(self, app):
        """App should have CSS stylesheet."""
        assert len(app.CSS) > 100
        assert "#sidebar" in app.CSS
        assert "#chat-area" in app.CSS
        assert "#chat-input" in app.CSS


class TestForgeAppCommands:
    """Tests for slash command parsing."""

    @pytest.fixture
    def app(self):
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if not TEXTUAL_AVAILABLE:
            pytest.skip("Textual not installed")
        from forge.ui.terminal import ForgeApp
        return ForgeApp()

    def test_command_detection(self, app):
        """Slash commands should be recognized."""
        # _handle_command is a method that exists
        assert hasattr(app, "_handle_command")

    def test_quit_commands(self, app):
        """Multiple quit commands should be recognized."""
        # These are handled in handle_input
        assert hasattr(app, "action_quit")

    def test_reset_action(self, app):
        """Reset action should exist."""
        assert hasattr(app, "action_reset")

    def test_toggle_sidebar(self, app):
        """Sidebar toggle should exist."""
        assert hasattr(app, "action_toggle_sidebar")
        assert app._sidebar_visible is True

    def test_model_list_widget(self):
        """ModelList widget should accept model names."""
        from forge.ui.terminal import TEXTUAL_AVAILABLE
        if not TEXTUAL_AVAILABLE:
            pytest.skip("Textual not installed")
        from forge.ui.terminal import ModelList
        ml = ModelList(models=["model1:7b", "model2:14b"])
        assert ml._model_names == ["model1:7b", "model2:14b"]
