"""Tests for the agent permission system."""

import pytest

from forge.agents.permissions import (
    PermissionManager,
    PermissionLevel,
    ActionPermission,
    AutoApproveManager,
    DEFAULT_PERMISSIONS,
)


class TestPermissionLevels:
    """Tests for permission level definitions."""

    def test_default_permissions_exist(self):
        """All core actions should have default permissions."""
        assert "read_file" in DEFAULT_PERMISSIONS
        assert "write_file" in DEFAULT_PERMISSIONS
        assert "run_command" in DEFAULT_PERMISSIONS
        assert "git_commit" in DEFAULT_PERMISSIONS
        assert "web_search" in DEFAULT_PERMISSIONS

    def test_read_operations_auto_approved(self):
        """Read operations should be auto-approved."""
        assert DEFAULT_PERMISSIONS["read_file"].level == PermissionLevel.AUTO_APPROVE
        assert DEFAULT_PERMISSIONS["list_files"].level == PermissionLevel.AUTO_APPROVE
        assert DEFAULT_PERMISSIONS["search_files"].level == PermissionLevel.AUTO_APPROVE
        assert DEFAULT_PERMISSIONS["git_status"].level == PermissionLevel.AUTO_APPROVE

    def test_write_operations_confirm_once(self):
        """Write operations should require one-time confirmation."""
        assert DEFAULT_PERMISSIONS["write_file"].level == PermissionLevel.CONFIRM_ONCE
        assert DEFAULT_PERMISSIONS["edit_file"].level == PermissionLevel.CONFIRM_ONCE

    def test_dangerous_operations_always_confirm(self):
        """Dangerous operations should always require confirmation."""
        assert DEFAULT_PERMISSIONS["run_command"].level == PermissionLevel.ALWAYS_CONFIRM
        assert DEFAULT_PERMISSIONS["git_commit"].level == PermissionLevel.ALWAYS_CONFIRM

    def test_web_operations_auto_approved(self):
        """Web operations should be auto-approved."""
        assert DEFAULT_PERMISSIONS["web_search"].level == PermissionLevel.AUTO_APPROVE
        assert DEFAULT_PERMISSIONS["web_fetch"].level == PermissionLevel.AUTO_APPROVE


class TestPermissionManager:
    """Tests for the PermissionManager class."""

    def test_auto_approve_read(self):
        """Read operations should pass without prompting."""
        mgr = PermissionManager()
        assert mgr.check("read_file") is True
        assert mgr.check("list_files") is True
        assert mgr.check("web_search") is True

    def test_auto_approve_all_flag(self):
        """auto_approve_all should bypass all checks."""
        mgr = PermissionManager(auto_approve_all=True)
        assert mgr.check("run_command") is True
        assert mgr.check("git_commit") is True
        assert mgr.check("unknown_action") is True

    def test_confirm_once_approved(self):
        """Confirm-once actions should auto-approve after first approval."""
        # Use a prompt_fn that always approves
        mgr = PermissionManager(prompt_fn=lambda msg: True)
        # First call should prompt (and approve)
        assert mgr.check("write_file") is True
        # Second call should auto-approve (no prompt)
        assert mgr.check("write_file") is True
        assert "write_file" in mgr._session_approvals

    def test_confirm_once_denied(self):
        """Denied confirm-once actions should NOT be cached."""
        mgr = PermissionManager(prompt_fn=lambda msg: False)
        assert mgr.check("write_file") is False
        assert "write_file" not in mgr._session_approvals

    def test_always_confirm_prompts_every_time(self):
        """Always-confirm actions should prompt on every call."""
        call_count = 0
        def counting_prompt(msg):
            nonlocal call_count
            call_count += 1
            return True

        mgr = PermissionManager(prompt_fn=counting_prompt)
        mgr.check("run_command")
        mgr.check("run_command")
        mgr.check("run_command")
        assert call_count == 3

    def test_unknown_action_prompts(self):
        """Unknown actions should require confirmation."""
        mgr = PermissionManager(prompt_fn=lambda msg: False)
        assert mgr.check("unknown_dangerous_thing") is False

    def test_approve_for_session(self):
        """Pre-approved actions should not prompt."""
        call_count = 0
        def counting_prompt(msg):
            nonlocal call_count
            call_count += 1
            return True

        mgr = PermissionManager(prompt_fn=counting_prompt)
        mgr.approve_for_session("write_file")
        mgr.check("write_file")
        assert call_count == 0  # No prompt needed

    def test_set_level(self):
        """Should be able to change permission level for an action."""
        mgr = PermissionManager()
        mgr.set_level("read_file", PermissionLevel.ALWAYS_CONFIRM)
        assert mgr.permissions["read_file"].level == PermissionLevel.ALWAYS_CONFIRM

    def test_set_level_new_action(self):
        """Setting level for unknown action should create it."""
        mgr = PermissionManager()
        mgr.set_level("custom_action", PermissionLevel.AUTO_APPROVE)
        assert "custom_action" in mgr.permissions
        assert mgr.permissions["custom_action"].category == "custom"

    def test_reset_session(self):
        """reset_session should clear all cached approvals."""
        mgr = PermissionManager(prompt_fn=lambda msg: True)
        mgr.check("write_file")
        assert "write_file" in mgr._session_approvals
        mgr.reset_session()
        assert len(mgr._session_approvals) == 0

    def test_context_passed_to_prompt(self):
        """Context should be included in prompt message."""
        received_msg = []
        def capture_prompt(msg):
            received_msg.append(msg)
            return True

        mgr = PermissionManager(prompt_fn=capture_prompt)
        mgr.check("run_command", context={"command": "ls -la"})
        assert "ls -la" in received_msg[0]


class TestAutoApproveManager:
    """Tests for the AutoApproveManager."""

    def test_approves_everything(self):
        """Should approve all actions without prompting."""
        mgr = AutoApproveManager()
        assert mgr.check("run_command") is True
        assert mgr.check("git_commit") is True
        assert mgr.check("write_file") is True
        assert mgr.check("random_action") is True

    def test_is_permission_manager(self):
        """Should be a subclass of PermissionManager."""
        mgr = AutoApproveManager()
        assert isinstance(mgr, PermissionManager)


class TestActionPermission:
    """Tests for ActionPermission dataclass."""

    def test_fields(self):
        perm = ActionPermission(
            action="test",
            description="Test action",
            level=PermissionLevel.AUTO_APPROVE,
            category="test",
        )
        assert perm.action == "test"
        assert perm.description == "Test action"
        assert perm.level == PermissionLevel.AUTO_APPROVE
        assert perm.category == "test"
