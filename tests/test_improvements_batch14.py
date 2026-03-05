"""Tests for batch 14 improvements: sessions limits, planner constants, tasks bounds, tracker validation, config constants."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# === Sessions: Constants and Validation ===

class TestSessionsConstants:
    """Tests for session constants."""

    def test_max_session_size_mb(self):
        from forge.agents.sessions import MAX_SESSION_SIZE_MB
        assert MAX_SESSION_SIZE_MB > 0

    def test_max_messages_per_session(self):
        from forge.agents.sessions import MAX_MESSAGES_PER_SESSION
        assert MAX_MESSAGES_PER_SESSION > 0

    def test_max_sessions_on_disk(self):
        from forge.agents.sessions import MAX_SESSIONS_ON_DISK
        assert MAX_SESSIONS_ON_DISK > 0

    def test_time_constants(self):
        from forge.agents.sessions import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, SECONDS_PER_DAY
        assert SECONDS_PER_MINUTE == 60
        assert SECONDS_PER_HOUR == 3600
        assert SECONDS_PER_DAY == 86400

    def test_display_limits(self):
        from forge.agents.sessions import MAX_TITLE_LENGTH, MAX_SYSTEM_CONTENT_DISPLAY, MAX_SEARCH_RESULT_CONTENT
        assert MAX_TITLE_LENGTH > 0
        assert MAX_SYSTEM_CONTENT_DISPLAY > 0
        assert MAX_SEARCH_RESULT_CONTENT > 0

    def test_list_limits(self):
        from forge.agents.sessions import DEFAULT_LIST_LIMIT, MIN_LIST_LIMIT, MAX_LIST_LIMIT
        assert MIN_LIST_LIMIT >= 1
        assert DEFAULT_LIST_LIMIT >= MIN_LIST_LIMIT
        assert MAX_LIST_LIMIT >= DEFAULT_LIST_LIMIT

    def test_search_limits(self):
        from forge.agents.sessions import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT
        assert DEFAULT_SEARCH_LIMIT > 0
        assert MAX_SEARCH_LIMIT >= DEFAULT_SEARCH_LIMIT

    def test_valid_export_formats(self):
        from forge.agents.sessions import VALID_EXPORT_FORMATS
        assert "markdown" in VALID_EXPORT_FORMATS
        assert "json" in VALID_EXPORT_FORMATS
        assert "html" in VALID_EXPORT_FORMATS

    def test_truncation_divisor(self):
        from forge.agents.sessions import TRUNCATION_DIVISOR
        assert TRUNCATION_DIVISOR >= 2


class TestSessionManager:
    """Tests for SessionManager validation."""

    def test_save_and_load(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            msgs = [{"role": "user", "content": "hello"}]
            sid = mgr.save(msgs, agent_name="test")
            assert sid.startswith("session-")
            loaded = mgr.load(sid)
            assert loaded is not None
            assert loaded.message_count == 1

    def test_list_sessions_limit_clamped(self):
        from forge.agents.sessions import SessionManager, MIN_LIST_LIMIT, MAX_LIST_LIMIT
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            # Should not crash with extreme limits
            result = mgr.list_sessions(limit=0)  # clamped to MIN_LIST_LIMIT
            assert isinstance(result, list)
            result = mgr.list_sessions(limit=999999)  # clamped to MAX_LIST_LIMIT
            assert isinstance(result, list)

    def test_export_invalid_format(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            result = mgr.export("some-id", format="csv")
            assert "Invalid format" in result

    def test_export_valid_formats(self):
        from forge.agents.sessions import SessionManager, VALID_EXPORT_FORMATS
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            msgs = [{"role": "user", "content": "hello"}]
            sid = mgr.save(msgs, agent_name="test")
            for fmt in VALID_EXPORT_FORMATS:
                result = mgr.export(sid, format=fmt)
                assert "Invalid format" not in result

    def test_search_empty_query(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            assert mgr.search("") == []
            assert mgr.search("   ") == []

    def test_search_limit_clamped(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            result = mgr.search("test", limit=999999)
            assert isinstance(result, list)

    def test_generate_title_truncated(self):
        from forge.agents.sessions import SessionManager, MAX_TITLE_LENGTH
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            long_msg = "A" * 200
            msgs = [{"role": "user", "content": long_msg}]
            sid = mgr.save(msgs)
            loaded = mgr.load(sid)
            assert len(loaded.title) <= MAX_TITLE_LENGTH

    def test_delete_session(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save([{"role": "user", "content": "hello"}])
            assert mgr.delete(sid)
            assert mgr.load(sid) is None

    def test_session_summary_age(self):
        from forge.agents.sessions import Session, SECONDS_PER_HOUR
        s = Session(
            session_id="test",
            title="Test",
            agent_name="test",
            model="test",
            messages=[],
            created_at=time.time() - 7200,
            updated_at=time.time() - 100,
        )
        summary = s.summary()
        assert "m ago" in summary  # 100s < 3600 → minutes


class TestSessionCleanup:
    """Tests for session cleanup."""

    def test_cleanup_old_sessions(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            # Create a few sessions
            for i in range(3):
                mgr.save([{"role": "user", "content": f"msg {i}"}])
            removed = mgr.cleanup_old_sessions()
            assert removed == 0  # under the limit

    def test_get_stats(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save([{"role": "user", "content": "hello"}])
            stats = mgr.get_stats()
            assert stats["session_count"] == 1
            assert stats["total_size_mb"] >= 0


# === Planner: Constants and Validation ===

class TestPlannerConstants:
    """Tests for planner constants."""

    def test_max_project_files_display(self):
        from forge.agents.planner import MAX_PROJECT_FILES_DISPLAY
        assert MAX_PROJECT_FILES_DISPLAY > 0

    def test_max_error_snippet_length(self):
        from forge.agents.planner import MAX_ERROR_SNIPPET_LENGTH
        assert MAX_ERROR_SNIPPET_LENGTH > 0

    def test_llm_plan_temperature(self):
        from forge.agents.planner import LLM_PLAN_TEMPERATURE
        assert 0.0 <= LLM_PLAN_TEMPERATURE <= 2.0

    def test_llm_plan_timeout(self):
        from forge.agents.planner import LLM_PLAN_TIMEOUT
        assert LLM_PLAN_TIMEOUT > 0

    def test_code_extensions(self):
        from forge.agents.planner import CODE_EXTENSIONS
        assert ".py" in CODE_EXTENSIONS
        assert ".js" in CODE_EXTENSIONS

    def test_ignore_dirs(self):
        from forge.agents.planner import IGNORE_DIRS
        assert "__pycache__" in IGNORE_DIRS
        assert ".git" in IGNORE_DIRS


class TestPlannerValidation:
    """Tests for planner input validation."""

    def test_empty_task_returns_empty_plan(self):
        from forge.agents.planner import EditPlanner
        planner = EditPlanner(client=None)
        plan = planner.plan("")
        assert plan.task == ""
        assert "Empty" in plan.reasoning

    def test_whitespace_task_returns_empty_plan(self):
        from forge.agents.planner import EditPlanner
        planner = EditPlanner(client=None)
        plan = planner.plan("   ")
        assert "Empty" in plan.reasoning

    def test_valid_task_without_client(self):
        from forge.agents.planner import EditPlanner
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(client=None, working_dir=tmpdir)
            plan = planner.plan("Rename function foo to bar")
            assert plan.task == "Rename function foo to bar"

    def test_validate_path_traversal(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="test",
                files=[FileEdit(path="../../etc/passwd", description="bad")]
            )
            errors = planner.validate(plan)
            assert any("escapes" in e for e in errors)

    def test_validate_file_not_found(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="test",
                files=[FileEdit(path="nonexistent.py", description="missing")]
            )
            errors = planner.validate(plan)
            assert any("not found" in e for e in errors)


class TestPlannerExecution:
    """Tests for planner execution."""

    def test_execute_valid_plan(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file to edit
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo():\n    pass\n")

            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="rename foo",
                files=[FileEdit(
                    path="test.py",
                    description="rename foo to bar",
                    edits=[{"old_string": "def foo():", "new_string": "def bar():"}],
                )],
            )
            result = planner.execute(plan)
            assert result.success
            assert "test.py" in result.files_modified

    def test_execute_invalid_plan_rolls_back(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original content")

            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="bad edit",
                files=[FileEdit(
                    path="test.py",
                    description="bad",
                    edits=[{"old_string": "nonexistent string", "new_string": "new"}],
                )],
            )
            result = planner.execute(plan)
            assert not result.success


# === Tasks: Constants and Validation ===

class TestTasksConstants:
    """Tests for task constants."""

    def test_default_task_timeout(self):
        from forge.agents.tasks import DEFAULT_TASK_TIMEOUT
        assert DEFAULT_TASK_TIMEOUT > 0

    def test_timeout_bounds(self):
        from forge.agents.tasks import MIN_TASK_TIMEOUT, MAX_TASK_TIMEOUT
        assert MIN_TASK_TIMEOUT >= 1
        assert MAX_TASK_TIMEOUT > MIN_TASK_TIMEOUT

    def test_concurrent_bounds(self):
        from forge.agents.tasks import DEFAULT_MAX_CONCURRENT, MIN_MAX_CONCURRENT, MAX_MAX_CONCURRENT
        assert MIN_MAX_CONCURRENT >= 1
        assert DEFAULT_MAX_CONCURRENT >= MIN_MAX_CONCURRENT
        assert MAX_MAX_CONCURRENT >= DEFAULT_MAX_CONCURRENT


class TestTaskManagerValidation:
    """Tests for TaskManager input validation."""

    def test_max_concurrent_clamped_high(self):
        from forge.agents.tasks import TaskManager, MAX_MAX_CONCURRENT
        tm = TaskManager(max_concurrent=9999)
        assert tm.max_concurrent == MAX_MAX_CONCURRENT

    def test_max_concurrent_clamped_low(self):
        from forge.agents.tasks import TaskManager, MIN_MAX_CONCURRENT
        tm = TaskManager(max_concurrent=0)
        assert tm.max_concurrent == MIN_MAX_CONCURRENT

    def test_normal_max_concurrent(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager(max_concurrent=3)
        assert tm.max_concurrent == 3

    def test_empty_command_fails(self):
        from forge.agents.tasks import TaskManager, TaskStatus
        tm = TaskManager()
        tid = tm.submit("test", "")
        result = tm.get_status(tid)
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert "Empty" in result.error

    def test_whitespace_command_fails(self):
        from forge.agents.tasks import TaskManager, TaskStatus
        tm = TaskManager()
        tid = tm.submit("test", "   ")
        result = tm.get_status(tid)
        assert result.status == TaskStatus.FAILED


class TestTaskManagerOperations:
    """Tests for task manager operations."""

    def test_submit_and_check_status(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        tid = tm.submit("test", "echo hello", timeout=5)
        assert tid.startswith("task-")
        # Wait for completion
        import time
        time.sleep(1)
        result = tm.get_status(tid)
        assert result is not None

    def test_list_tasks(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        tid = tm.submit("test", "echo hello", timeout=5)
        tasks = tm.list_tasks()
        assert len(tasks) >= 1

    def test_cancel_nonexistent(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        assert not tm.cancel("nonexistent")

    def test_cleanup(self):
        from forge.agents.tasks import TaskManager
        tm = TaskManager()
        tid = tm.submit("test", "echo hello", timeout=5)
        import time
        time.sleep(1)
        count = tm.cleanup()
        assert isinstance(count, int)

    def test_submit_callable(self):
        from forge.agents.tasks import TaskManager, TaskStatus
        tm = TaskManager()
        tid = tm.submit_callable("test", lambda: "result")
        import time
        time.sleep(0.5)
        result = tm.get_status(tid)
        assert result is not None


# === Tracker: Constants and Validation ===

class TestTrackerConstants:
    """Tests for tracker constants."""

    def test_max_system_name_length(self):
        from forge.agents.tracker import MAX_SYSTEM_NAME_LENGTH
        assert MAX_SYSTEM_NAME_LENGTH > 0

    def test_max_description_length(self):
        from forge.agents.tracker import MAX_DESCRIPTION_LENGTH
        assert MAX_DESCRIPTION_LENGTH > 0

    def test_max_agents_per_system(self):
        from forge.agents.tracker import MAX_AGENTS_PER_SYSTEM
        assert MAX_AGENTS_PER_SYSTEM > 0

    def test_valid_system_types(self):
        from forge.agents.tracker import VALID_SYSTEM_TYPES
        assert "single" in VALID_SYSTEM_TYPES
        assert "multi" in VALID_SYSTEM_TYPES

    def test_name_column_width(self):
        from forge.agents.tracker import NAME_COLUMN_WIDTH
        assert NAME_COLUMN_WIDTH > 0


class TestTrackerValidation:
    """Tests for tracker input validation."""

    def test_create_empty_name(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system("", "single", ["agent1"])
            assert "empty" in result.lower()

    def test_create_long_name(self):
        from forge.agents.tracker import AgentTracker, MAX_SYSTEM_NAME_LENGTH
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system("A" * (MAX_SYSTEM_NAME_LENGTH + 1), "single", ["agent1"])
            assert "too long" in result.lower()

    def test_create_invalid_system_type(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system("test", "invalid", ["agent1"])
            assert "Invalid system_type" in result

    def test_create_empty_agents(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system("test", "single", [])
            assert "At least one agent" in result

    def test_create_too_many_agents(self):
        from forge.agents.tracker import AgentTracker, MAX_AGENTS_PER_SYSTEM
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            agents = [f"agent{i}" for i in range(MAX_AGENTS_PER_SYSTEM + 1)]
            result = tracker.create_system("test", "multi", agents)
            assert "Too many agents" in result

    def test_create_valid_system(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system("my-system", "single", ["coder"])
            assert "Created" in result

    def test_record_activity_negative_values_clamped(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            tracker.create_system("test", "single", ["agent1"])
            tracker.record_activity("test", messages=-5, tool_calls=-3)
            system = tracker.get_system("test")
            assert system.total_messages == 0
            assert system.total_tool_calls == 0

    def test_list_systems_empty(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.list_systems()
            assert "No agent systems" in result

    def test_delete_system(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            tracker.create_system("test", "single", ["agent1"])
            result = tracker.delete_system("test")
            assert "Deleted" in result

    def test_delete_nonexistent_system(self):
        from forge.agents.tracker import AgentTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.delete_system("nonexistent")
            assert "not found" in result


# === Config: Constants and Validation ===

class TestConfigConstants:
    """Tests for config constants."""

    def test_min_context_tokens(self):
        from forge.config import MIN_CONTEXT_TOKENS
        assert MIN_CONTEXT_TOKENS > 0

    def test_max_context_tokens(self):
        from forge.config import MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS
        assert MAX_CONTEXT_TOKENS > MIN_CONTEXT_TOKENS

    def test_port_bounds(self):
        from forge.config import MIN_PORT, MAX_PORT
        assert MIN_PORT >= 1
        assert MAX_PORT == 65535

    def test_valid_compression_strategies(self):
        from forge.config import VALID_COMPRESSION_STRATEGIES
        assert "sliding_summary" in VALID_COMPRESSION_STRATEGIES
        assert "truncate" in VALID_COMPRESSION_STRATEGIES

    def test_valid_log_levels(self):
        from forge.config import VALID_LOG_LEVELS
        assert "DEBUG" in VALID_LOG_LEVELS
        assert "INFO" in VALID_LOG_LEVELS
        assert "ERROR" in VALID_LOG_LEVELS


class TestConfigValidation:
    """Tests for config validation."""

    def test_valid_config(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig()
        errors = validate_config(config)
        assert errors == []

    def test_invalid_url(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig(ollama_base_url="ftp://localhost")
        errors = validate_config(config)
        assert any("ollama_base_url" in e for e in errors)

    def test_context_tokens_too_small(self):
        from forge.config import ForgeConfig, validate_config, MIN_CONTEXT_TOKENS
        config = ForgeConfig(max_context_tokens=10)
        errors = validate_config(config)
        assert any(str(MIN_CONTEXT_TOKENS) in e for e in errors)

    def test_context_tokens_too_large(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig(max_context_tokens=10_000_000)
        errors = validate_config(config)
        assert any("too large" in e for e in errors)

    def test_invalid_port(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig(web_port=0)
        errors = validate_config(config)
        assert any("web_port" in e for e in errors)

    def test_invalid_compression_strategy(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig(compression_strategy="unknown")
        errors = validate_config(config)
        assert any("compression_strategy" in e for e in errors)

    def test_invalid_log_level(self):
        from forge.config import ForgeConfig, validate_config
        config = ForgeConfig(log_level="VERBOSE")
        errors = validate_config(config)
        assert any("log_level" in e for e in errors)


# === Integration Tests ===

class TestBatch14Integration:
    """Integration tests verifying all batch 14 constants are importable."""

    def test_sessions_constants_importable(self):
        from forge.agents.sessions import (
            SECONDS_PER_MINUTE, SECONDS_PER_HOUR, SECONDS_PER_DAY,
            MAX_TITLE_LENGTH, MAX_SYSTEM_CONTENT_DISPLAY, MAX_SEARCH_RESULT_CONTENT,
            DEFAULT_LIST_LIMIT, MIN_LIST_LIMIT, MAX_LIST_LIMIT,
            DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
            VALID_EXPORT_FORMATS, TRUNCATION_DIVISOR,
        )

    def test_planner_constants_importable(self):
        from forge.agents.planner import (
            MAX_PROJECT_FILES_DISPLAY, MAX_ERROR_SNIPPET_LENGTH,
            LLM_PLAN_TEMPERATURE, LLM_PLAN_TIMEOUT,
            CODE_EXTENSIONS, IGNORE_DIRS,
        )

    def test_tasks_constants_importable(self):
        from forge.agents.tasks import (
            DEFAULT_TASK_TIMEOUT, MIN_TASK_TIMEOUT, MAX_TASK_TIMEOUT,
            DEFAULT_MAX_CONCURRENT, MIN_MAX_CONCURRENT, MAX_MAX_CONCURRENT,
        )

    def test_tracker_constants_importable(self):
        from forge.agents.tracker import (
            MAX_SYSTEM_NAME_LENGTH, MAX_DESCRIPTION_LENGTH,
            MAX_AGENTS_PER_SYSTEM, VALID_SYSTEM_TYPES, NAME_COLUMN_WIDTH,
        )

    def test_config_constants_importable(self):
        from forge.config import (
            MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS,
            MIN_PORT, MAX_PORT,
            VALID_COMPRESSION_STRATEGIES, VALID_LOG_LEVELS,
        )
