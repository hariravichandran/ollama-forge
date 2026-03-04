"""Tests for the agent framework."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.agents.autofix import AutoFixer, Check, AutoFixResult
from forge.agents.base import BaseAgent, AgentConfig
from forge.agents.cascade import CascadeAgent, CascadeConfig, auto_cascade_config
from forge.agents.memory import ConversationMemory
from forge.agents.permissions import (
    PermissionManager, PermissionLevel, AutoApproveManager, DEFAULT_PERMISSIONS,
)
from forge.agents.qa import QAAgent, QAResult
from forge.agents.reflect import ReflectiveAgent
from forge.agents.rules import load_project_rules, create_rules_template
from forge.agents.tasks import TaskManager, TaskStatus, TaskResult
from forge.agents.tracker import AgentTracker
from forge.community.ideas import IdeaCollector


class MockOllamaClient:
    """Mock client for testing agents without Ollama."""

    def __init__(self, response="Mock response"):
        self.model = "test:7b"
        self._response = response
        self._switched_to = []
        self.stats = type("Stats", (), {
            "total_calls": 0, "total_tokens": 0,
            "avg_time_s": 0, "errors": 0,
        })()

    def chat(self, messages, tools=None, temperature=0.7, timeout=300):
        return {"response": self._response, "tokens": 10, "time_s": 0.5}

    def stream_chat(self, messages, system="", tools=None, images=None, timeout=300, model=None):
        yield {"type": "text", "content": "Mock "}
        yield {"type": "text", "content": "streamed "}
        yield {"type": "text", "content": "response"}
        yield {"type": "done", "tokens": 10, "time_s": 0.5, "tokens_per_sec": 20.0}

    def generate(self, prompt, system="", json_mode=False, temperature=0.7, timeout=300):
        return {"response": self._response, "tokens": 10, "time_s": 0.5}

    def list_models(self):
        return [{"name": self.model}, {"name": "test:14b"}]

    def switch_model(self, model_name):
        self._switched_to.append(model_name)
        self.model = model_name
        return True


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_default_config(self):
        config = AgentConfig()
        assert config.name == "assistant"
        assert config.temperature == 0.7
        assert "filesystem" in config.tools

    def test_custom_config(self):
        config = AgentConfig(
            name="test-agent",
            model="custom:7b",
            tools=["shell"],
            temperature=0.3,
        )
        assert config.name == "test-agent"
        assert config.model == "custom:7b"
        assert config.tools == ["shell"]


class TestBaseAgent:
    """Tests for the base agent."""

    def test_agent_init(self):
        client = MockOllamaClient()
        agent = BaseAgent(client=client)
        assert agent.config.name == "assistant"
        assert len(agent.messages) == 0

    def test_agent_chat(self):
        client = MockOllamaClient()
        agent = BaseAgent(client=client)
        response = agent.chat("Hello")
        assert response == "Mock response"
        assert len(agent.messages) == 2  # user + assistant

    def test_agent_reset(self):
        client = MockOllamaClient()
        agent = BaseAgent(client=client)
        agent.chat("Hello")
        assert len(agent.messages) > 0
        agent.reset()
        assert len(agent.messages) == 0

    def test_agent_stats(self):
        client = MockOllamaClient()
        agent = BaseAgent(client=client)
        stats = agent.get_stats()
        assert stats["name"] == "assistant"
        assert "llm_stats" in stats


class TestAgentTracker:
    """Tests for agent system tracking."""

    def test_create_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.create_system(
                name="test-system",
                system_type="single",
                agents=["coder"],
                description="Test system",
            )
            assert "Created" in result
            assert "test-system" in tracker.systems

    def test_delete_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            tracker.create_system("test", "single", ["coder"])
            result = tracker.delete_system("test")
            assert "Deleted" in result
            assert "test" not in tracker.systems

    def test_record_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            tracker.create_system("test", "single", ["coder"])
            tracker.record_activity("test", messages=5, tool_calls=2)
            system = tracker.get_system("test")
            assert system.total_messages == 5
            assert system.total_tool_calls == 2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker1 = AgentTracker(state_dir=tmpdir)
            tracker1.create_system("test", "multi", ["coder", "researcher"])

            tracker2 = AgentTracker(state_dir=tmpdir)
            assert "test" in tracker2.systems
            assert tracker2.systems["test"].system_type == "multi"

    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=tmpdir)
            result = tracker.list_systems()
            assert "No agent systems" in result


class TestIdeaCollector:
    """Tests for community ideas collection."""

    def test_submit_idea(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.submit("Add PDF support", "Support reading PDF files")
            assert "submitted" in result.lower()
            assert len(collector.list_ideas()) == 1

    def test_duplicate_votes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Same idea", "Description")
            result = collector.submit("Same idea", "Description")
            assert "vote" in result.lower()
            ideas = collector.list_ideas()
            assert ideas[0].votes == 2

    def test_disabled_collector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir, enabled=False)
            result = collector.submit("Test", "Test")
            assert "disabled" in result.lower()

    def test_filter_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea 1", "Desc 1")
            collector.submit("Idea 2", "Desc 2")
            new_ideas = collector.list_ideas(status="new")
            assert len(new_ideas) == 2
            evaluated = collector.list_ideas(status="evaluated")
            assert len(evaluated) == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            c1 = IdeaCollector(ideas_dir=tmpdir)
            c1.submit("Persistent idea", "Should survive reload")

            c2 = IdeaCollector(ideas_dir=tmpdir)
            assert len(c2.list_ideas()) == 1


# ─── Permission System Tests ─────────────────────────────────────────────────


class TestPermissionManager:
    """Tests for the permission system."""

    def test_auto_approve_reads(self):
        pm = PermissionManager()
        assert pm.check("read_file") is True
        assert pm.check("list_files") is True
        assert pm.check("search_files") is True
        assert pm.check("web_search") is True

    def test_always_confirm_shell(self):
        # Use a prompt function that always denies
        pm = PermissionManager(prompt_fn=lambda msg: False)
        assert pm.check("run_command", {"command": "ls"}) is False

    def test_always_confirm_shell_approved(self):
        pm = PermissionManager(prompt_fn=lambda msg: True)
        assert pm.check("run_command", {"command": "ls"}) is True

    def test_confirm_once_session(self):
        call_count = 0

        def counting_prompt(msg):
            nonlocal call_count
            call_count += 1
            return True

        pm = PermissionManager(prompt_fn=counting_prompt)
        # First call should prompt
        assert pm.check("write_file") is True
        assert call_count == 1
        # Second call should be auto-approved (session cached)
        assert pm.check("write_file") is True
        assert call_count == 1  # no additional prompt

    def test_reset_session(self):
        call_count = 0

        def counting_prompt(msg):
            nonlocal call_count
            call_count += 1
            return True

        pm = PermissionManager(prompt_fn=counting_prompt)
        pm.check("edit_file")
        assert call_count == 1
        pm.reset_session()
        pm.check("edit_file")
        assert call_count == 2  # prompted again after reset

    def test_auto_approve_all(self):
        pm = PermissionManager(auto_approve_all=True)
        assert pm.check("run_command") is True
        assert pm.check("unknown_action") is True

    def test_unknown_action_requires_confirm(self):
        pm = PermissionManager(prompt_fn=lambda msg: False)
        assert pm.check("destroy_database") is False

    def test_set_level(self):
        pm = PermissionManager()
        pm.set_level("read_file", PermissionLevel.ALWAYS_CONFIRM)
        # Now read_file requires confirmation
        pm._prompt_fn = lambda msg: False
        assert pm.check("read_file") is False

    def test_approve_for_session(self):
        pm = PermissionManager(prompt_fn=lambda msg: False)
        # Pre-approve shell for this session
        pm.approve_for_session("run_command")
        # This should NOT prompt (ALWAYS_CONFIRM ignores session approvals)
        # Actually, approve_for_session adds to _session_approvals which
        # is only checked for CONFIRM_ONCE, not ALWAYS_CONFIRM
        assert pm.check("run_command") is False  # still denied (ALWAYS_CONFIRM)

    def test_default_permissions_exist(self):
        assert "read_file" in DEFAULT_PERMISSIONS
        assert "run_command" in DEFAULT_PERMISSIONS
        assert "git_commit" in DEFAULT_PERMISSIONS


class TestAutoApproveManager:
    """Tests for the auto-approve permission manager."""

    def test_approves_everything(self):
        pm = AutoApproveManager()
        assert pm.check("run_command") is True
        assert pm.check("destroy_everything") is True
        assert pm.check("read_file") is True


# ─── Cascade Agent Tests ─────────────────────────────────────────────────────


class TestCascadeAgent:
    """Tests for the cascading model agent."""

    def test_normal_response_no_escalation(self):
        client = MockOllamaClient(response="This is a perfectly good and detailed response to the user's question.")
        config = CascadeConfig(primary_model="test:7b", escalation_model="test:14b")
        agent = CascadeAgent(client=client, cascade_config=config)

        response = agent.chat("Hello")
        assert response == "This is a perfectly good and detailed response to the user's question."
        assert not agent._is_escalated
        assert agent._consecutive_poor == 0

    def test_poor_response_detection(self):
        agent = CascadeAgent(client=MockOllamaClient())
        assert agent._is_poor_response("") is True
        assert agent._is_poor_response("ok") is True  # too short
        assert agent._is_poor_response("I don't know how to help with that.") is True
        assert agent._is_poor_response("I apologize, but I cannot do that for you.") is True
        assert agent._is_poor_response(
            "Here is a detailed explanation of how Python decorators work "
            "and how to use them effectively in your code."
        ) is False

    def test_escalation_after_threshold(self):
        # Create client that always returns short responses
        client = MockOllamaClient(response="ok")
        config = CascadeConfig(
            primary_model="test:7b",
            escalation_model="test:14b",
            escalation_threshold=2,
        )
        agent = CascadeAgent(client=client, cascade_config=config)

        # First poor response
        agent.chat("question 1")
        assert agent._consecutive_poor == 1
        assert not agent._is_escalated

        # Second poor response triggers escalation
        agent.chat("question 2")
        assert agent._is_escalated
        assert "test:14b" in client._switched_to

    def test_deescalation_after_good_response(self):
        client = MockOllamaClient(
            response="This is a detailed and thorough response to the question asked."
        )
        config = CascadeConfig(
            primary_model="test:7b",
            escalation_model="test:14b",
            auto_deescalate=True,
        )
        agent = CascadeAgent(client=client, cascade_config=config)
        # Manually set escalated state
        agent._is_escalated = True
        agent.client.model = "test:14b"

        agent.chat("A good question")
        # Should have de-escalated back to primary
        assert not agent._is_escalated
        assert "test:7b" in client._switched_to

    def test_stats_include_cascade(self):
        client = MockOllamaClient()
        config = CascadeConfig(primary_model="test:7b", escalation_model="test:14b")
        agent = CascadeAgent(client=client, cascade_config=config)
        stats = agent.get_stats()
        assert "cascade" in stats
        assert stats["cascade"]["primary_model"] == "test:7b"
        assert stats["cascade"]["escalation_model"] == "test:14b"

    def test_no_escalation_model(self):
        client = MockOllamaClient(response="ok")
        config = CascadeConfig(primary_model="test:3b", escalation_model="", escalation_threshold=1)
        agent = CascadeAgent(client=client, cascade_config=config)
        # Should handle gracefully with no escalation model
        response = agent.chat("test")
        assert response is not None


class TestAutoCascadeConfig:
    """Tests for automatic cascade configuration."""

    def test_auto_config_returns_config(self):
        config = auto_cascade_config(16.0)
        assert isinstance(config, CascadeConfig)

    def test_auto_config_small_gpu(self):
        config = auto_cascade_config(4.0)
        # With only 4GB, might only fit one model or none
        assert isinstance(config, CascadeConfig)


# ─── QA Agent Tests ──────────────────────────────────────────────────────────


class TestQAResult:
    """Tests for QA result dataclass."""

    def test_passed_result(self):
        result = QAResult(
            passed=True,
            existing_tests_passed=True,
            generated_tests_passed=True,
            summary="All passed",
            generated_test_code="def test_x(): pass",
            test_output="1 passed",
        )
        assert result.passed
        assert "passed" in repr(result).lower()

    def test_failed_result(self):
        result = QAResult(
            passed=False,
            existing_tests_passed=True,
            generated_tests_passed=False,
            summary="Generated tests failed",
            generated_test_code="",
            test_output="FAILED",
        )
        assert not result.passed


class TestQAAgent:
    """Tests for the QA agent."""

    def test_init(self):
        client = MockOllamaClient()
        qa = QAAgent(client=client, repo_dir="/tmp")
        assert qa.repo_dir == Path("/tmp")

    def test_review_code_returns_string(self):
        client = MockOllamaClient(response="LGTM - no issues found")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): return 'world'")

            qa = QAAgent(client=client, repo_dir=tmpdir)
            review = qa.review_code(files_changed=["test.py"])
            assert isinstance(review, str)
            assert len(review) > 0


# ─── Auto-Fix Tests ──────────────────────────────────────────────────────────


class TestAutoFixer:
    """Tests for the auto-fix loop."""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            assert fixer.max_attempts == 3
            assert len(fixer.checks) == 0

    def test_add_check(self):
        fixer = AutoFixer(working_dir="/tmp", auto_detect=False)
        fixer.add_check("test-check", "echo ok")
        assert len(fixer.checks) == 1
        assert fixer.checks[0].name == "test-check"

    def test_run_checks_on_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a .txt file to avoid DEFAULT_CHECKS for .py
            txt_file = Path(tmpdir) / "valid.txt"
            txt_file.write_text("hello world\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("always-pass", "true")
            results = fixer.run_checks([str(txt_file)])
            assert all(r.passed for r in results)

    def test_run_checks_with_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "test.txt"
            txt_file.write_text("content\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("always-fail", "false")
            results = fixer.run_checks([str(txt_file)])
            assert any(not r.passed for r in results)

    def test_check_and_fix_all_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "good.txt"
            txt_file.write_text("ok\n")

            fixer = AutoFixer(working_dir=tmpdir, auto_detect=False)
            fixer.add_check("check", "true")
            result = fixer.check_and_fix([str(txt_file)])
            assert result.all_passed
            assert result.fixes_attempted == 0

    def test_auto_fix_result(self):
        result = AutoFixResult(
            all_passed=True,
            checks_run=5,
            fixes_attempted=1,
        )
        assert result.all_passed
        assert result.checks_run == 5


# ─── Conversation Memory Tests ───────────────────────────────────────────────


class TestConversationMemory:
    """Tests for persistent conversation memory."""

    def test_store_and_retrieve_fact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("language", "Python")
            assert mem.get_fact("language") == "Python"
            assert mem.get_fact("nonexistent") is None

    def test_facts_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = ConversationMemory(memory_dir=tmpdir)
            mem1.store_fact("user_name", "Alice")

            mem2 = ConversationMemory(memory_dir=tmpdir)
            assert mem2.get_fact("user_name") == "Alice"

    def test_save_and_load_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            path = mem.save_conversation(messages, session_id="test123")
            assert path
            assert Path(path).exists()

            recent = mem.get_recent_context(max_messages=10)
            assert len(recent) == 2
            assert recent[0]["content"] == "Hello"

    def test_facts_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("pref_editor", "vim")
            mem.store_fact("pref_lang", "Python")
            context = mem.get_facts_context()
            assert "pref_editor" in context
            assert "vim" in context

    def test_clear_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key", "value")
            mem.save_conversation([{"role": "user", "content": "test"}])
            mem.clear()
            assert mem.get_fact("key") is None
            assert mem.get_recent_context() == []

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.save_summary("User was working on a Python project")
            assert "Python project" in mem.get_summary()


# ─── Project Rules Tests ─────────────────────────────────────────────────────


class TestProjectRules:
    """Tests for project rules file support."""

    def test_no_rules_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules = load_project_rules(tmpdir)
            assert rules == ""

    def test_load_forge_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / ".forge-rules"
            rules_file.write_text("Always use type hints")
            rules = load_project_rules(tmpdir)
            assert "type hints" in rules

    def test_load_claude_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / "CLAUDE.md"
            rules_file.write_text("Use pytest for testing")
            rules = load_project_rules(tmpdir)
            assert "pytest" in rules

    def test_create_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_rules_template(tmpdir)
            assert "Created" in result
            assert (Path(tmpdir) / ".forge-rules").exists()

    def test_create_template_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".forge-rules").write_text("existing")
            result = create_rules_template(tmpdir)
            assert "already exists" in result


# ─── Reflective Agent Tests ──────────────────────────────────────────────────


class TestReflectiveAgent:
    """Tests for the self-reflecting agent."""

    def test_init(self):
        client = MockOllamaClient()
        agent = ReflectiveAgent(client=client)
        assert agent.max_revisions == 1
        assert agent._review_count == 0

    def test_short_response_not_reviewed(self):
        client = MockOllamaClient(response="ok")
        agent = ReflectiveAgent(client=client)
        response = agent.chat("hi")
        assert response == "ok"
        assert agent._review_count == 0  # too short to review

    def test_stats_include_reflection(self):
        client = MockOllamaClient()
        agent = ReflectiveAgent(client=client)
        stats = agent.get_stats()
        assert "reflection" in stats
        assert stats["reflection"]["reviews"] == 0
        assert stats["reflection"]["revisions"] == 0


# ─── Background Task Tests ───────────────────────────────────────────────────


class TestTaskResult:
    """Tests for task result."""

    def test_done_states(self):
        r = TaskResult(task_id="t1", name="test", status=TaskStatus.COMPLETED)
        assert r.done

        r2 = TaskResult(task_id="t2", name="test", status=TaskStatus.RUNNING)
        assert not r2.done

    def test_elapsed(self):
        r = TaskResult(
            task_id="t1", name="test", status=TaskStatus.COMPLETED,
            started_at=100.0, completed_at=105.0,
        )
        assert r.elapsed_s == 5.0


class TestTaskManager:
    """Tests for background task execution."""

    def test_submit_and_complete(self):
        import time
        tm = TaskManager()
        task_id = tm.submit("echo", "echo hello")

        # Wait for completion
        for _ in range(50):
            status = tm.get_status(task_id)
            if status and status.done:
                break
            time.sleep(0.1)

        status = tm.get_status(task_id)
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert "hello" in status.output

    def test_submit_failing_command(self):
        import time
        tm = TaskManager()
        task_id = tm.submit("fail", "exit 1")

        for _ in range(50):
            status = tm.get_status(task_id)
            if status and status.done:
                break
            time.sleep(0.1)

        status = tm.get_status(task_id)
        assert status.status == TaskStatus.FAILED

    def test_list_tasks(self):
        import time
        tm = TaskManager()
        tm.submit("t1", "echo 1")
        tm.submit("t2", "echo 2")

        time.sleep(0.5)
        tasks = tm.list_tasks()
        assert len(tasks) == 2

    def test_cancel_task(self):
        import time
        tm = TaskManager()
        task_id = tm.submit("slow", "sleep 10")
        time.sleep(0.2)
        success = tm.cancel(task_id)
        assert success
        status = tm.get_status(task_id)
        assert status.status == TaskStatus.CANCELLED

    def test_cleanup(self):
        import time
        tm = TaskManager()
        tm.submit("quick", "echo done")
        time.sleep(0.5)
        removed = tm.cleanup()
        assert removed >= 1
        assert len(tm.list_tasks()) == 0

    def test_max_concurrent(self):
        tm = TaskManager(max_concurrent=1)
        tm.submit("t1", "sleep 5")
        import time
        time.sleep(0.1)
        task_id2 = tm.submit("t2", "echo 2")
        status = tm.get_status(task_id2)
        assert status.status == TaskStatus.FAILED
        assert "Too many" in status.error


# ─── Codebase Indexer Tests ─────────────────────────────────────────────────


class TestCodebaseIndexer:
    """Tests for the codebase indexing and search system."""

    def test_index_python_file(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            py_file = Path(tmpdir) / "example.py"
            py_file.write_text(
                'class UserAuth:\n'
                '    """Handles user authentication."""\n'
                '    def login(self, username: str, password: str) -> bool:\n'
                '        """Authenticate a user."""\n'
                '        return True\n'
                '\n'
                'def helper_function(x: int) -> str:\n'
                '    return str(x)\n'
            )

            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()

            assert stats["files"] >= 1
            assert stats["symbols"] >= 3  # class + method + function

    def test_search_by_symbol(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "auth.py"
            py_file.write_text(
                'class AuthManager:\n'
                '    def validate_token(self, token: str) -> bool:\n'
                '        return True\n'
            )

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            results = indexer.search("AuthManager")
            assert len(results) > 0
            assert any(r.symbol == "AuthManager" for r in results)

    def test_search_by_content(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "config.py"
            py_file.write_text('DATABASE_URL = "postgresql://localhost/mydb"\n')

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            results = indexer.search("DATABASE_URL")
            assert len(results) > 0

    def test_find_symbol(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "utils.py"
            py_file.write_text(
                'def format_date(dt):\n    return str(dt)\n'
                'def format_time(t):\n    return str(t)\n'
            )

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            syms = indexer.find_symbol("format_date")
            assert len(syms) == 1
            assert syms[0].kind == "function"

    def test_incremental_update(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build initial index
            f1 = Path(tmpdir) / "a.py"
            f1.write_text("def foo(): pass\n")

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert indexer.find_symbol("foo")

            # Add a new file
            f2 = Path(tmpdir) / "b.py"
            f2.write_text("def bar(): pass\n")

            stats = indexer.update_index()
            assert stats["added"] == 1
            assert indexer.find_symbol("bar")

    def test_persistence(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "mod.py"
            f1.write_text("class MyModel:\n    pass\n")

            # Build and save
            idx1 = CodebaseIndexer(tmpdir)
            idx1.build_index()
            assert idx1.find_symbol("MyModel")

            # Load from disk
            idx2 = CodebaseIndexer(tmpdir)
            assert idx2.find_symbol("MyModel")

    def test_project_overview(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "main.py"
            f1.write_text("def main(): pass\n")
            f2 = Path(tmpdir) / "utils.py"
            f2.write_text("def helper(): pass\n")

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            overview = indexer.get_project_overview()
            assert "python" in overview.lower()
            assert "2" in overview  # 2 files

    def test_js_symbol_extraction(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            js_file = Path(tmpdir) / "app.js"
            js_file.write_text(
                'function handleRequest(req, res) {\n'
                '  return res.json({ok: true});\n'
                '}\n'
                'class Router {\n'
                '  constructor() {}\n'
                '}\n'
            )

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            assert indexer.find_symbol("handleRequest")
            assert indexer.find_symbol("Router")

    def test_gitignore_respect(self):
        from forge.tools.codebase import CodebaseIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create gitignore
            gitignore = Path(tmpdir) / ".gitignore"
            gitignore.write_text("secret.py\n")

            # Create files
            visible = Path(tmpdir) / "visible.py"
            visible.write_text("def public(): pass\n")
            secret = Path(tmpdir) / "secret.py"
            secret.write_text("API_KEY = 'secret123'\n")

            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            # Should find visible but not secret
            assert indexer.find_symbol("public")
            results = indexer.search("API_KEY")
            assert len(results) == 0


# ─── Sandbox Tests ──────────────────────────────────────────────────────────


class TestSandbox:
    """Tests for sandboxed code execution."""

    def test_run_python_success(self):
        from forge.tools.sandbox import Sandbox

        sandbox = Sandbox(timeout=10)
        result = sandbox.run_python("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    def test_run_python_error(self):
        from forge.tools.sandbox import Sandbox

        sandbox = Sandbox(timeout=10)
        result = sandbox.run_python("raise ValueError('test error')")
        assert not result.success
        assert result.return_code != 0

    def test_run_python_timeout(self):
        from forge.tools.sandbox import Sandbox

        sandbox = Sandbox(timeout=2)
        result = sandbox.run_python("import time; time.sleep(10)")
        assert not result.success
        assert result.timed_out

    def test_run_python_with_files(self):
        from forge.tools.sandbox import Sandbox

        sandbox = Sandbox(timeout=10)
        result = sandbox.run_python(
            "data = open('input.txt').read(); print(data)",
            files={"input.txt": "hello from file"},
        )
        assert result.success
        assert "hello from file" in result.stdout

    def test_run_command(self):
        from forge.tools.sandbox import Sandbox

        sandbox = Sandbox(timeout=10)
        result = sandbox.run_command("echo 'test output'")
        assert result.success
        assert "test output" in result.stdout

    def test_execution_result_output(self):
        from forge.tools.sandbox import ExecutionResult

        result = ExecutionResult(stdout="out", stderr="err", return_code=0, duration_s=1.0)
        assert result.success
        assert "out" in result.output
        assert "err" in result.output

    def test_execution_result_failure(self):
        from forge.tools.sandbox import ExecutionResult

        result = ExecutionResult(stdout="", stderr="", return_code=1, duration_s=1.0)
        assert not result.success

    def test_sandbox_tool_interface(self):
        from forge.tools.sandbox import SandboxTool

        tool = SandboxTool()
        defs = tool.get_tool_definitions()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert "run_code" in names
        assert "run_shell" in names

    def test_sandbox_tool_execute(self):
        from forge.tools.sandbox import SandboxTool

        tool = SandboxTool()
        result = tool.execute("run_code", {"code": "print(2 + 2)"})
        assert "4" in result
        assert "OK" in result

    def test_sandbox_cleanup(self):
        """Verify temp directory is cleaned up after execution."""
        from forge.tools.sandbox import Sandbox
        import os

        sandbox = Sandbox(timeout=10)
        result = sandbox.run_python("import os; print(os.getcwd())")
        assert result.success
        # The temp dir should be cleaned up
        tmpdir = result.stdout.strip()
        assert not os.path.exists(tmpdir)


# ─── Fuzzy Edit Tests ──────────────────────────────────────────────────────


class TestFuzzyEdit:
    """Tests for fuzzy matching in file edits."""

    def test_exact_match(self):
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.py"
            f.write_text("def hello():\n    print('hello')\n")

            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._edit_file("test.py", "print('hello')", "print('world')")
            assert "Replaced 1 occurrence" in result
            assert "world" in f.read_text()

    def test_fuzzy_match_whitespace(self):
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.py"
            f.write_text("def hello():\n    print('hello')\n    return True\n")

            tool = FilesystemTool(working_dir=tmpdir)
            # Search with slightly different whitespace
            result = tool._edit_file(
                "test.py",
                "def hello():\n  print('hello')\n  return True",  # 2 spaces vs 4
                "def hello():\n    print('world')\n    return True",
            )
            assert "fuzzy match" in result.lower() or "Replaced" in result

    def test_fuzzy_find_method(self):
        from forge.tools.filesystem import FilesystemTool

        tool = FilesystemTool()
        content = "line 1\nline 2\nline 3\nline 4\n"
        # Search for something similar to line 2-3
        result = tool._fuzzy_find(content, "line 2\nline 3")
        assert result is not None
        actual, ratio = result
        assert ratio >= 0.75

    def test_fuzzy_no_match(self):
        from forge.tools.filesystem import FilesystemTool

        tool = FilesystemTool()
        content = "completely different text here\n"
        result = tool._fuzzy_find(content, "nothing like this at all exists")
        assert result is None

    def test_edit_file_not_found(self):
        from forge.tools.filesystem import FilesystemTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(working_dir=tmpdir)
            result = tool._edit_file("nonexistent.py", "old", "new")
            assert "not found" in result.lower()


# ─── Git Tool Tests ─────────────────────────────────────────────────────────


class TestGitTool:
    """Tests for the enhanced git tool."""

    def test_git_tool_definitions(self):
        from forge.tools.git import GitTool

        tool = GitTool()
        defs = tool.get_tool_definitions()
        names = [d["function"]["name"] for d in defs]
        assert "git_status" in names
        assert "git_undo" in names
        assert "git_create_branch" in names
        assert "git_stash" in names

    def test_agent_commit_tag(self):
        from forge.tools.git import AGENT_COMMIT_TAG
        assert "[forge]" in AGENT_COMMIT_TAG

    def test_execute_unknown_function(self):
        from forge.tools.git import GitTool

        tool = GitTool()
        result = tool.execute("unknown_function", {})
        assert "Unknown function" in result

    def test_generate_commit_message_no_client(self):
        from forge.tools.git import GitTool

        tool = GitTool(client=None)
        msg = tool._generate_commit_message(["file1.py", "file2.py"], "Fix bug")
        assert msg == "Fix bug"

    def test_generate_commit_message_with_client(self):
        from forge.tools.git import GitTool

        client = MockOllamaClient(response="Fix authentication bug in login flow")
        tool = GitTool(client=client)
        msg = tool._generate_commit_message(["auth.py"], "fix auth")
        assert len(msg) <= 72
        assert msg  # non-empty


# ─── LLM Client Enhancement Tests ──────────────────────────────────────────


class TestLLMClientEnhancements:
    """Tests for the enhanced LLM client features."""

    def test_keep_alive_default(self):
        from forge.llm.client import OllamaClient

        client = OllamaClient()
        assert client.keep_alive == "30m"

    def test_keep_alive_custom(self):
        from forge.llm.client import OllamaClient

        client = OllamaClient(keep_alive="1h")
        assert client.keep_alive == "1h"

    def test_image_injection_with_base64(self):
        from forge.llm.client import OllamaClient

        client = OllamaClient()
        messages = [{"role": "user", "content": "What's in this image?"}]
        result = client._inject_images(messages, ["dGVzdA=="])  # base64 "test"
        assert "images" in result[-1]
        assert result[-1]["images"] == ["dGVzdA=="]

    def test_image_injection_preserves_messages(self):
        from forge.llm.client import OllamaClient

        client = OllamaClient()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = client._inject_images(messages, ["img1"])
        assert result[0].get("role") == "system"
        assert "images" not in result[0]
        assert "images" in result[1]

    def test_image_injection_with_file(self):
        from forge.llm.client import OllamaClient

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            f.flush()

            client = OllamaClient()
            messages = [{"role": "user", "content": "Describe this"}]
            result = client._inject_images(messages, [f.name])
            assert "images" in result[-1]
            # Should be base64 encoded
            import base64
            decoded = base64.b64decode(result[-1]["images"][0])
            assert decoded == b"fake image data"

            import os
            os.unlink(f.name)

    def test_stream_chat_signature(self):
        """Verify stream_chat accepts new parameters."""
        from forge.llm.client import OllamaClient
        import inspect

        sig = inspect.signature(OllamaClient.stream_chat)
        assert "tools" in sig.parameters
        assert "images" in sig.parameters

    def test_chat_images_parameter(self):
        """Verify chat accepts images parameter."""
        from forge.llm.client import OllamaClient
        import inspect

        sig = inspect.signature(OllamaClient.chat)
        assert "images" in sig.parameters

    def test_llm_stats(self):
        from forge.llm.client import LLMStats

        stats = LLMStats(total_calls=10, total_tokens=1000, total_time_s=5.0)
        assert stats.avg_time_s == 0.5
        assert stats.avg_tokens_per_sec == 200.0


# ─── Tools Registration Tests ──────────────────────────────────────────────


class TestToolsRegistration:
    """Tests for tool registration and discovery."""

    def test_builtin_tools_include_sandbox(self):
        from forge.tools import BUILTIN_TOOLS
        assert "sandbox" in BUILTIN_TOOLS

    def test_sandbox_tool_class(self):
        from forge.tools import SandboxTool
        tool = SandboxTool()
        assert tool.name == "sandbox"
        assert tool.sandbox is not None


# ─── Edit Planner Tests ────────────────────────────────────────────────────


class TestEditPlanner:
    """Tests for multi-file edit planning."""

    def test_edit_plan_summary(self):
        from forge.agents.planner import EditPlan, FileEdit

        plan = EditPlan(
            task="Rename function",
            files=[
                FileEdit(path="a.py", description="Update definition", edits=[
                    {"old_string": "def old_name", "new_string": "def new_name"},
                ]),
                FileEdit(path="b.py", description="Update import", edits=[
                    {"old_string": "from a import old_name", "new_string": "from a import new_name"},
                ]),
            ],
        )
        summary = plan.summary()
        assert "Rename function" in summary
        assert "a.py" in summary
        assert "b.py" in summary
        assert plan.file_count == 2

    def test_execute_plan_success(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source files
            (Path(tmpdir) / "a.py").write_text("def old_name():\n    pass\n")
            (Path(tmpdir) / "b.py").write_text("from a import old_name\n")

            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Rename",
                files=[
                    FileEdit(path="a.py", description="def", edits=[
                        {"old_string": "def old_name", "new_string": "def new_name"},
                    ]),
                    FileEdit(path="b.py", description="import", edits=[
                        {"old_string": "old_name", "new_string": "new_name"},
                    ]),
                ],
                dependency_order=["a.py", "b.py"],
            )

            result = planner.execute(plan)
            assert result.success
            assert len(result.files_modified) == 2
            assert "new_name" in (Path(tmpdir) / "a.py").read_text()
            assert "new_name" in (Path(tmpdir) / "b.py").read_text()

    def test_execute_plan_rollback(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("original content\n")
            (Path(tmpdir) / "b.py").write_text("also original\n")

            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Bad rename",
                files=[
                    FileEdit(path="a.py", description="ok edit", edits=[
                        {"old_string": "original content", "new_string": "new content"},
                    ]),
                    FileEdit(path="b.py", description="will fail", edits=[
                        {"old_string": "nonexistent string", "new_string": "something"},
                    ]),
                ],
                dependency_order=["a.py", "b.py"],
            )

            result = planner.execute(plan)
            assert not result.success
            # Validation now catches the error before execution,
            # so no files are modified and no rollback is needed
            assert not result.rolled_back
            # Original content should be untouched (validation prevented changes)
            assert "original content" in (Path(tmpdir) / "a.py").read_text()

    def test_create_new_file(self):
        from forge.agents.planner import EditPlanner, EditPlan, FileEdit

        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Create file",
                files=[
                    FileEdit(
                        path="new_module.py",
                        description="Create new module",
                        create=True,
                        new_content="# New module\ndef hello(): pass\n",
                    ),
                ],
            )

            result = planner.execute(plan)
            assert result.success
            assert (Path(tmpdir) / "new_module.py").exists()

    def test_analyze_dependencies(self):
        from forge.agents.planner import EditPlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "models.py").write_text("class User: pass\n")
            (Path(tmpdir) / "views.py").write_text("from models import User\n")

            planner = EditPlanner(working_dir=tmpdir)
            files = planner._get_project_files()
            deps = planner._analyze_dependencies(files)
            # views.py depends on models.py
            assert "models.py" in deps.get("views.py", [])


# ─── Session Tests ──────────────────────────────────────────────────────────


class TestSessionManager:
    """Tests for session persistence."""

    def test_save_and_load(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]

            session_id = mgr.save(messages, agent_name="test")
            assert session_id.startswith("session-")

            session = mgr.load(session_id)
            assert session is not None
            assert session.message_count == 2
            assert session.agent_name == "test"

    def test_list_sessions(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))

            mgr.save([{"role": "user", "content": "First"}], title="Session 1")
            mgr.save([{"role": "user", "content": "Second"}], title="Session 2")

            sessions = mgr.list_sessions()
            assert len(sessions) == 2

    def test_delete_session(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save([{"role": "user", "content": "Delete me"}])

            assert mgr.delete(sid)
            assert mgr.load(sid) is None

    def test_export_markdown(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            messages = [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ]
            sid = mgr.save(messages, agent_name="assistant", title="Python Q&A")

            md = mgr.export(sid, format="markdown")
            assert "Python Q&A" in md
            assert "**User:**" in md
            assert "**Assistant:**" in md

    def test_export_json(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save([{"role": "user", "content": "test"}])

            exported = mgr.export(sid, format="json")
            data = json.loads(exported)
            assert data["session_id"] == sid

    def test_partial_id_match(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save([{"role": "user", "content": "partial"}])

            # Load with partial ID (first 12 chars)
            session = mgr.load(sid[:12])
            assert session is not None

    def test_auto_title_generation(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save([{"role": "user", "content": "How do I install Docker?"}])

            session = mgr.load(sid)
            assert "Docker" in session.title

    def test_update_existing_session(self):
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            messages1 = [{"role": "user", "content": "Hello"}]
            sid = mgr.save(messages1, title="Test")

            # Update with more messages
            messages2 = messages1 + [
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ]
            mgr.save(messages2, session_id=sid)

            session = mgr.load(sid)
            assert session.message_count == 3

    def test_session_summary(self):
        from forge.agents.sessions import Session
        import time

        session = Session(
            session_id="session-test1234",
            title="Test Session",
            agent_name="coder",
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "hi"}],
            created_at=time.time() - 3600,
            updated_at=time.time() - 60,
        )
        summary = session.summary()
        assert "session-" in summary
        assert "Test Session" in summary
        assert "coder" in summary


class TestBaseAgentStreaming:
    """Tests for the improved stream_chat with tool support."""

    def test_stream_chat_yields_events(self):
        """stream_chat should yield typed event dicts."""
        client = MockOllamaClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        events = list(agent.stream_chat("Hello"))
        text_events = [e for e in events if e.get("type") == "text"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(text_events) >= 1
        assert len(done_events) == 1

    def test_stream_chat_accumulates_response(self):
        """stream_chat should save the full response in message history."""
        client = MockOllamaClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        _ = list(agent.stream_chat("Hello"))
        assert len(agent.messages) == 2  # user + assistant
        assert agent.messages[1]["role"] == "assistant"

    def test_stream_chat_handles_errors(self):
        """stream_chat should handle error events gracefully."""
        class ErrorStreamClient(MockOllamaClient):
            def stream_chat(self, messages, system="", tools=None, images=None, timeout=300, model=None):
                yield {"type": "error", "error": "Test error"}
        client = ErrorStreamClient()
        agent = BaseAgent(client=client, config=AgentConfig(tools=[]))
        events = list(agent.stream_chat("Hello"))
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1


class TestCLISessionCommands:
    """Tests for CLI session management commands."""

    def test_session_manager_importable(self):
        """SessionManager should be importable for CLI use."""
        from forge.agents.sessions import SessionManager
        mgr = SessionManager()
        assert hasattr(mgr, "list_sessions")
        assert hasattr(mgr, "export")
        assert hasattr(mgr, "delete")

    def test_session_export_formats(self):
        """export() should support markdown and json formats."""
        from forge.agents.sessions import SessionManager
        mgr = SessionManager()
        # Export nonexistent session should handle gracefully
        result = mgr.export("nonexistent-session", format="markdown")
        assert "not found" in result.lower()

        result = mgr.export("nonexistent-session", format="json")
        assert "not found" in result.lower()
