"""Tests for the agent framework."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.agents.base import BaseAgent, AgentConfig
from forge.agents.cascade import CascadeAgent, CascadeConfig, auto_cascade_config
from forge.agents.permissions import (
    PermissionManager, PermissionLevel, AutoApproveManager, DEFAULT_PERMISSIONS,
)
from forge.agents.qa import QAAgent, QAResult
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

    def stream_chat(self, messages, system="", timeout=300, model=None):
        yield "Mock "
        yield "streamed "
        yield "response"

    def generate(self, prompt, system="", json_mode=False, temperature=0.7, timeout=300):
        return {"response": self._response, "tokens": 10, "time_s": 0.5}

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
