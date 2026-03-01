"""Tests for the agent framework."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.agents.base import BaseAgent, AgentConfig
from forge.agents.tracker import AgentTracker
from forge.community.ideas import IdeaCollector


class MockOllamaClient:
    """Mock client for testing agents without Ollama."""

    def __init__(self):
        self.model = "test:7b"
        self.stats = type("Stats", (), {
            "total_calls": 0, "total_tokens": 0,
            "avg_time_s": 0, "errors": 0,
        })()

    def chat(self, messages, tools=None, temperature=0.7, timeout=300):
        return {"response": "Mock response", "tokens": 10, "time_s": 0.5}

    def stream_chat(self, messages, system="", timeout=300, model=None):
        yield "Mock "
        yield "streamed "
        yield "response"


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
