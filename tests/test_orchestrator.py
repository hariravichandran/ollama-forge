"""Tests for the agent orchestrator."""

import tempfile
from pathlib import Path

import pytest
import yaml

from forge.agents.orchestrator import AgentOrchestrator
from forge.llm.client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:19999")


@pytest.fixture
def orchestrator(client):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield AgentOrchestrator(client=client, working_dir=tmpdir)


class TestBuiltinAgents:
    """Tests for built-in agent registration."""

    def test_has_assistant(self, orchestrator):
        assert "assistant" in orchestrator.agents

    def test_has_coder(self, orchestrator):
        assert "coder" in orchestrator.agents

    def test_has_researcher(self, orchestrator):
        assert "researcher" in orchestrator.agents

    def test_default_active_is_assistant(self, orchestrator):
        assert orchestrator.active_agent == "assistant"

    def test_assistant_has_all_tools(self, orchestrator):
        agent = orchestrator.agents["assistant"]
        assert "filesystem" in agent.config.tools
        assert "shell" in agent.config.tools
        assert "git" in agent.config.tools
        assert "web" in agent.config.tools

    def test_coder_has_coding_tools(self, orchestrator):
        agent = orchestrator.agents["coder"]
        assert "filesystem" in agent.config.tools
        assert "shell" in agent.config.tools
        assert "git" in agent.config.tools

    def test_researcher_has_research_tools(self, orchestrator):
        agent = orchestrator.agents["researcher"]
        assert "web" in agent.config.tools
        assert "filesystem" in agent.config.tools


class TestSwitchAgent:
    """Tests for agent switching."""

    def test_switch_to_valid_agent(self, orchestrator):
        result = orchestrator.switch_agent("coder")
        assert "Switched to agent: coder" in result
        assert orchestrator.active_agent == "coder"

    def test_switch_to_researcher(self, orchestrator):
        result = orchestrator.switch_agent("researcher")
        assert orchestrator.active_agent == "researcher"

    def test_switch_to_invalid_agent(self, orchestrator):
        result = orchestrator.switch_agent("nonexistent")
        assert "Unknown agent" in result
        assert orchestrator.active_agent == "assistant"  # unchanged

    def test_switch_back_to_assistant(self, orchestrator):
        orchestrator.switch_agent("coder")
        orchestrator.switch_agent("assistant")
        assert orchestrator.active_agent == "assistant"


class TestCreateAgent:
    """Tests for dynamic agent creation."""

    def test_create_agent(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            result = orch.create_agent(
                name="tester",
                description="Testing agent",
                system_prompt="You are a tester.",
                save=False,
            )
            assert "created" in result.lower()
            assert "tester" in orch.agents

    def test_create_agent_saves_yaml(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.create_agent(
                name="custom",
                description="Custom agent",
                system_prompt="You are custom.",
                save=True,
            )
            yaml_path = Path(tmpdir) / "agents" / "custom.yaml"
            assert yaml_path.exists()
            data = yaml.safe_load(yaml_path.read_text())
            assert data["name"] == "custom"
            assert data["description"] == "Custom agent"

    def test_create_duplicate_name(self, orchestrator):
        result = orchestrator.create_agent(
            name="assistant",
            description="Duplicate",
            system_prompt="test",
        )
        # Built-in names are rejected by validation (either as built-in or duplicate)
        assert "cannot" in result.lower() or "already exists" in result.lower()

    def test_create_agent_custom_config(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.create_agent(
                name="precise",
                description="Precise agent",
                system_prompt="Be precise.",
                tools=["filesystem"],
                temperature=0.1,
                save=False,
            )
            agent = orch.agents["precise"]
            assert agent.config.temperature == 0.1
            assert agent.config.tools == ["filesystem"]


class TestDeleteAgent:
    """Tests for agent deletion."""

    def test_delete_custom_agent(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.create_agent(
                name="temp",
                description="Temp",
                system_prompt="temp",
                save=False,
            )
            assert "temp" in orch.agents
            result = orch.delete_agent("temp")
            assert "deleted" in result.lower()
            assert "temp" not in orch.agents

    def test_delete_builtin_blocked(self, orchestrator):
        result = orchestrator.delete_agent("assistant")
        assert "Cannot delete built-in" in result

    def test_delete_coder_blocked(self, orchestrator):
        result = orchestrator.delete_agent("coder")
        assert "Cannot delete built-in" in result

    def test_delete_researcher_blocked(self, orchestrator):
        result = orchestrator.delete_agent("researcher")
        assert "Cannot delete built-in" in result

    def test_delete_nonexistent(self, orchestrator):
        result = orchestrator.delete_agent("ghost")
        assert "not found" in result.lower()

    def test_delete_active_resets_to_assistant(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.create_agent(
                name="active_one", description="x", system_prompt="x", save=False,
            )
            orch.switch_agent("active_one")
            assert orch.active_agent == "active_one"
            orch.delete_agent("active_one")
            assert orch.active_agent == "assistant"

    def test_delete_removes_yaml(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.create_agent(
                name="saved_agent", description="x", system_prompt="x", save=True,
            )
            yaml_path = Path(tmpdir) / "agents" / "saved_agent.yaml"
            assert yaml_path.exists()
            orch.delete_agent("saved_agent")
            assert not yaml_path.exists()


class TestChatRouting:
    """Tests for message routing via chat()."""

    def test_agent_command_routes(self, orchestrator):
        result = orchestrator.chat("/agent coder")
        assert "Switched" in result
        assert orchestrator.active_agent == "coder"

    def test_agents_list_command(self, orchestrator):
        result = orchestrator.chat("/agents")
        assert "assistant" in result
        assert "coder" in result
        assert "researcher" in result

    def test_no_active_agent(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            orch.active_agent = "nonexistent"
            result = orch.chat("hello")
            assert "No active agent" in result


class TestListAgents:
    """Tests for the agent listing format."""

    def test_list_shows_active_marker(self, orchestrator):
        result = orchestrator._list_agents()
        assert "*" in result  # active agent marker
        assert "assistant" in result

    def test_list_shows_all_agents(self, orchestrator):
        result = orchestrator._list_agents()
        assert "assistant" in result
        assert "coder" in result
        assert "researcher" in result


class TestLoadUserAgents:
    """Tests for loading agents from YAML files."""

    def test_load_from_yaml(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            yaml_data = {
                "name": "yaml_agent",
                "description": "From YAML",
                "system_prompt": "You are a YAML agent.",
                "tools": ["filesystem"],
                "temperature": 0.5,
            }
            (agents_dir / "yaml_agent.yaml").write_text(yaml.dump(yaml_data))
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            assert "yaml_agent" in orch.agents

    def test_no_agents_dir(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No agents/ directory — should work fine
            orch = AgentOrchestrator(client=client, working_dir=tmpdir)
            assert len(orch.agents) >= 3  # builtins only


class TestGetAllStats:
    """Tests for aggregate stats."""

    def test_stats_has_all_agents(self, orchestrator):
        stats = orchestrator.get_all_stats()
        assert "assistant" in stats
        assert "coder" in stats
        assert "researcher" in stats

    def test_stats_are_dicts(self, orchestrator):
        stats = orchestrator.get_all_stats()
        for name, agent_stats in stats.items():
            assert isinstance(agent_stats, dict)
