"""Extended tests for the agent framework: BaseAgent, AgentConfig, CoderAgent, ResearcherAgent.

Covers initialization, chat loop, tool dispatch, streaming, permissions,
YAML loading, caching, error handling, and specialized agent factories.
"""

import json
import tempfile
from pathlib import Path

import pytest

from forge.agents.base import BaseAgent, AgentConfig, load_agent_from_yaml
from forge.agents.coder import create_coder_agent, CODER_SYSTEM_PROMPT
from forge.agents.researcher import create_researcher_agent, RESEARCHER_SYSTEM_PROMPT
from forge.agents.permissions import PermissionManager, AutoApproveManager


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockOllamaClient:
    """Mock client that returns configurable sequences of responses."""

    def __init__(self, responses=None):
        self.responses = responses or [{"response": "mock response"}]
        self._call_idx = 0
        self.model = "test-model"
        self.stats = type(
            "Stats", (), {
                "total_calls": 0, "total_tokens": 0,
                "avg_time_s": 0.0, "errors": 0,
            },
        )()
        self._chat_calls = []
        self._stream_events = None

    def chat(self, messages=None, tools=None, temperature=None, timeout=300):
        self._chat_calls.append({
            "messages": messages, "tools": tools, "temperature": temperature,
        })
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return {"response": "default"}

    def generate(self, prompt="", system="", timeout=30, temperature=0.7,
                 json_mode=False):
        return {"response": "generated text"}

    def stream_chat(self, messages=None, tools=None, system="",
                    images=None, timeout=300, model=None):
        if self._stream_events is not None:
            yield from self._stream_events
            return
        yield {"type": "text", "content": "streamed "}
        yield {"type": "text", "content": "response"}
        yield {"type": "done"}

    def switch_model(self, model_name):
        self.model = model_name
        return True


class MockContextCompressor:
    """Compressor that simply returns messages unmodified."""

    def __init__(self, **kwargs):
        self.reset_count = 0

    def compress(self, messages):
        return messages

    def reset(self):
        self.reset_count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(client=None, config=None, working_dir=None,
                permissions=None, patch_compressor=True):
    """Create a BaseAgent with a mock compressor to avoid real LLM calls."""
    client = client or MockOllamaClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = working_dir or tmpdir
        agent = BaseAgent(
            client=client,
            config=config,
            working_dir=wd,
            permissions=permissions or AutoApproveManager(),
        )
        if patch_compressor:
            agent.compressor = MockContextCompressor()
        yield agent  # generator so TemporaryDirectory stays alive


@pytest.fixture
def tmp_dir():
    """Provide a clean temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def client():
    return MockOllamaClient()


@pytest.fixture
def agent(client, tmp_dir):
    """BaseAgent with default config, auto-approve, and mock compressor."""
    a = BaseAgent(
        client=client,
        working_dir=tmp_dir,
        permissions=AutoApproveManager(),
    )
    a.compressor = MockContextCompressor()
    return a


# ===================================================================
# AgentConfig tests
# ===================================================================

class TestAgentConfigExtended:
    """Extended tests for AgentConfig dataclass."""

    def test_default_values(self):
        cfg = AgentConfig()
        assert cfg.name == "assistant"
        assert cfg.model == ""
        assert cfg.temperature == 0.7
        assert cfg.max_context == 8192
        assert cfg.description == "General-purpose assistant"
        assert "You are a helpful AI assistant" in cfg.system_prompt

    def test_custom_values(self):
        cfg = AgentConfig(
            name="my-agent",
            model="llama3:70b",
            system_prompt="Custom prompt",
            tools=["git"],
            temperature=0.1,
            max_context=4096,
            description="My custom agent",
        )
        assert cfg.name == "my-agent"
        assert cfg.model == "llama3:70b"
        assert cfg.system_prompt == "Custom prompt"
        assert cfg.tools == ["git"]
        assert cfg.temperature == 0.1
        assert cfg.max_context == 4096
        assert cfg.description == "My custom agent"

    def test_tool_list_defaults(self):
        cfg = AgentConfig()
        assert "filesystem" in cfg.tools
        assert "shell" in cfg.tools
        assert "web" in cfg.tools
        assert len(cfg.tools) == 3

    def test_independent_tool_lists(self):
        """Two AgentConfig instances must have independent tool lists."""
        a = AgentConfig()
        b = AgentConfig()
        a.tools.append("extra")
        assert "extra" not in b.tools


# ===================================================================
# BaseAgent init tests
# ===================================================================

class TestBaseAgentInit:
    """Tests for BaseAgent.__init__."""

    def test_init_default_config(self, client, tmp_dir):
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        assert agent.config.name == "assistant"
        assert agent.messages == []
        assert agent.client is client

    def test_init_custom_config(self, client, tmp_dir):
        cfg = AgentConfig(name="custom", temperature=0.2, tools=["filesystem"])
        agent = BaseAgent(client=client, config=cfg, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        assert agent.config.name == "custom"
        assert agent.config.temperature == 0.2
        assert "filesystem" in agent._tools

    def test_init_loads_project_rules(self, client, tmp_dir):
        rules_path = Path(tmp_dir) / ".forge-rules"
        rules_path.write_text("Always use snake_case.")
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        assert "Project Rules" in agent._system_prompt
        assert "snake_case" in agent._system_prompt

    def test_init_no_rules_uses_raw_prompt(self, client, tmp_dir):
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        # No rules file => system prompt equals default prompt
        assert agent._system_prompt == agent.config.system_prompt

    def test_init_unrecognised_tool_ignored(self, client, tmp_dir):
        cfg = AgentConfig(tools=["nonexistent_tool_xyz"])
        agent = BaseAgent(client=client, config=cfg, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        assert len(agent._tools) == 0

    def test_init_creates_permission_manager_by_default(self, client, tmp_dir):
        agent = BaseAgent(client=client, working_dir=tmp_dir)
        assert isinstance(agent.permissions, PermissionManager)


# ===================================================================
# get_tool_definitions and caching
# ===================================================================

class TestGetToolDefinitions:
    """Tests for get_tool_definitions and its caching behaviour."""

    def test_returns_list(self, agent):
        defs = agent.get_tool_definitions()
        assert isinstance(defs, list)
        # Default tools are filesystem, shell, web — each defines several funcs
        assert len(defs) > 0

    def test_caching_same_object(self, agent):
        first = agent.get_tool_definitions()
        second = agent.get_tool_definitions()
        assert first is second  # exact same list object

    def test_definitions_contain_function_names(self, agent):
        defs = agent.get_tool_definitions()
        names = [d["function"]["name"] for d in defs]
        # filesystem tool should contribute read_file at minimum
        assert "read_file" in names

    def test_no_tools_returns_empty(self, client, tmp_dir):
        cfg = AgentConfig(tools=[])
        a = BaseAgent(client=client, config=cfg, working_dir=tmp_dir,
                      permissions=AutoApproveManager())
        a.compressor = MockContextCompressor()
        assert a.get_tool_definitions() == []


# ===================================================================
# chat() tests
# ===================================================================

class TestChat:
    """Tests for the synchronous chat loop."""

    def test_simple_response(self, client, tmp_dir):
        client.responses = [{"response": "Hello!"}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent.chat("Hi")
        assert result == "Hello!"
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[1]["role"] == "assistant"

    def test_chat_with_tool_calls(self, client, tmp_dir):
        """When the LLM returns tool_calls, the agent executes them and loops."""
        tool_response = {
            "response": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "test.txt"},
                },
            }],
        }
        final_response = {"response": "File contents are: hello"}

        client.responses = [tool_response, final_response]

        # Create the file so read_file succeeds
        (Path(tmp_dir) / "test.txt").write_text("hello world")

        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent.chat("Read test.txt")
        assert "File contents" in result or "hello" in result.lower() or result == "File contents are: hello"
        # Two LLM calls: one returning tool_calls, one returning final text
        assert len(client._chat_calls) == 2

    def test_chat_error_response(self, client, tmp_dir):
        client.responses = [{"error": "model not found"}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent.chat("Hey")
        assert "LLM error" in result
        assert "model not found" in result

    def test_chat_max_tool_rounds(self, client, tmp_dir):
        """When tool calls keep happening beyond the limit, bail out."""
        # Every response has tool_calls — agent should stop after 10 rounds
        endless_tool = {
            "response": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "x.txt"},
                },
            }],
        }
        client.responses = [endless_tool] * 15

        (Path(tmp_dir) / "x.txt").write_text("data")

        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent.chat("loop forever")
        assert "Max tool rounds" in result
        # Should have made exactly 10 calls
        assert len(client._chat_calls) == 10

    def test_chat_appends_user_message(self, agent):
        agent.client.responses = [{"response": "ok"}]
        agent.chat("question")
        assert agent.messages[0] == {"role": "user", "content": "question"}

    def test_chat_empty_response(self, client, tmp_dir):
        """An empty string response is still stored."""
        client.responses = [{"response": ""}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        result = agent.chat("say nothing")
        assert result == ""
        assert agent.messages[-1]["content"] == ""

    def test_chat_passes_temperature(self, client, tmp_dir):
        cfg = AgentConfig(temperature=0.42)
        client.responses = [{"response": "ok"}]
        agent = BaseAgent(client=client, config=cfg, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        agent.chat("hi")
        assert client._chat_calls[0]["temperature"] == 0.42

    def test_chat_includes_system_prompt(self, client, tmp_dir):
        client.responses = [{"response": "ok"}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        agent.chat("hi")
        sent_messages = client._chat_calls[0]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert "helpful AI assistant" in sent_messages[0]["content"]


# ===================================================================
# stream_chat() tests
# ===================================================================

class TestStreamChat:
    """Tests for the streaming chat interface."""

    def test_yields_text_and_done(self, agent):
        events = list(agent.stream_chat("hi"))
        types = [e["type"] for e in events]
        assert "text" in types
        assert "done" in types

    def test_stream_concatenated_response_stored(self, agent):
        events = list(agent.stream_chat("hi"))
        # After streaming, the full response should be in messages
        assert agent.messages[-1]["role"] == "assistant"
        assert "streamed" in agent.messages[-1]["content"]

    def test_stream_tool_call_events(self, client, tmp_dir):
        """When stream yields a tool_call event, the agent executes it."""
        (Path(tmp_dir) / "f.txt").write_text("data")

        client._stream_events = [
            {"type": "tool_call", "tool_calls": [{
                "function": {"name": "read_file", "arguments": {"path": "f.txt"}},
            }]},
            {"type": "text", "content": "got it"},
            {"type": "done"},
        ]

        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        events = list(agent.stream_chat("read f.txt"))

        types = [e["type"] for e in events]
        assert "tool_result" in types
        assert "done" in types

    def test_stream_stores_tool_results_in_history(self, client, tmp_dir):
        (Path(tmp_dir) / "a.txt").write_text("content-a")
        client._stream_events = [
            {"type": "tool_call", "tool_calls": [{
                "function": {"name": "read_file", "arguments": {"path": "a.txt"}},
            }]},
            {"type": "text", "content": "done"},
            {"type": "done"},
        ]

        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        list(agent.stream_chat("read it"))

        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "content-a" in tool_msgs[0]["content"]

    def test_stream_error_event_breaks(self, client, tmp_dir):
        client._stream_events = [
            {"type": "text", "content": "partial"},
            {"type": "error", "error": "something broke"},
        ]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        events = list(agent.stream_chat("hi"))
        types = [e["type"] for e in events]
        assert "error" in types
        # Should stop after the error
        assert types[-1] == "error"

    def test_stream_user_message_stored(self, agent):
        list(agent.stream_chat("hello stream"))
        assert agent.messages[0] == {"role": "user", "content": "hello stream"}


# ===================================================================
# _execute_tool tests
# ===================================================================

class TestExecuteTool:
    """Tests for the tool dispatch method."""

    def test_permission_denied(self, client, tmp_dir):
        """When the permission manager denies an action, return denial message."""
        perms = PermissionManager(
            auto_approve_all=False,
            prompt_fn=lambda _: False,  # always deny
        )
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=perms)
        agent.compressor = MockContextCompressor()

        result = agent._execute_tool("run_command", {"command": "ls"})
        assert "denied" in result.lower()

    def test_unknown_tool_function(self, agent):
        result = agent._execute_tool("nonexistent_function_xyz", {})
        assert "Unknown tool function" in result

    def test_routes_to_correct_tool(self, tmp_dir, client):
        (Path(tmp_dir) / "hello.txt").write_text("world")
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent._execute_tool("read_file", {"path": "hello.txt"})
        assert "world" in result

    def test_auto_approve_allows_everything(self, client, tmp_dir):
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        (Path(tmp_dir) / "w.txt").write_text("ok")
        result = agent._execute_tool("read_file", {"path": "w.txt"})
        assert "ok" in result

    def test_permission_check_for_write(self, client, tmp_dir):
        """write_file requires CONFIRM_ONCE — denied when prompt_fn returns False."""
        perms = PermissionManager(
            auto_approve_all=False,
            prompt_fn=lambda _: False,
        )
        agent = BaseAgent(client=client, working_dir=tmp_dir, permissions=perms)
        agent.compressor = MockContextCompressor()

        result = agent._execute_tool("write_file", {
            "path": "new.txt", "content": "hello",
        })
        assert "denied" in result.lower()

    def test_auto_approve_read_file(self, client, tmp_dir):
        """read_file is AUTO_APPROVE — should always succeed without prompting."""
        deny_prompt = PermissionManager(
            auto_approve_all=False,
            prompt_fn=lambda _: False,  # never called for auto-approve
        )
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=deny_prompt)
        agent.compressor = MockContextCompressor()

        (Path(tmp_dir) / "r.txt").write_text("readable")
        result = agent._execute_tool("read_file", {"path": "r.txt"})
        # read_file is auto-approve so it should work even with deny prompt
        assert "readable" in result


# ===================================================================
# reset() tests
# ===================================================================

class TestReset:
    """Tests for conversation reset."""

    def test_clears_messages(self, agent):
        agent.client.responses = [{"response": "hi"}]
        agent.chat("hello")
        assert len(agent.messages) > 0
        agent.reset()
        assert len(agent.messages) == 0

    def test_resets_compressor(self, agent):
        agent.chat("hello")
        agent.reset()
        assert agent.compressor.reset_count == 1

    def test_multiple_resets(self, agent):
        agent.reset()
        agent.reset()
        assert agent.compressor.reset_count == 2
        assert agent.messages == []


# ===================================================================
# get_stats() tests
# ===================================================================

class TestGetStats:
    """Tests for usage statistics reporting."""

    def test_returns_correct_structure(self, agent):
        stats = agent.get_stats()
        assert "name" in stats
        assert "model" in stats
        assert "messages" in stats
        assert "llm_stats" in stats

    def test_stats_reflect_config(self, agent):
        stats = agent.get_stats()
        assert stats["name"] == "assistant"
        assert stats["model"] == "test-model"

    def test_stats_message_count(self, agent):
        assert agent.get_stats()["messages"] == 0
        agent.client.responses = [{"response": "ok"}]
        agent.chat("hi")
        assert agent.get_stats()["messages"] == 2  # user + assistant

    def test_stats_llm_stats_keys(self, agent):
        llm = agent.get_stats()["llm_stats"]
        assert "total_calls" in llm
        assert "total_tokens" in llm
        assert "avg_time_s" in llm
        assert "errors" in llm

    def test_stats_avg_time_rounded(self, agent):
        agent.client.stats.avg_time_s = 1.23456
        stats = agent.get_stats()
        assert stats["llm_stats"]["avg_time_s"] == 1.23


# ===================================================================
# load_agent_from_yaml tests
# ===================================================================

class TestLoadAgentFromYaml:
    """Tests for YAML-based agent loading."""

    def test_load_valid_yaml(self, client, tmp_dir):
        yaml_path = Path(tmp_dir) / "agent.yaml"
        yaml_path.write_text(
            "name: yaml-agent\n"
            "system_prompt: You are a YAML agent.\n"
            "tools:\n  - filesystem\n"
            "temperature: 0.4\n"
            "max_context: 2048\n"
            "description: Test YAML agent\n"
        )
        agent = load_agent_from_yaml(str(yaml_path), client, working_dir=tmp_dir)
        assert isinstance(agent, BaseAgent)
        assert agent.config.name == "yaml-agent"
        assert agent.config.temperature == 0.4
        assert agent.config.max_context == 2048
        assert agent.config.tools == ["filesystem"]
        assert agent.config.description == "Test YAML agent"

    def test_load_yaml_missing_fields_uses_defaults(self, client, tmp_dir):
        yaml_path = Path(tmp_dir) / "minimal.yaml"
        yaml_path.write_text("name: minimal\n")
        agent = load_agent_from_yaml(str(yaml_path), client, working_dir=tmp_dir)
        assert agent.config.name == "minimal"
        assert agent.config.temperature == 0.7
        assert agent.config.max_context == 8192
        assert "filesystem" in agent.config.tools

    def test_load_nonexistent_yaml_raises(self, client, tmp_dir):
        with pytest.raises(FileNotFoundError, match="Agent config not found"):
            load_agent_from_yaml(
                str(Path(tmp_dir) / "nope.yaml"), client, working_dir=tmp_dir,
            )

    def test_load_yaml_with_model_switches(self, client, tmp_dir):
        yaml_path = Path(tmp_dir) / "model.yaml"
        yaml_path.write_text("name: modeled\nmodel: llama3:8b\n")
        agent = load_agent_from_yaml(str(yaml_path), client, working_dir=tmp_dir)
        assert client.model == "llama3:8b"
        assert agent.config.model == "llama3:8b"

    def test_load_yaml_empty_model_no_switch(self, client, tmp_dir):
        yaml_path = Path(tmp_dir) / "no_model.yaml"
        yaml_path.write_text("name: no-model\nmodel: ''\n")
        original_model = client.model
        load_agent_from_yaml(str(yaml_path), client, working_dir=tmp_dir)
        assert client.model == original_model

    def test_load_yaml_uses_stem_as_default_name(self, client, tmp_dir):
        yaml_path = Path(tmp_dir) / "my_cool_agent.yaml"
        yaml_path.write_text("temperature: 0.9\n")
        agent = load_agent_from_yaml(str(yaml_path), client, working_dir=tmp_dir)
        assert agent.config.name == "my_cool_agent"


# ===================================================================
# CoderAgent tests
# ===================================================================

class TestCoderAgent:
    """Tests for the coder agent factory."""

    def test_creates_base_agent(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert isinstance(agent, BaseAgent)

    def test_correct_tool_set(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert "filesystem" in agent.config.tools
        assert "shell" in agent.config.tools
        assert "git" in agent.config.tools
        assert "codebase" in agent.config.tools
        assert len(agent.config.tools) == 4

    def test_low_temperature(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert agent.config.temperature == 0.3

    def test_system_prompt_includes_coding_guidelines(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert "Read code before modifying" in agent._system_prompt or \
               "Read code before modifying" in CODER_SYSTEM_PROMPT
        # Verify the system prompt is based on CODER_SYSTEM_PROMPT
        assert CODER_SYSTEM_PROMPT in agent._system_prompt

    def test_name_is_coder(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert agent.config.name == "coder"

    def test_description(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert "code" in agent.config.description.lower() or \
               "Coding" in agent.config.description

    def test_has_correct_tool_instances(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        assert "filesystem" in agent._tools
        assert "shell" in agent._tools
        assert "git" in agent._tools

    def test_tool_definitions_include_read_file(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "read_file" in func_names

    def test_tool_definitions_include_run_command(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "run_command" in func_names

    def test_tool_definitions_include_git_status(self, client, tmp_dir):
        agent = create_coder_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "git_status" in func_names


# ===================================================================
# ResearcherAgent tests
# ===================================================================

class TestResearcherAgent:
    """Tests for the researcher agent factory."""

    def test_creates_base_agent(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert isinstance(agent, BaseAgent)

    def test_correct_tool_set(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert "web" in agent.config.tools
        assert "filesystem" in agent.config.tools
        assert len(agent.config.tools) == 2

    def test_medium_temperature(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert agent.config.temperature == 0.5

    def test_system_prompt_includes_research_guidelines(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert "Cite your sources" in agent._system_prompt or \
               "Cite your sources" in RESEARCHER_SYSTEM_PROMPT
        assert RESEARCHER_SYSTEM_PROMPT in agent._system_prompt

    def test_name_is_researcher(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert agent.config.name == "researcher"

    def test_description(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert "research" in agent.config.description.lower() or \
               "Research" in agent.config.description

    def test_has_correct_tool_instances(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        assert "web" in agent._tools
        assert "filesystem" in agent._tools
        assert "shell" not in agent._tools
        assert "git" not in agent._tools

    def test_tool_definitions_include_web_search(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "web_search" in func_names

    def test_tool_definitions_include_read_file(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "read_file" in func_names

    def test_no_shell_or_git_tools(self, client, tmp_dir):
        agent = create_researcher_agent(client, working_dir=tmp_dir)
        defs = agent.get_tool_definitions()
        func_names = [d["function"]["name"] for d in defs]
        assert "run_command" not in func_names
        assert "git_commit" not in func_names


# ===================================================================
# Integration-style tests combining multiple features
# ===================================================================

class TestAgentIntegration:
    """Integration tests exercising multiple agent features together."""

    def test_chat_then_stats_then_reset(self, client, tmp_dir):
        client.responses = [{"response": "answer"}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        agent.chat("question")
        stats = agent.get_stats()
        assert stats["messages"] == 2

        agent.reset()
        stats = agent.get_stats()
        assert stats["messages"] == 0

    def test_multiple_chat_rounds(self, client, tmp_dir):
        client.responses = [
            {"response": "first"},
            {"response": "second"},
            {"response": "third"},
        ]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        assert agent.chat("q1") == "first"
        assert agent.chat("q2") == "second"
        assert agent.chat("q3") == "third"
        assert len(agent.messages) == 6  # 3 user + 3 assistant

    def test_tool_call_then_final_response(self, client, tmp_dir):
        """Full round-trip: LLM asks for a tool, gets result, gives final answer."""
        (Path(tmp_dir) / "data.txt").write_text("secret=42")

        client.responses = [
            {
                "response": "",
                "tool_calls": [{
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "data.txt"},
                    },
                }],
            },
            {"response": "The secret is 42."},
        ]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        result = agent.chat("What is the secret?")
        assert result == "The secret is 42."
        # user + assistant messages only (tool messages go into the internal loop)
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[-1]["role"] == "assistant"

    def test_coder_agent_reads_file(self, client, tmp_dir):
        """Coder agent can read files via tool dispatch."""
        (Path(tmp_dir) / "main.py").write_text("print('hello')")
        agent = create_coder_agent(client, working_dir=tmp_dir)
        agent.permissions = AutoApproveManager()
        agent.compressor = MockContextCompressor()

        result = agent._execute_tool("read_file", {"path": "main.py"})
        assert "print" in result

    def test_system_prompt_with_rules_in_chat(self, client, tmp_dir):
        """Project rules should appear in messages sent to the LLM."""
        rules_path = Path(tmp_dir) / ".forge-rules"
        rules_path.write_text("Always respond in JSON format.")

        client.responses = [{"response": '{"ok": true}'}]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()
        agent.chat("test")

        system_msg = client._chat_calls[0]["messages"][0]
        assert system_msg["role"] == "system"
        assert "JSON format" in system_msg["content"]

    def test_tool_cache_survives_reset(self, agent):
        """Tool definition cache should be independent of message reset."""
        defs1 = agent.get_tool_definitions()
        agent.reset()
        defs2 = agent.get_tool_definitions()
        assert defs1 is defs2  # same cached object

    def test_stream_then_chat(self, client, tmp_dir):
        """Streaming and regular chat can be used interchangeably."""
        client._stream_events = [
            {"type": "text", "content": "stream reply"},
            {"type": "done"},
        ]
        agent = BaseAgent(client=client, working_dir=tmp_dir,
                          permissions=AutoApproveManager())
        agent.compressor = MockContextCompressor()

        list(agent.stream_chat("first"))
        assert agent.messages[-1]["role"] == "assistant"

        # Now regular chat
        client.responses = [{"response": "regular reply"}]
        result = agent.chat("second")
        assert result == "regular reply"
        assert len(agent.messages) == 4  # stream: user+assistant, chat: user+assistant
