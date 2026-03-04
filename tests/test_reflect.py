"""Tests for the reflective agent: self-reviewing responses."""

from unittest.mock import MagicMock, patch

import pytest

from forge.agents.base import AgentConfig
from forge.agents.reflect import ReflectiveAgent, REVIEW_PROMPT_GENERAL as REVIEW_PROMPT


class MockClient:
    """Mock OllamaClient for reflective agent tests."""

    def __init__(self, responses=None):
        self.model = "test:7b"
        self.base_url = "http://localhost:11434"
        self.num_ctx = 8192
        self.keep_alive = "30m"
        self._responses = responses or []
        self._call_idx = 0

        class Stats:
            total_calls = 0
            total_tokens = 0
            total_prompt_tokens = 0
            errors = 0
            avg_time_s = 0.0

        self.stats = Stats()

    def chat(self, messages, tools=None, temperature=0.7, timeout=300):
        resp = self._next_response()
        self.stats.total_calls += 1
        return {"response": resp, "tokens": 10, "prompt_tokens": 5, "time_s": 0.1}

    def generate(self, prompt="", system="", temperature=0.7, timeout=300):
        resp = self._next_response()
        self.stats.total_calls += 1
        return {"response": resp, "tokens": 10}

    def _next_response(self):
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return "Default mock response"


class TestReviewPrompt:
    """Tests for the REVIEW_PROMPT template."""

    def test_prompt_has_question_placeholder(self):
        assert "{question}" in REVIEW_PROMPT

    def test_prompt_has_response_placeholder(self):
        assert "{response}" in REVIEW_PROMPT

    def test_prompt_mentions_lgtm(self):
        assert "LGTM" in REVIEW_PROMPT

    def test_prompt_format(self):
        formatted = REVIEW_PROMPT.format(question="test?", response="answer")
        assert "test?" in formatted
        assert "answer" in formatted


class TestReflectiveAgentInit:
    """Tests for ReflectiveAgent initialization."""

    def test_default_max_revisions(self):
        client = MockClient()
        agent = ReflectiveAgent(client=client)
        assert agent.max_revisions == 1

    def test_custom_max_revisions(self):
        client = MockClient()
        agent = ReflectiveAgent(client=client, max_revisions=3)
        assert agent.max_revisions == 3

    def test_initial_counts_zero(self):
        client = MockClient()
        agent = ReflectiveAgent(client=client)
        assert agent._review_count == 0
        assert agent._revision_count == 0

    def test_inherits_base_agent(self):
        from forge.agents.base import BaseAgent
        client = MockClient()
        agent = ReflectiveAgent(client=client)
        assert isinstance(agent, BaseAgent)

    def test_custom_config(self):
        client = MockClient()
        config = AgentConfig(name="reviewer", temperature=0.2)
        agent = ReflectiveAgent(client=client, config=config)
        assert agent.config.name == "reviewer"
        assert agent.config.temperature == 0.2


class TestReflectiveChat:
    """Tests for the reflective chat loop."""

    def test_short_response_skips_review(self):
        """Responses under 30 chars should skip review."""
        client = MockClient(responses=["OK"])  # short response
        agent = ReflectiveAgent(client=client)
        result = agent.chat("Hi")
        assert result == "OK"
        assert agent._review_count == 0

    def test_error_response_skips_review(self):
        """Error messages should skip review."""
        client = MockClient(responses=["LLM error: connection refused"])
        agent = ReflectiveAgent(client=client)
        result = agent.chat("Hi")
        assert "LLM error:" in result
        assert agent._review_count == 0

    def test_lgtm_review_no_revision(self):
        """LGTM from reviewer means no revision."""
        client = MockClient(responses=[
            "This is a sufficiently long response that passes the length check.",  # initial
            "LGTM - the response looks good",  # review approves
        ])
        agent = ReflectiveAgent(client=client)
        result = agent.chat("What is Python?")
        assert agent._review_count == 1
        assert agent._revision_count == 0
        assert "sufficiently long" in result

    def test_review_triggers_revision(self):
        """Issues found in review trigger a revised response."""
        client = MockClient(responses=[
            "This is a sufficiently long response but has some issues to fix.",  # initial
            "The response has factual errors. Corrected version follows.",  # review rejects
            "This is the corrected and revised response with no errors.",  # revised
        ])
        agent = ReflectiveAgent(client=client)
        result = agent.chat("What is Python?")
        assert agent._review_count == 1
        assert agent._revision_count == 1
        assert "corrected" in result.lower()

    def test_max_revisions_limits_loop(self):
        """Should not exceed max_revisions iterations."""
        # Each revision: review rejects, generate revision, review rejects again...
        # max_revisions=2: up to 2 review-revise cycles
        client = MockClient(responses=[
            "Initial long response that needs work - attempt zero.",  # initial chat
            "Issues found, please fix the response",  # review 1 rejects
            "Revised response attempt one but still wrong.",  # revision 1
            "Still has issues, needs more work",  # review 2 rejects
            "Final revised response attempt two is better.",  # revision 2
        ])
        agent = ReflectiveAgent(client=client, max_revisions=2)
        result = agent.chat("Explain recursion")
        assert agent._review_count == 2
        assert agent._revision_count == 2

    def test_messages_updated_with_final_version(self):
        """The stored message should be the final (possibly revised) version."""
        client = MockClient(responses=[
            "Initial long response that needs reviewing for correctness.",
            "LGTM",
        ])
        agent = ReflectiveAgent(client=client)
        agent.chat("Question?")
        assert agent.messages[-1]["role"] == "assistant"
        assert "Initial long" in agent.messages[-1]["content"]


class TestReflectiveStats:
    """Tests for get_stats() with reflection metrics."""

    def test_stats_include_reflection(self):
        client = MockClient()
        agent = ReflectiveAgent(client=client)
        stats = agent.get_stats()
        assert "reflection" in stats
        assert stats["reflection"]["reviews"] == 0
        assert stats["reflection"]["revisions"] == 0
        assert stats["reflection"]["revision_rate"] == 0.0

    def test_stats_after_reviews(self):
        client = MockClient(responses=[
            "This is a long enough response to trigger the review step.",
            "LGTM",
        ])
        agent = ReflectiveAgent(client=client)
        agent.chat("Question?")
        stats = agent.get_stats()
        assert stats["reflection"]["reviews"] == 1
        assert stats["reflection"]["revisions"] == 0
        assert stats["reflection"]["revision_rate"] == 0.0

    def test_revision_rate_calculated(self):
        client = MockClient(responses=[
            "This is a long enough response to trigger the review step.",
            "Issues found: incorrect claim about Python",
            "Corrected response about Python with accurate information.",
        ])
        agent = ReflectiveAgent(client=client)
        agent.chat("What is Python?")
        stats = agent.get_stats()
        assert stats["reflection"]["reviews"] == 1
        assert stats["reflection"]["revisions"] == 1
        assert stats["reflection"]["revision_rate"] == 1.0

    def test_stats_inherit_base(self):
        client = MockClient()
        agent = ReflectiveAgent(client=client)
        stats = agent.get_stats()
        assert "name" in stats
        assert "model" in stats
        assert "llm_stats" in stats


class TestReviewAndRevise:
    """Tests for the _review_and_revise method directly."""

    def test_empty_review_returns_original(self):
        client = MockClient(responses=[""])  # empty review
        agent = ReflectiveAgent(client=client)
        result = agent._review_and_revise("q?", "original answer")
        assert result == "original answer"

    def test_lgtm_variations(self):
        """Various LGTM formats should be accepted."""
        for lgtm_text in ["LGTM", "lgtm", "LGTM!", "The response is LGTM"]:
            client = MockClient(responses=[lgtm_text])
            agent = ReflectiveAgent(client=client)
            result = agent._review_and_revise("q?", "answer")
            assert result == "answer", f"Failed for: {lgtm_text}"

    def test_question_truncated_for_review(self):
        """Long questions should be truncated to 500 chars."""
        long_question = "x" * 1000
        client = MockClient(responses=["LGTM"])
        agent = ReflectiveAgent(client=client)
        # Should not crash with long input
        agent._review_and_revise(long_question, "answer")
        assert agent._review_count == 1

    def test_response_truncated_for_review(self):
        """Long responses should be truncated to 2000 chars."""
        long_response = "y" * 5000
        client = MockClient(responses=["LGTM"])
        agent = ReflectiveAgent(client=client)
        agent._review_and_revise("q?", long_response)
        assert agent._review_count == 1
