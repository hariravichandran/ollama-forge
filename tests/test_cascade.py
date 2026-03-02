"""Tests for the cascade (auto-escalation) agent."""

from forge.agents.cascade import (
    CascadeAgent,
    CascadeConfig,
    auto_cascade_config,
    STUCK_PATTERNS,
    MIN_USEFUL_LENGTH,
    ESCALATION_THRESHOLD,
)
from forge.agents.base import AgentConfig
from forge.llm.client import OllamaClient


class TestCascadeConfig:
    """Tests for CascadeConfig."""

    def test_defaults(self):
        cfg = CascadeConfig()
        assert cfg.primary_model == ""
        assert cfg.escalation_model == ""
        assert cfg.escalation_threshold == ESCALATION_THRESHOLD
        assert cfg.auto_deescalate is True

    def test_custom_config(self):
        cfg = CascadeConfig(
            primary_model="qwen2.5-coder:3b",
            escalation_model="qwen2.5-coder:7b",
            escalation_threshold=5,
            auto_deescalate=False,
        )
        assert cfg.primary_model == "qwen2.5-coder:3b"
        assert cfg.escalation_threshold == 5


class TestStuckDetection:
    """Tests for poor response detection."""

    def test_empty_response_is_poor(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        assert agent._is_poor_response("") is True

    def test_short_response_is_poor(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        assert agent._is_poor_response("ok") is True
        assert agent._is_poor_response("x" * (MIN_USEFUL_LENGTH - 1)) is True

    def test_long_response_is_not_poor(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        good_response = "This is a helpful and detailed response. " * 5
        assert agent._is_poor_response(good_response) is False

    def test_stuck_patterns_detected(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        for pattern in STUCK_PATTERNS:
            # Build a response long enough to pass length check
            response = f"{pattern}. Let me explain in more detail why this is the case."
            assert agent._is_poor_response(response) is True, f"Pattern not detected: {pattern}"

    def test_stuck_patterns_case_insensitive(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        response = "I DON'T KNOW the answer to that question, sorry about that my friend."
        assert agent._is_poor_response(response) is True


class TestCascadeStats:
    """Tests for cascade stats tracking."""

    def test_stats_include_cascade_info(self):
        client = OllamaClient(base_url="http://localhost:19999")
        cfg = CascadeConfig(primary_model="test:3b", escalation_model="test:7b")
        agent = CascadeAgent(client, cascade_config=cfg)
        stats = agent.get_stats()
        assert "cascade" in stats
        assert stats["cascade"]["primary_model"] == "test:3b"
        assert stats["cascade"]["escalation_model"] == "test:7b"
        assert stats["cascade"]["is_escalated"] is False
        assert stats["cascade"]["consecutive_poor"] == 0

    def test_initial_state(self):
        client = OllamaClient(base_url="http://localhost:19999")
        agent = CascadeAgent(client)
        assert agent._is_escalated is False
        assert agent._consecutive_poor == 0


class TestAutoCascadeConfig:
    """Tests for automatic cascade configuration."""

    def test_no_models_fit(self):
        """With tiny GPU, should return empty config."""
        cfg = auto_cascade_config(0.5)
        assert cfg.primary_model == "" or cfg.escalation_model == ""

    def test_large_gpu_has_both_models(self):
        """With large GPU, should have both primary and escalation."""
        cfg = auto_cascade_config(24.0)
        # With 24 GB, should fit at least two coding models
        if cfg.primary_model:
            assert cfg.primary_model != cfg.escalation_model


class TestStuckPatterns:
    """Tests for the stuck patterns list."""

    def test_patterns_are_strings(self):
        for p in STUCK_PATTERNS:
            assert isinstance(p, str)
            assert len(p) > 3

    def test_min_useful_length_reasonable(self):
        assert MIN_USEFUL_LENGTH >= 20
        assert MIN_USEFUL_LENGTH <= 200

    def test_escalation_threshold_reasonable(self):
        assert ESCALATION_THRESHOLD >= 2
        assert ESCALATION_THRESHOLD <= 10
