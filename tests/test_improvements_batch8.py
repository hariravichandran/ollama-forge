"""Tests for batch 8 improvements: memory limits, session recovery, cascade metrics, env parsing."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# === Memory: Fact Limits & TTL ===

class TestMemoryFactLimits:
    """Tests for memory fact validation and limits."""

    def test_reject_empty_key(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("", "value")
            assert len(mem._facts) == 0

    def test_reject_empty_value(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key", "")
            assert len(mem._facts) == 0

    def test_key_truncated(self):
        from forge.agents.memory import ConversationMemory, MAX_FACT_KEY_LENGTH
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            long_key = "k" * (MAX_FACT_KEY_LENGTH + 50)
            mem.store_fact(long_key, "value")
            assert len(mem._facts) == 1
            stored_key = list(mem._facts.keys())[0]
            assert len(stored_key) == MAX_FACT_KEY_LENGTH

    def test_value_truncated(self):
        from forge.agents.memory import ConversationMemory, MAX_FACT_VALUE_LENGTH
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            long_value = "v" * (MAX_FACT_VALUE_LENGTH + 50)
            mem.store_fact("key", long_value)
            fact = list(mem._facts.values())[0]
            assert len(fact.value) == MAX_FACT_VALUE_LENGTH

    def test_max_facts_eviction(self):
        from forge.agents.memory import ConversationMemory
        import forge.agents.memory as mem_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            # Use a small limit to avoid slow test with 500 facts
            original = mem_mod.MAX_FACTS
            mem_mod.MAX_FACTS = 5
            try:
                # Use completely different values to avoid dedup similarity matching
                distinct_values = [
                    "The sky is blue and clouds are white",
                    "Python was created by Guido van Rossum",
                    "Tokyo is the capital city of Japan",
                    "Water boils at one hundred degrees celsius",
                    "The speed of light is very fast indeed",
                ]
                for i, val in enumerate(distinct_values):
                    mem.store_fact(f"fact_{i}", val)
                assert len(mem._facts) == 5
                # Add one more — should evict oldest
                mem.store_fact("new_key", "Elephants are the largest land animals")
                assert len(mem._facts) == 5
                assert "new_key" in mem._facts
            finally:
                mem_mod.MAX_FACTS = original


class TestMemoryFactTTL:
    """Tests for fact TTL pruning."""

    def test_prune_old_facts(self):
        from forge.agents.memory import ConversationMemory, MemoryFact, FACT_TTL_DAYS
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            # Add an old fact directly
            old_ts = time.time() - (FACT_TTL_DAYS + 1) * 86400
            mem._facts["old"] = MemoryFact(key="old", value="old_val", source="test", timestamp=old_ts)
            mem._facts["new"] = MemoryFact(key="new", value="new_val", source="test", timestamp=time.time())
            pruned = mem._prune_old_facts()
            assert pruned == 1
            assert "old" not in mem._facts
            assert "new" in mem._facts

    def test_no_prune_if_all_fresh(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("fresh", "fresh_val")
            pruned = mem._prune_old_facts()
            assert pruned == 0


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_empty_stats(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            stats = mem.get_stats()
            assert stats["fact_count"] == 0

    def test_stats_with_facts(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("name", "Alice")
            mem.store_fact("lang", "Python")
            stats = mem.get_stats()
            assert stats["fact_count"] == 2
            assert stats["avg_confidence"] == 1.0

    def test_constants(self):
        from forge.agents.memory import MAX_FACTS, MAX_FACT_KEY_LENGTH, MAX_FACT_VALUE_LENGTH, FACT_TTL_DAYS
        assert MAX_FACTS > 0
        assert MAX_FACT_KEY_LENGTH > 0
        assert MAX_FACT_VALUE_LENGTH > 0
        assert FACT_TTL_DAYS > 0


# === Sessions: Recovery & Limits ===

class TestSessionLimits:
    """Tests for session size and count limits."""

    def test_max_messages_constant(self):
        from forge.agents.sessions import MAX_MESSAGES_PER_SESSION
        assert MAX_MESSAGES_PER_SESSION > 0
        assert MAX_MESSAGES_PER_SESSION <= 100_000

    def test_max_session_size_constant(self):
        from forge.agents.sessions import MAX_SESSION_SIZE_MB
        assert MAX_SESSION_SIZE_MB > 0
        assert MAX_SESSION_SIZE_MB <= 100

    def test_max_sessions_constant(self):
        from forge.agents.sessions import MAX_SESSIONS_ON_DISK
        assert MAX_SESSIONS_ON_DISK > 0


class TestSessionCorruptionRecovery:
    """Tests for session corruption handling."""

    def test_load_corrupted_json(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            # Write corrupted JSON
            bad_path = Path(tmpdir) / "session-bad.json"
            bad_path.write_text("{not valid json!!")
            result = mgr.load("session-bad")
            assert result is None
            # Should have created .corrupted backup
            assert bad_path.with_suffix(".json.corrupted").exists()

    def test_load_missing_fields(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            # Write JSON without required fields
            bad_path = Path(tmpdir) / "session-incomplete.json"
            bad_path.write_text('{"title": "test"}')
            result = mgr.load("session-incomplete")
            assert result is None

    def test_load_valid_session(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save(
                messages=[{"role": "user", "content": "hello"}],
                agent_name="assistant",
            )
            session = mgr.load(sid)
            assert session is not None
            assert len(session.messages) == 1


class TestSessionStats:
    """Tests for session storage statistics."""

    def test_empty_stats(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            stats = mgr.get_stats()
            assert stats["session_count"] == 0
            assert stats["total_size_mb"] == 0

    def test_stats_with_sessions(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save([{"role": "user", "content": "hi"}])
            stats = mgr.get_stats()
            assert stats["session_count"] == 1
            assert stats["total_size_mb"] >= 0  # small files round to 0.0 MB


class TestSessionCleanup:
    """Tests for old session cleanup."""

    def test_cleanup_no_op_when_under_limit(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save([{"role": "user", "content": "hi"}])
            removed = mgr.cleanup_old_sessions()
            assert removed == 0


# === Cascade: Escalation Metrics ===

class TestCascadeMetrics:
    """Tests for cascade escalation metrics."""

    def _make_mock_client(self, responses=None):
        responses = responses or []
        client = MagicMock()
        client.model = "test:7b"
        client.base_url = "http://localhost:11434"
        client.num_ctx = 8192
        client.keep_alive = "30m"

        class Stats:
            total_calls = 0
            total_tokens = 0
            total_prompt_tokens = 0
            errors = 0
            avg_time_s = 0.0
        client.stats = Stats()

        call_idx = [0]
        def chat_fn(messages, tools=None, temperature=0.7, timeout=300):
            resp = responses[call_idx[0]] if call_idx[0] < len(responses) else "Default"
            call_idx[0] += 1
            client.stats.total_calls += 1
            return {"response": resp, "tokens": 10, "prompt_tokens": 5, "time_s": 0.1}
        client.chat = chat_fn
        return client

    def test_initial_metrics_zero(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        client = self._make_mock_client()
        agent = CascadeAgent(client=client, cascade_config=CascadeConfig(
            primary_model="test:7b", escalation_model="test:14b"
        ))
        stats = agent.get_stats()
        assert stats["cascade"]["escalation_count"] == 0
        assert stats["cascade"]["deescalation_count"] == 0
        assert stats["cascade"]["escalation_successes"] == 0

    def test_stats_include_success_rate(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        client = self._make_mock_client()
        agent = CascadeAgent(client=client, cascade_config=CascadeConfig(
            primary_model="test:7b", escalation_model="test:14b"
        ))
        stats = agent.get_stats()
        assert "escalation_success_rate" in stats["cascade"]
        assert stats["cascade"]["escalation_success_rate"] == 0.0

    def test_same_model_warning(self):
        """Setting primary == escalation should log a warning (no crash)."""
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        client = self._make_mock_client()
        # Should not crash — just warns
        agent = CascadeAgent(client=client, cascade_config=CascadeConfig(
            primary_model="test:7b", escalation_model="test:7b"
        ))
        assert agent._primary_model == agent._escalation_model


class TestCascadeStuckPatterns:
    """Tests for stuck pattern detection."""

    def test_short_response_is_poor(self):
        from forge.agents.cascade import CascadeAgent
        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("OK") is True

    def test_long_response_is_good(self):
        from forge.agents.cascade import CascadeAgent
        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("x" * 100) is False

    def test_stuck_pattern_detected(self):
        from forge.agents.cascade import CascadeAgent
        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("I don't know how to do that. I cannot help with this.") is True

    def test_empty_response_is_poor(self):
        from forge.agents.cascade import CascadeAgent
        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock()
        agent = CascadeAgent(client=client)
        assert agent._is_poor_response("") is True


# === Env Utils: Parsing Improvements ===

class TestEnvParsing:
    """Tests for improved .env file parsing."""

    def test_inline_comment_stripped(self):
        from forge.utils.env import load_env
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("TEST_INLINE_COMMENT=value # this is a comment\n")
            # Clear any existing value
            os.environ.pop("TEST_INLINE_COMMENT", None)
            loaded = load_env(env_file)
            assert loaded.get("TEST_INLINE_COMMENT") == "value"
            os.environ.pop("TEST_INLINE_COMMENT", None)

    def test_quoted_value_preserved(self):
        from forge.utils.env import load_env
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text('TEST_QUOTED="value with # hash"\n')
            os.environ.pop("TEST_QUOTED", None)
            loaded = load_env(env_file)
            assert loaded.get("TEST_QUOTED") == "value with # hash"
            os.environ.pop("TEST_QUOTED", None)

    def test_export_prefix_stripped(self):
        from forge.utils.env import load_env
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("export TEST_EXPORT=exported_value\n")
            os.environ.pop("TEST_EXPORT", None)
            loaded = load_env(env_file)
            assert loaded.get("TEST_EXPORT") == "exported_value"
            os.environ.pop("TEST_EXPORT", None)

    def test_empty_file(self):
        from forge.utils.env import load_env
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            loaded = load_env(env_file)
            assert len(loaded) == 0

    def test_comments_only(self):
        from forge.utils.env import load_env
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("# Just a comment\n# Another comment\n")
            loaded = load_env(env_file)
            assert len(loaded) == 0

    def test_nonexistent_file(self):
        from forge.utils.env import load_env
        loaded = load_env(Path("/nonexistent/.env"))
        assert len(loaded) == 0


class TestEnvHelpers:
    """Tests for get_env_bool and get_env_int."""

    def test_get_env_bool_true_values(self):
        from forge.utils.env import get_env_bool
        import os
        for val in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL") is True
            os.environ.pop("TEST_BOOL")

    def test_get_env_bool_false_values(self):
        from forge.utils.env import get_env_bool
        import os
        for val in ["0", "false", "no", "off", "random"]:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL") is False
            os.environ.pop("TEST_BOOL")

    def test_get_env_bool_default(self):
        from forge.utils.env import get_env_bool
        import os
        os.environ.pop("TEST_BOOL_MISSING", None)
        assert get_env_bool("TEST_BOOL_MISSING") is False
        assert get_env_bool("TEST_BOOL_MISSING", True) is True

    def test_get_env_int(self):
        from forge.utils.env import get_env_int
        import os
        os.environ["TEST_INT"] = "42"
        assert get_env_int("TEST_INT") == 42
        os.environ.pop("TEST_INT")

    def test_get_env_int_default(self):
        from forge.utils.env import get_env_int
        import os
        os.environ.pop("TEST_INT_MISSING", None)
        assert get_env_int("TEST_INT_MISSING") == 0
        assert get_env_int("TEST_INT_MISSING", 99) == 99

    def test_get_env_int_invalid(self):
        from forge.utils.env import get_env_int
        import os
        os.environ["TEST_INT_BAD"] = "not_a_number"
        assert get_env_int("TEST_INT_BAD", 5) == 5
        os.environ.pop("TEST_INT_BAD")


# === Integration Tests ===

class TestBatch8Integration:
    """Integration tests across batch 8 improvements."""

    def test_memory_roundtrip_with_limits(self):
        from forge.agents.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = ConversationMemory(memory_dir=tmpdir)
            mem1.store_fact("lang", "Python")
            mem1.store_fact("editor", "VSCode")

            # Reload
            mem2 = ConversationMemory(memory_dir=tmpdir)
            assert len(mem2._facts) == 2
            assert mem2.get_fact("lang") == "Python"

    def test_session_save_load_with_recovery(self):
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save(
                messages=[{"role": "user", "content": "test message"}],
                agent_name="coder",
            )
            session = mgr.load(sid)
            assert session is not None
            assert session.agent_name == "coder"

    def test_cascade_stats_complete(self):
        from forge.agents.cascade import CascadeAgent, CascadeConfig
        client = MagicMock()
        client.model = "test:7b"
        client.stats = MagicMock()
        agent = CascadeAgent(client=client, cascade_config=CascadeConfig(
            primary_model="small", escalation_model="big",
        ))
        stats = agent.get_stats()
        cascade = stats["cascade"]
        assert "escalation_count" in cascade
        assert "deescalation_count" in cascade
        assert "escalation_success_rate" in cascade

    def test_env_parsing_integration(self):
        from forge.utils.env import load_env
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "# Config file\n"
                "export DB_HOST=localhost\n"
                "DB_PORT=5432 # default port\n"
                'DB_NAME="my_database"\n'
            )
            for key in ["DB_HOST", "DB_PORT", "DB_NAME"]:
                os.environ.pop(key, None)
            loaded = load_env(env_file)
            assert loaded["DB_HOST"] == "localhost"
            assert loaded["DB_PORT"] == "5432"
            assert loaded["DB_NAME"] == "my_database"
            for key in ["DB_HOST", "DB_PORT", "DB_NAME"]:
                os.environ.pop(key, None)
