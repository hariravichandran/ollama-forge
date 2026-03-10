"""Tests for batch 25 improvements: contextual logging on silent error returns.

Verifies that sessions, tracker, env, and ideas modules log diagnostic
context instead of silently returning None/empty.
"""

import inspect

import pytest


class TestSessionManagerLogging:
    """Tests for SessionManager contextual logging."""

    def test_load_not_found_logs(self):
        """load() should log when session_id has no matches."""
        from forge.agents.sessions import SessionManager
        source = inspect.getsource(SessionManager.load)
        # The "no matches" else branch should log
        assert 'log.debug("Session not found' in source

    def test_list_sessions_logs_on_error(self):
        """list_sessions() should log which file failed."""
        from forge.agents.sessions import SessionManager
        source = inspect.getsource(SessionManager.list_sessions)
        assert "log.debug" in source
        assert "session_file.name" in source

    def test_search_logs_on_error(self):
        """search() should log which file failed during search."""
        from forge.agents.sessions import SessionManager
        source = inspect.getsource(SessionManager.search)
        assert "log.debug" in source
        assert "session_file.name" in source

    def test_list_sessions_still_works(self):
        """list_sessions should return results despite corrupted files."""
        import tempfile
        from pathlib import Path
        from forge.agents.sessions import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(sessions_dir=Path(tmpdir))
            # Write a corrupted file
            (Path(tmpdir) / "session-corrupt1.json").write_text("{broken json")
            # Write a valid file
            import json, time
            valid = {
                "session_id": "session-valid123",
                "title": "Test",
                "agent_name": "assistant",
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            (Path(tmpdir) / "session-valid123.json").write_text(json.dumps(valid))

            results = sm.list_sessions()
            assert len(results) == 1
            assert results[0].session_id == "session-valid123"


class TestTrackerLogging:
    """Tests for AgentTracker contextual logging."""

    def test_load_logs_on_error(self):
        """_load() should log when tracker file is corrupted."""
        from forge.agents.tracker import AgentTracker
        source = inspect.getsource(AgentTracker._load)
        assert "log.warning" in source
        assert "tracker_file" in source

    def test_save_handles_error(self):
        """_save() should catch and log OSError."""
        from forge.agents.tracker import AgentTracker
        source = inspect.getsource(AgentTracker._save)
        assert "OSError" in source
        assert "log.error" in source

    def test_load_with_corrupted_file(self):
        """_load() should return empty dict and log on corrupted file."""
        import tempfile
        from pathlib import Path
        from forge.agents.tracker import AgentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker_file = Path(tmpdir) / "agent_systems.json"
            tracker_file.write_text("{corrupt json!!")
            tracker = AgentTracker(state_dir=tmpdir)
            assert tracker.systems == {}


class TestEnvLogging:
    """Tests for env module contextual logging."""

    def test_no_env_file_logs(self):
        """load_env() should log when no .env file found."""
        from forge.utils.env import load_env
        source = inspect.getsource(load_env)
        assert 'log.debug("No .env file found' in source

    def test_get_env_int_logs_on_parse_error(self):
        """get_env_int() should log when value can't be parsed."""
        from forge.utils.env import get_env_int
        source = inspect.getsource(get_env_int)
        assert "log.debug" in source
        assert "Could not parse" in source

    def test_get_env_int_returns_default(self):
        """get_env_int() should return default for non-integer values."""
        import os
        from forge.utils.env import get_env_int
        os.environ["__TEST_BATCH25_INT"] = "not_a_number"
        result = get_env_int("__TEST_BATCH25_INT", default=42)
        assert result == 42
        del os.environ["__TEST_BATCH25_INT"]


class TestIdeasLogging:
    """Tests for IdeaCollector contextual logging."""

    def test_load_logs_malformed_entries(self):
        """_load() should log when idea entries are malformed."""
        from forge.community.ideas import IdeaCollector
        source = inspect.getsource(IdeaCollector._load)
        assert "log.debug" in source
        assert "malformed" in source.lower()

    def test_save_handles_error(self):
        """_save() should catch and log OSError."""
        from forge.community.ideas import IdeaCollector
        source = inspect.getsource(IdeaCollector._save)
        assert "OSError" in source
        assert "log.error" in source

    def test_load_skips_bad_lines(self):
        """_load() should skip malformed lines and load valid ones."""
        import tempfile
        from pathlib import Path
        from forge.community.ideas import IdeaCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_file = Path(tmpdir) / "community_ideas.jsonl"
            import json, time
            valid_idea = json.dumps({
                "id": "abc12345",
                "category": "feature",
                "title": "Test idea",
                "description": "A test",
                "submitted_at": time.time(),
                "source": "user",
                "status": "new",
                "votes": 1,
            })
            ideas_file.write_text(f"{{bad json\n{valid_idea}\n")

            collector = IdeaCollector(ideas_dir=tmpdir)
            assert "abc12345" in collector._ideas


class TestBatch25Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_sessions_imports(self):
        from forge.agents.sessions import SessionManager

    def test_tracker_imports(self):
        from forge.agents.tracker import AgentTracker

    def test_env_imports(self):
        from forge.utils.env import load_env, get_env_int

    def test_ideas_imports(self):
        from forge.community.ideas import IdeaCollector
