"""Tests for session persistence: save, load, search, export."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from forge.agents.sessions import Session, SessionSummary, SessionManager


class TestSession:
    """Tests for the Session dataclass."""

    def test_session_fields(self):
        s = Session(
            session_id="session-abc12345",
            title="Test session",
            agent_name="coder",
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "Hello"}],
            created_at=1000.0,
            updated_at=2000.0,
        )
        assert s.session_id == "session-abc12345"
        assert s.title == "Test session"
        assert s.agent_name == "coder"
        assert s.model == "qwen2.5:7b"

    def test_message_count(self):
        s = Session(
            session_id="s1", title="t", agent_name="a", model="m",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            created_at=0, updated_at=0,
        )
        assert s.message_count == 2

    def test_duration(self):
        s = Session(
            session_id="s1", title="t", agent_name="a", model="m",
            messages=[],
            created_at=1000.0, updated_at=1060.0,
        )
        assert s.duration_s == 60.0

    def test_summary_format(self):
        now = time.time()
        s = Session(
            session_id="session-abc12345", title="My chat", agent_name="coder",
            model="test:7b", messages=[{"role": "user", "content": "x"}],
            created_at=now - 300, updated_at=now - 60,
        )
        summary = s.summary()
        # session_id[:8] = "session-"
        assert "session-" in summary
        assert "My chat" in summary
        assert "coder" in summary
        assert "1 msgs" in summary

    def test_empty_metadata_default(self):
        s = Session(
            session_id="s1", title="t", agent_name="a", model="m",
            messages=[], created_at=0, updated_at=0,
        )
        assert s.metadata == {}

    def test_custom_metadata(self):
        s = Session(
            session_id="s1", title="t", agent_name="a", model="m",
            messages=[], created_at=0, updated_at=0,
            metadata={"tags": ["test"]},
        )
        assert s.metadata["tags"] == ["test"]


class TestSessionSummary:
    """Tests for the SessionSummary dataclass."""

    def test_fields(self):
        ss = SessionSummary(
            session_id="s1", title="Test", agent_name="assistant",
            model="test:7b", message_count=5,
            created_at=1000.0, updated_at=2000.0,
        )
        assert ss.session_id == "s1"
        assert ss.message_count == 5


class TestSessionManager:
    """Tests for SessionManager save/load/list."""

    def _make_manager(self, tmpdir):
        return SessionManager(sessions_dir=Path(tmpdir))

    def test_save_returns_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(
                messages=[{"role": "user", "content": "hello"}],
                agent_name="assistant",
            )
            assert sid.startswith("session-")
            assert len(sid) > 10

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(messages=[{"role": "user", "content": "hi"}])
            session_file = Path(tmpdir) / f"{sid}.json"
            assert session_file.exists()

    def test_load_returns_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(
                messages=[{"role": "user", "content": "test"}],
                title="Load test",
            )
            loaded = mgr.load(sid)
            assert loaded is not None
            assert loaded.title == "Load test"
            assert loaded.message_count == 1

    def test_load_partial_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(messages=[{"role": "user", "content": "x"}])
            # Load with first 12 chars
            partial = sid[:12]
            loaded = mgr.load(partial)
            assert loaded is not None
            assert loaded.session_id == sid

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            assert mgr.load("nonexistent-session") is None

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            mgr.save(messages=[{"role": "user", "content": "first"}], title="First")
            mgr.save(messages=[{"role": "user", "content": "second"}], title="Second")
            sessions = mgr.list_sessions()
            assert len(sessions) == 2
            # Most recent first
            titles = [s.title for s in sessions]
            assert "Second" in titles

    def test_list_sessions_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            for i in range(5):
                mgr.save(messages=[{"role": "user", "content": f"msg {i}"}])
            sessions = mgr.list_sessions(limit=3)
            assert len(sessions) == 3

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(messages=[{"role": "user", "content": "delete me"}])
            assert mgr.delete(sid) is True
            assert mgr.load(sid) is None

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            assert mgr.delete("nonexistent") is False

    def test_update_existing_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(
                messages=[{"role": "user", "content": "v1"}],
                title="Original",
            )
            # Update with more messages
            mgr.save(
                messages=[
                    {"role": "user", "content": "v1"},
                    {"role": "assistant", "content": "response"},
                ],
                session_id=sid,
                title="Updated",
            )
            loaded = mgr.load(sid)
            assert loaded is not None
            assert loaded.title == "Updated"
            assert loaded.message_count == 2

    def test_auto_title_from_first_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            sid = mgr.save(messages=[{"role": "user", "content": "How do I install Python?"}])
            loaded = mgr.load(sid)
            assert loaded is not None
            assert "install Python" in loaded.title


class TestSessionExport:
    """Tests for session export (markdown, JSON, HTML)."""

    def _save_session(self, mgr):
        return mgr.save(
            messages=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            agent_name="coder",
            model="test:7b",
            title="Python Q&A",
        )

    def test_export_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = self._save_session(mgr)
            md = mgr.export(sid, format="markdown")
            assert "# Chat Session: Python Q&A" in md
            assert "**User:**" in md
            assert "**Assistant:**" in md
            assert "Python is a programming language" in md

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = self._save_session(mgr)
            result = mgr.export(sid, format="json")
            data = json.loads(result)
            assert data["title"] == "Python Q&A"
            assert len(data["messages"]) == 2

    def test_export_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = self._save_session(mgr)
            html = mgr.export(sid, format="html")
            assert "<!DOCTYPE html>" in html
            assert "Python Q&amp;A" in html or "Python Q&A" in html
            assert "message user" in html
            assert "message assistant" in html

    def test_export_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            result = mgr.export("nonexistent")
            assert "not found" in result.lower()


class TestSessionSearch:
    """Tests for searching across sessions."""

    def test_search_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(
                messages=[{"role": "user", "content": "How to sort a list in Python?"}],
                title="Sorting help",
            )
            mgr.save(
                messages=[{"role": "user", "content": "What is JavaScript?"}],
                title="JS question",
            )
            results = mgr.search("sort")
            assert len(results) >= 1
            assert any("sort" in r["content"].lower() for r in results)

    def test_search_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(messages=[{"role": "user", "content": "PYTHON is great"}])
            results = mgr.search("python")
            assert len(results) >= 1

    def test_search_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            for i in range(10):
                mgr.save(messages=[{"role": "user", "content": f"common word {i}"}])
            results = mgr.search("common", limit=3)
            assert len(results) <= 3

    def test_search_no_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(messages=[{"role": "user", "content": "hello world"}])
            results = mgr.search("xyznonexistent")
            assert len(results) == 0
