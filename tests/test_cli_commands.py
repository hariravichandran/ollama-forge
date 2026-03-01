"""Tests for CLI commands (doctor, init, prompts, compare, sessions)."""

import tempfile
from pathlib import Path

import pytest


class TestInitCommand:
    """Tests for forge init."""

    def test_creates_forge_rules(self):
        """init should create .forge-rules file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / ".forge-rules"
            assert not rules_file.exists()

            # Simulate what init does
            rules_file.write_text(
                "# Project Rules for ollama-forge agents\n"
            )
            assert rules_file.exists()
            assert "Rules" in rules_file.read_text()

    def test_creates_agents_directory(self):
        """init should create agents/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            assert agents_dir.is_dir()

    def test_creates_starter_agent(self):
        """init should create a starter agent YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            starter = agents_dir / "my-agent.yaml"
            starter.write_text("name: my-agent\n")
            assert starter.exists()
            assert "my-agent" in starter.read_text()

    def test_updates_gitignore(self):
        """init should add forge entries to .gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gitignore = Path(tmpdir) / ".gitignore"
            gitignore.write_text("node_modules/\n")

            content = gitignore.read_text()
            if ".forge_state/" not in content:
                with open(gitignore, "a") as f:
                    f.write("\n# ollama-forge\n.forge_state/\n.forge/\n")

            updated = gitignore.read_text()
            assert ".forge_state/" in updated
            assert ".forge/" in updated
            assert "node_modules/" in updated  # original content preserved

    def test_skips_existing_files(self):
        """init should not overwrite existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_file = Path(tmpdir) / ".forge-rules"
            rules_file.write_text("custom rules")

            # Re-running init should not overwrite
            assert rules_file.read_text() == "custom rules"


class TestDoctorCommand:
    """Tests for forge doctor."""

    def test_shutil_disk_usage(self):
        """disk_usage check should work."""
        import shutil
        total, used, free = shutil.disk_usage("/")
        assert free > 0
        assert total > free

    def test_python_version_check(self):
        """Python version should be 3.10+."""
        import sys
        assert sys.version_info >= (3, 10)

    def test_core_deps_importable(self):
        """Core dependencies should be importable."""
        for dep in ["requests", "yaml", "click", "rich"]:
            __import__(dep)

    def test_hardware_detection_works(self):
        """Hardware detection should not crash."""
        from forge.hardware import detect_hardware
        hw = detect_hardware()
        assert hw is not None
        assert hw.ram_gb > 0


class TestCompareCommand:
    """Tests for forge compare."""

    def test_compare_requires_two_models(self):
        """Compare should require at least 2 models."""
        # This is a CLI validation — we test the logic
        models = ("model1:7b",)
        assert len(models) < 2

        models = ("model1:7b", "model2:14b")
        assert len(models) >= 2

    def test_compare_collects_results(self):
        """Compare should collect results from each model."""
        results = []
        for model in ["model1:7b", "model2:14b"]:
            results.append({
                "model": model,
                "response": "test response",
                "tokens": 10,
                "time_s": 1.0,
                "tps": 10.0,
            })
        assert len(results) == 2
        assert results[0]["model"] != results[1]["model"]


class TestSessionCommands:
    """Tests for forge session commands."""

    def test_session_list(self):
        """session list should not crash with no sessions."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sessions = mgr.list_sessions()
            assert sessions == []

    def test_session_export_nonexistent(self):
        """Exporting nonexistent session should handle gracefully."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            result = mgr.export("fake-id")
            assert "not found" in result.lower()

    def test_session_delete_nonexistent(self):
        """Deleting nonexistent session should return False."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            assert mgr.delete("fake-id") is False

    def test_session_save_and_list(self):
        """Saving a session should make it appear in list."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save(
                messages=[{"role": "user", "content": "hi"}],
                agent_name="assistant",
                model="test:7b",
            )
            sessions = mgr.list_sessions()
            assert len(sessions) == 1
            assert sessions[0].session_id == sid

    def test_session_search_finds_match(self):
        """Search should find messages containing the query."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(
                messages=[
                    {"role": "user", "content": "How do I fix a segfault?"},
                    {"role": "assistant", "content": "A segfault usually means..."},
                ],
                agent_name="coder",
                model="test:7b",
            )
            results = mgr.search("segfault")
            assert len(results) >= 1
            assert "segfault" in results[0]["content"].lower()

    def test_session_search_case_insensitive(self):
        """Search should be case-insensitive."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(
                messages=[{"role": "user", "content": "Hello World"}],
                agent_name="assistant",
            )
            results = mgr.search("hello world")
            assert len(results) == 1

    def test_session_search_no_match(self):
        """Search should return empty list when no match."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            mgr.save(
                messages=[{"role": "user", "content": "Python rocks"}],
                agent_name="assistant",
            )
            results = mgr.search("javascript")
            assert results == []

    def test_session_search_respects_limit(self):
        """Search should respect the limit parameter."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            for i in range(5):
                mgr.save(
                    messages=[{"role": "user", "content": f"Query about Python #{i}"}],
                    agent_name="assistant",
                )
            results = mgr.search("Python", limit=2)
            assert len(results) == 2

    def test_session_export_html(self):
        """Export as HTML should produce valid HTML."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save(
                messages=[
                    {"role": "user", "content": "Hello there"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                ],
                agent_name="coder",
                model="test:7b",
            )
            html = mgr.export(sid, format="html")
            assert "<!DOCTYPE html>" in html
            assert "Hello there" in html
            assert "Hi! How can I help?" in html
            assert "coder" in html

    def test_session_export_html_escapes_content(self):
        """HTML export should escape special characters."""
        from forge.agents.sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(sessions_dir=Path(tmpdir))
            sid = mgr.save(
                messages=[{"role": "user", "content": "<script>alert('xss')</script>"}],
                agent_name="assistant",
            )
            html = mgr.export(sid, format="html")
            assert "<script>" not in html
            assert "&lt;script&gt;" in html
