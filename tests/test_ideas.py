"""Tests for community ideas collection."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.community.ideas import Idea, IdeaCollector


class TestIdea:
    """Tests for the Idea dataclass."""

    def test_basic_fields(self):
        idea = Idea(
            id="abc12345",
            category="feature",
            title="Add dark mode",
            description="Support dark mode in terminal UI",
            submitted_at=1000.0,
        )
        assert idea.id == "abc12345"
        assert idea.category == "feature"
        assert idea.title == "Add dark mode"

    def test_default_source(self):
        idea = Idea(id="x", category="improvement", title="t", description="d", submitted_at=0)
        assert idea.source == "user"

    def test_default_status(self):
        idea = Idea(id="x", category="improvement", title="t", description="d", submitted_at=0)
        assert idea.status == "new"

    def test_default_votes(self):
        idea = Idea(id="x", category="improvement", title="t", description="d", submitted_at=0)
        assert idea.votes == 1


class TestIdeaCollectorInit:
    """Tests for IdeaCollector initialization."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_dir = Path(tmpdir) / "ideas"
            collector = IdeaCollector(ideas_dir=str(ideas_dir))
            assert ideas_dir.exists()

    def test_enabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            assert collector.enabled is True

    def test_can_disable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir, enabled=False)
            assert collector.enabled is False

    def test_empty_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            assert len(collector._ideas) == 0


class TestSubmit:
    """Tests for submitting ideas."""

    def test_submit_returns_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.submit("Add dark mode", "Support dark mode in TUI")
            assert "submitted" in result.lower()
            assert "ID:" in result

    def test_submitted_idea_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Feature X", "Description of feature X")
            assert len(collector._ideas) == 1

    def test_submit_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir, enabled=False)
            result = collector.submit("Feature", "Description")
            assert "disabled" in result.lower()
            assert len(collector._ideas) == 0

    def test_duplicate_adds_vote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Same idea", "Same description")
            result = collector.submit("Same idea", "Same description")
            assert "vote" in result.lower()
            assert len(collector._ideas) == 1
            idea = list(collector._ideas.values())[0]
            assert idea.votes == 2

    def test_different_ideas_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea A", "First idea")
            collector.submit("Idea B", "Second idea")
            assert len(collector._ideas) == 2

    def test_custom_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Fix bug", "Fix the crash", category="bugfix")
            idea = list(collector._ideas.values())[0]
            assert idea.category == "bugfix"

    def test_custom_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("AI suggestion", "From the agent", source="agent")
            idea = list(collector._ideas.values())[0]
            assert idea.source == "agent"


class TestListIdeas:
    """Tests for listing ideas."""

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea Alpha", "desc A")
            collector.submit("Idea Bravo", "desc B")
            ideas = collector.list_ideas()
            assert len(ideas) == 2

    def test_filter_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea Alpha", "desc A")
            collector.submit("Idea Bravo", "desc B")
            # All should be "new"
            ideas = collector.list_ideas(status="new")
            assert len(ideas) == 2
            ideas = collector.list_ideas(status="accepted")
            assert len(ideas) == 0

    def test_filter_by_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Feature request item", "desc", category="feature")
            collector.submit("Bugfix report item", "desc", category="bugfix")
            ideas = collector.list_ideas(category="bugfix")
            assert len(ideas) == 1
            assert ideas[0].category == "bugfix"

    def test_sorted_by_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("First", "submitted first")
            collector.submit("Second", "submitted second")
            ideas = collector.list_ideas()
            # Most recent first
            assert ideas[0].title == "Second"


class TestGetNewIdeas:
    """Tests for get_new_ideas."""

    def test_returns_new_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea Alpha", "desc A")
            collector.submit("Idea Bravo", "desc B")
            # Mark one as evaluated
            idea_id = list(collector._ideas.keys())[0]
            collector.update_status(idea_id, "evaluated")
            new = collector.get_new_ideas()
            assert len(new) == 1


class TestUpdateStatus:
    """Tests for update_status."""

    def test_update_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Idea for update test", "description")
            idea_id = list(collector._ideas.keys())[0]
            result = collector.update_status(idea_id, "accepted")
            assert "updated" in result.lower()
            assert collector._ideas[idea_id].status == "accepted"

    def test_update_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.update_status("nonexistent", "accepted")
            assert "not found" in result.lower()


class TestFormatIdeas:
    """Tests for format_ideas."""

    def test_format_with_ideas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Dark mode", "Add dark mode to TUI", category="feature")
            result = collector.format_ideas()
            assert "Community Ideas" in result
            assert "Dark mode" in result
            assert "[feature]" in result
            assert "[new]" in result

    def test_format_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            result = collector.format_ideas()
            assert "No community ideas" in result

    def test_status_icons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Done idea", "Already implemented")
            idea_id = list(collector._ideas.keys())[0]
            collector.update_status(idea_id, "implemented")
            result = collector.format_ideas()
            assert "[done]" in result


class TestPersistence:
    """Tests for save/load round-trip."""

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Persistent idea", "Should survive reload")

            # Reload
            collector2 = IdeaCollector(ideas_dir=tmpdir)
            assert len(collector2._ideas) == 1
            idea = list(collector2._ideas.values())[0]
            assert idea.title == "Persistent idea"

    def test_handles_corrupted_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_file = Path(tmpdir) / "community_ideas.jsonl"
            ideas_file.write_text('not valid json\n{"bad": "data"}\n')
            collector = IdeaCollector(ideas_dir=tmpdir)
            # Should not crash, just skip bad lines
            assert len(collector._ideas) == 0

    def test_votes_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = IdeaCollector(ideas_dir=tmpdir)
            collector.submit("Voted idea", "Popular one")
            collector.submit("Voted idea", "Popular one")  # +1 vote

            collector2 = IdeaCollector(ideas_dir=tmpdir)
            idea = list(collector2._ideas.values())[0]
            assert idea.votes == 2
