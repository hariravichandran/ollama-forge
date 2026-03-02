"""Tests for the agent tracker: managing and monitoring agent systems."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from forge.agents.tracker import AgentTracker, AgentSystemInfo


class TestAgentSystemInfo:
    """Tests for AgentSystemInfo dataclass."""

    def test_basic_fields(self):
        info = AgentSystemInfo(
            name="my-system",
            system_type="single",
            agents=["coder"],
            description="A coding helper",
        )
        assert info.name == "my-system"
        assert info.system_type == "single"
        assert info.agents == ["coder"]
        assert info.description == "A coding helper"

    def test_default_counters(self):
        info = AgentSystemInfo(name="s", system_type="single", agents=["a"])
        assert info.total_messages == 0
        assert info.total_tool_calls == 0

    def test_timestamps_auto_set(self):
        before = time.time()
        info = AgentSystemInfo(name="s", system_type="multi", agents=["a", "b"])
        after = time.time()
        assert before <= info.created_at <= after
        assert before <= info.last_active <= after

    def test_multi_agent_list(self):
        info = AgentSystemInfo(
            name="team", system_type="multi",
            agents=["coder", "researcher", "reviewer"],
        )
        assert len(info.agents) == 3


class TestAgentTrackerInit:
    """Tests for AgentTracker initialization."""

    def test_creates_state_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            tracker = AgentTracker(state_dir=str(state_dir))
            assert state_dir.exists()

    def test_empty_systems_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            assert len(tracker.systems) == 0

    def test_loads_existing_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            # Write some state
            data = {
                "test-sys": {
                    "name": "test-sys",
                    "system_type": "single",
                    "agents": ["assistant"],
                    "description": "test",
                    "created_at": 1000.0,
                    "last_active": 1000.0,
                    "total_messages": 5,
                    "total_tool_calls": 2,
                }
            }
            (state_dir / "agent_systems.json").write_text(json.dumps(data))
            tracker = AgentTracker(state_dir=str(state_dir))
            assert "test-sys" in tracker.systems
            assert tracker.systems["test-sys"].total_messages == 5


class TestCreateSystem:
    """Tests for creating agent systems."""

    def test_create_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            result = tracker.create_system("my-agent", "single", ["assistant"])
            assert "Created" in result
            assert "my-agent" in result
            assert "my-agent" in tracker.systems

    def test_create_multi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            result = tracker.create_system(
                "team", "multi", ["coder", "researcher"], description="Dev team",
            )
            assert "multi" in result
            assert len(tracker.systems["team"].agents) == 2

    def test_duplicate_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys1", "single", ["a"])
            result = tracker.create_system("sys1", "single", ["b"])
            assert "already exists" in result

    def test_create_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = str(Path(tmpdir) / "state")
            tracker = AgentTracker(state_dir=state_dir)
            tracker.create_system("persistent", "single", ["coder"])

            # Reload from disk
            tracker2 = AgentTracker(state_dir=state_dir)
            assert "persistent" in tracker2.systems


class TestDeleteSystem:
    """Tests for deleting agent systems."""

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("to-delete", "single", ["a"])
            result = tracker.delete_system("to-delete")
            assert "Deleted" in result
            assert "to-delete" not in tracker.systems

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            result = tracker.delete_system("nope")
            assert "not found" in result


class TestRecordActivity:
    """Tests for recording agent activity."""

    def test_record_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            tracker.record_activity("sys", messages=5)
            assert tracker.systems["sys"].total_messages == 5

    def test_record_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            tracker.record_activity("sys", tool_calls=3)
            assert tracker.systems["sys"].total_tool_calls == 3

    def test_cumulative_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            tracker.record_activity("sys", messages=2, tool_calls=1)
            tracker.record_activity("sys", messages=3, tool_calls=2)
            assert tracker.systems["sys"].total_messages == 5
            assert tracker.systems["sys"].total_tool_calls == 3

    def test_updates_last_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            old_active = tracker.systems["sys"].last_active
            time.sleep(0.01)
            tracker.record_activity("sys", messages=1)
            assert tracker.systems["sys"].last_active >= old_active

    def test_nonexistent_system_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            # Should not crash
            tracker.record_activity("nonexistent", messages=1)


class TestListSystems:
    """Tests for listing agent systems."""

    def test_empty_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            result = tracker.list_systems()
            assert "No agent systems" in result

    def test_lists_created_systems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("coder-bot", "single", ["coder"])
            tracker.create_system("team", "multi", ["coder", "researcher"])
            result = tracker.list_systems()
            assert "coder-bot" in result
            assert "team" in result
            assert "[single]" in result
            assert "[multi]" in result

    def test_shows_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            tracker.record_activity("sys", messages=10, tool_calls=3)
            result = tracker.list_systems()
            assert "msgs: 10" in result
            assert "tools: 3" in result


class TestGetSystem:
    """Tests for get_system."""

    def test_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            tracker.create_system("sys", "single", ["a"])
            info = tracker.get_system("sys")
            assert info is not None
            assert info.name == "sys"

    def test_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = AgentTracker(state_dir=str(Path(tmpdir) / "state"))
            assert tracker.get_system("nope") is None


class TestPersistence:
    """Tests for save/load round-trip."""

    def test_corrupted_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (state_dir / "agent_systems.json").write_text("not json{{{")
            tracker = AgentTracker(state_dir=str(state_dir))
            assert len(tracker.systems) == 0

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = str(Path(tmpdir) / "state")
            tracker = AgentTracker(state_dir=state_dir)
            tracker.create_system("a", "single", ["assistant"], description="Helper")
            tracker.record_activity("a", messages=7, tool_calls=2)

            tracker2 = AgentTracker(state_dir=state_dir)
            sys = tracker2.get_system("a")
            assert sys is not None
            assert sys.description == "Helper"
            assert sys.total_messages == 7
            assert sys.total_tool_calls == 2
