"""Tests for ConversationMemory — persistent storage across sessions."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from forge.agents.memory import ConversationMemory, MemoryFact


class TestMemoryFact:
    """Tests for MemoryFact dataclass."""

    def test_basic_fact(self):
        f = MemoryFact(key="name", value="Alice", source="test", timestamp=1.0)
        assert f.key == "name"
        assert f.value == "Alice"
        assert f.confidence == 1.0

    def test_fact_with_confidence(self):
        f = MemoryFact(key="x", value="y", source="s", timestamp=1.0, confidence=0.5)
        assert f.confidence == 0.5


class TestConversationMemoryInit:
    """Tests for memory initialization."""

    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_dir = Path(tmpdir) / "memory"
            mem = ConversationMemory(memory_dir=mem_dir)
            assert mem_dir.exists()
            assert (mem_dir / "conversations").exists()

    def test_empty_facts_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            assert len(mem._facts) == 0


class TestSaveConversation:
    """Tests for save_conversation."""

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            path = mem.save_conversation(messages, session_id="test123")
            assert path  # non-empty
            assert Path(path).exists()

    def test_save_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            path = mem.save_conversation(messages, session_id="test123")
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["role"] == "user"

    def test_save_empty_messages_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            result = mem.save_conversation([])
            assert result == ""

    def test_save_auto_generates_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [{"role": "user", "content": "test"}]
            path = mem.save_conversation(messages)
            assert path
            assert Path(path).exists()

    def test_filename_includes_date_and_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [{"role": "user", "content": "test"}]
            path = mem.save_conversation(messages, session_id="abc")
            filename = Path(path).name
            assert "abc" in filename
            assert ".jsonl" in filename


class TestGetRecentContext:
    """Tests for get_recent_context."""

    def test_no_conversations_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            assert mem.get_recent_context() == []

    def test_loads_recent_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            mem.save_conversation(messages, session_id="s1")
            recent = mem.get_recent_context()
            assert len(recent) == 2
            assert recent[0]["content"] == "Hello"

    def test_respects_max_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
            mem.save_conversation(messages, session_id="s1")
            recent = mem.get_recent_context(max_messages=5)
            assert len(recent) == 5

    def test_returns_most_recent_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            messages = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
            mem.save_conversation(messages, session_id="s1")
            recent = mem.get_recent_context(max_messages=3)
            # Should return the LAST 3 messages
            assert recent[-1]["content"] == "msg19"


class TestStoreFact:
    """Tests for fact storage and retrieval."""

    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("language", "Python")
            assert mem.get_fact("language") == "Python"

    def test_overwrite_fact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("color", "red")
            mem.store_fact("color", "blue")
            assert mem.get_fact("color") == "blue"

    def test_get_nonexistent_fact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            assert mem.get_fact("nonexistent") is None

    def test_fact_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = ConversationMemory(memory_dir=tmpdir)
            mem1.store_fact("key1", "value1")

            # Load fresh
            mem2 = ConversationMemory(memory_dir=tmpdir)
            assert mem2.get_fact("key1") == "value1"

    def test_multiple_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("a", "1")
            mem.store_fact("b", "2")
            mem.store_fact("c", "3")
            assert mem.get_fact("a") == "1"
            assert mem.get_fact("b") == "2"
            assert mem.get_fact("c") == "3"


class TestGetFactsContext:
    """Tests for facts context formatting."""

    def test_empty_facts_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            assert mem.get_facts_context() == ""

    def test_formats_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("name", "Alice")
            mem.store_fact("role", "developer")
            context = mem.get_facts_context()
            assert "name: Alice" in context
            assert "role: developer" in context
            assert "Known facts" in context


class TestSummary:
    """Tests for conversation summary."""

    def test_save_and_get_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.save_summary("We discussed Python and testing.")
            assert mem.get_summary() == "We discussed Python and testing."

    def test_get_summary_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            assert mem.get_summary() == ""

    def test_summary_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = ConversationMemory(memory_dir=tmpdir)
            mem1.save_summary("Test summary")

            mem2 = ConversationMemory(memory_dir=tmpdir)
            assert mem2.get_summary() == "Test summary"


class TestClear:
    """Tests for clearing memory."""

    def test_clear_removes_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.save_conversation([{"role": "user", "content": "hi"}], session_id="s1")
            assert len(list((Path(tmpdir) / "conversations").glob("*.jsonl"))) == 1
            mem.clear()
            assert len(list((Path(tmpdir) / "conversations").glob("*.jsonl"))) == 0

    def test_clear_removes_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.store_fact("key", "value")
            assert mem.get_fact("key") == "value"
            mem.clear()
            assert mem.get_fact("key") is None

    def test_clear_removes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.save_summary("something")
            mem.clear()
            assert mem.get_summary() == ""

    def test_clear_on_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            mem.clear()  # should not crash


class TestLoadFacts:
    """Tests for fact loading edge cases."""

    def test_corrupted_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(memory_dir=tmpdir)
            # Write valid + corrupted line
            facts_file = Path(tmpdir) / "facts.jsonl"
            valid = json.dumps({"key": "k1", "value": "v1", "source": "s", "timestamp": time.time()})
            facts_file.write_text(f"{valid}\n{{invalid json\n")

            # Reload — should load the valid fact and skip the corrupted one
            mem2 = ConversationMemory(memory_dir=tmpdir)
            assert mem2.get_fact("k1") == "v1"

    def test_empty_facts_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "facts.jsonl").write_text("")
            mem = ConversationMemory(memory_dir=tmpdir)
            assert len(mem._facts) == 0
