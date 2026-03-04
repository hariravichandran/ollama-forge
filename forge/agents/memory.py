"""Conversation memory: persistent storage across sessions.

Saves conversation history and key facts to disk so agents can remember
context from previous sessions. This is one of the most requested features
for local AI tools.

Storage structure:
    ~/.config/ollama-forge/memory/
    ├── conversations/          # Full conversation logs (JSONL)
    │   ├── 2026-03-01_abc123.jsonl
    │   └── ...
    ├── facts.jsonl             # Extracted key facts (persistent memory)
    └── summary.md              # Latest conversation summary
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("agents.memory")

DEFAULT_MEMORY_DIR = Path.home() / ".config" / "ollama-forge" / "memory"


@dataclass
class MemoryFact:
    """A persistent fact extracted from conversations."""

    key: str          # short identifier (e.g., "preferred_language")
    value: str        # the fact content
    source: str       # which conversation it came from
    timestamp: float  # when it was recorded
    confidence: float = 1.0  # how confident we are (0-1)


class ConversationMemory:
    """Persists conversation history and key facts across sessions.

    Usage:
        memory = ConversationMemory()

        # Save current conversation
        memory.save_conversation(messages, session_id="abc123")

        # Load recent conversations for context
        recent = memory.get_recent_context(max_messages=20)

        # Store/retrieve specific facts
        memory.store_fact("user_name", "Alice")
        name = memory.get_fact("user_name")

        # Get all facts as context string
        context = memory.get_facts_context()
    """

    def __init__(self, memory_dir: str | Path | None = None):
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.conversations_dir = self.memory_dir / "conversations"
        self.facts_file = self.memory_dir / "facts.jsonl"
        self.summary_file = self.memory_dir / "summary.md"

        # Create directories
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)

        # Load facts into memory
        self._facts: dict[str, MemoryFact] = self._load_facts()

    def save_conversation(
        self,
        messages: list[dict[str, str]],
        session_id: str = "",
    ) -> str:
        """Save a conversation to disk.

        Returns the path to the saved file.
        """
        if not messages:
            return ""

        if not session_id:
            session_id = f"{int(time.time())}"

        date_str = time.strftime("%Y-%m-%d")
        filename = f"{date_str}_{session_id}.jsonl"
        filepath = self.conversations_dir / filename

        lines = [json.dumps(msg) for msg in messages]
        filepath.write_text("\n".join(lines) + "\n")

        log.info("Saved conversation: %s (%d messages)", filename, len(messages))
        return str(filepath)

    def get_recent_context(self, max_messages: int = 20) -> list[dict[str, str]]:
        """Load recent messages from the last conversation for context.

        Returns the most recent messages that fit within max_messages.
        """
        # Find the most recent conversation file
        conv_files = sorted(self.conversations_dir.glob("*.jsonl"), reverse=True)
        if not conv_files:
            return []

        messages = []
        for conv_file in conv_files[:3]:  # check last 3 sessions max
            try:
                for line in conv_file.read_text().splitlines():
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                continue

            if len(messages) >= max_messages:
                break

        # Return the most recent messages
        return messages[-max_messages:]

    def store_fact(
        self,
        key: str,
        value: str,
        source: str = "conversation",
        confidence: float = 1.0,
    ) -> None:
        """Store a persistent fact, deduplicating near-identical values.

        If a fact with the same key exists and the value is similar
        (>80% match), it updates the existing fact instead of adding
        a duplicate.
        """
        # Check for near-duplicate values under different keys
        for existing_key, existing_fact in self._facts.items():
            if existing_key != key and self._is_similar(existing_fact.value, value):
                log.debug(
                    "Dedup: new fact '%s' is similar to existing '%s', updating existing",
                    key, existing_key,
                )
                # Update existing fact with newer value and higher confidence
                existing_fact.value = value
                existing_fact.timestamp = time.time()
                existing_fact.confidence = max(existing_fact.confidence, confidence)
                self._save_facts()
                return

        fact = MemoryFact(
            key=key,
            value=value,
            source=source,
            timestamp=time.time(),
            confidence=confidence,
        )
        self._facts[key] = fact
        self._save_facts()
        log.debug("Stored fact: %s = %s", key, value[:50])

    @staticmethod
    def _is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar using a simple ratio.

        Uses sequence matching ratio — fast enough for fact comparison
        without requiring external dependencies.
        """
        if a == b:
            return True
        if not a or not b:
            return False

        # Quick length check — very different lengths are unlikely similar
        if min(len(a), len(b)) / max(len(a), len(b)) < 0.5:
            return False

        # Compute similarity ratio using longest common subsequence
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

    def get_fact(self, key: str) -> str | None:
        """Retrieve a stored fact by key."""
        fact = self._facts.get(key)
        return fact.value if fact else None

    def get_facts_context(self) -> str:
        """Format all stored facts as a context string for the system prompt."""
        if not self._facts:
            return ""

        lines = ["Known facts from previous sessions:"]
        for fact in sorted(self._facts.values(), key=lambda f: f.timestamp, reverse=True):
            lines.append(f"- {fact.key}: {fact.value}")

        return "\n".join(lines)

    def save_summary(self, summary: str) -> None:
        """Save a conversation summary for quick context loading."""
        self.summary_file.write_text(summary)

    def get_summary(self) -> str:
        """Load the latest conversation summary."""
        if self.summary_file.exists():
            return self.summary_file.read_text().strip()
        return ""

    def clear(self) -> None:
        """Clear all memory (conversations, facts, summary)."""
        for f in self.conversations_dir.glob("*.jsonl"):
            f.unlink()
        if self.facts_file.exists():
            self.facts_file.unlink()
        if self.summary_file.exists():
            self.summary_file.unlink()
        self._facts.clear()
        log.info("Memory cleared")

    def _load_facts(self) -> dict[str, MemoryFact]:
        """Load facts from disk."""
        if not self.facts_file.exists():
            return {}

        facts = {}
        for line in self.facts_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fact = MemoryFact(**data)
                facts[fact.key] = fact
            except (json.JSONDecodeError, TypeError):
                continue
        return facts

    def _save_facts(self) -> None:
        """Save facts to disk."""
        lines = [json.dumps(asdict(fact)) for fact in self._facts.values()]
        self.facts_file.write_text("\n".join(lines) + "\n" if lines else "")
