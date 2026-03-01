"""Session persistence: save, load, and resume chat sessions.

Allows users to save their work and resume later exactly where they left off.
Sessions include the full conversation history, agent state, and metadata.

Storage: ~/.config/ollama-forge/sessions/

Usage:
    sessions = SessionManager()

    # Save current session
    session_id = sessions.save(messages, agent_name="coder", metadata={...})

    # List saved sessions
    for s in sessions.list_sessions():
        print(f"{s.session_id}: {s.title} ({s.message_count} messages)")

    # Load and resume
    session = sessions.load(session_id)
    messages = session.messages
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

log = get_logger("agents.sessions")

SESSIONS_DIR = Path.home() / ".config" / "ollama-forge" / "sessions"


@dataclass
class Session:
    """A saved chat session."""

    session_id: str
    title: str
    agent_name: str
    model: str
    messages: list[dict[str, str]]
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def duration_s(self) -> float:
        return self.updated_at - self.created_at

    def summary(self) -> str:
        """One-line summary for listing."""
        age = time.time() - self.updated_at
        if age < 3600:
            age_str = f"{int(age / 60)}m ago"
        elif age < 86400:
            age_str = f"{int(age / 3600)}h ago"
        else:
            age_str = f"{int(age / 86400)}d ago"
        return f"{self.session_id[:8]} | {self.title} | {self.agent_name} | {self.message_count} msgs | {age_str}"


@dataclass
class SessionSummary:
    """Lightweight session info for listing (without full messages)."""

    session_id: str
    title: str
    agent_name: str
    model: str
    message_count: int
    created_at: float
    updated_at: float


class SessionManager:
    """Manages persistent chat sessions."""

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        messages: list[dict[str, str]],
        agent_name: str = "assistant",
        model: str = "",
        title: str = "",
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a chat session.

        Args:
            messages: Conversation message history.
            agent_name: Name of the active agent.
            model: Model used in this session.
            title: Session title (auto-generated if empty).
            session_id: Existing session ID to update, or empty for new.
            metadata: Additional metadata to store.

        Returns:
            Session ID.
        """
        now = time.time()

        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:8]}"

        if not title:
            title = self._generate_title(messages)

        session = Session(
            session_id=session_id,
            title=title,
            agent_name=agent_name,
            model=model,
            messages=messages,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        # Check if updating existing session
        existing_path = self.sessions_dir / f"{session_id}.json"
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text())
                session.created_at = existing.get("created_at", now)
            except Exception:
                pass

        session.updated_at = now

        # Write to disk
        session_path = self.sessions_dir / f"{session_id}.json"
        session_path.write_text(json.dumps(asdict(session), indent=2))
        log.info("Saved session %s (%d messages)", session_id, len(messages))
        return session_id

    def load(self, session_id: str) -> Session | None:
        """Load a saved session by ID.

        Supports partial ID matching (first 8 chars).
        """
        # Try exact match first
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            # Try partial match
            matches = list(self.sessions_dir.glob(f"{session_id}*.json"))
            if len(matches) == 1:
                session_path = matches[0]
            elif len(matches) > 1:
                log.warning("Ambiguous session ID: %s (matches %d sessions)", session_id, len(matches))
                return None
            else:
                return None

        try:
            data = json.loads(session_path.read_text())
            return Session(**data)
        except Exception as e:
            log.error("Failed to load session %s: %s", session_id, e)
            return None

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """List saved sessions, most recent first."""
        summaries = []

        for session_file in self.sessions_dir.glob("session-*.json"):
            try:
                data = json.loads(session_file.read_text())
                summaries.append(SessionSummary(
                    session_id=data["session_id"],
                    title=data.get("title", "Untitled"),
                    agent_name=data.get("agent_name", "assistant"),
                    model=data.get("model", ""),
                    message_count=len(data.get("messages", [])),
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                ))
            except Exception:
                continue

        # Sort by updated_at descending
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            log.info("Deleted session %s", session_id)
            return True

        # Try partial match
        matches = list(self.sessions_dir.glob(f"{session_id}*.json"))
        if len(matches) == 1:
            matches[0].unlink()
            return True

        return False

    def export(self, session_id: str, format: str = "markdown") -> str:
        """Export a session as markdown or JSON.

        Args:
            session_id: Session to export.
            format: "markdown" or "json".

        Returns:
            Formatted string.
        """
        session = self.load(session_id)
        if not session:
            return f"Session not found: {session_id}"

        if format == "json":
            return json.dumps(asdict(session), indent=2)

        # Markdown format
        lines = [
            f"# Chat Session: {session.title}",
            f"",
            f"- **Agent**: {session.agent_name}",
            f"- **Model**: {session.model}",
            f"- **Messages**: {session.message_count}",
            f"",
            "---",
            "",
        ]

        for msg in session.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"*System: {content[:100]}...*")
            elif role == "user":
                lines.append(f"**User:** {content}")
            elif role == "assistant":
                lines.append(f"**Assistant:** {content}")
            lines.append("")

        return "\n".join(lines)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search across all sessions for messages matching a query.

        Args:
            query: Text to search for (case-insensitive).
            limit: Maximum results to return.

        Returns:
            List of dicts with session_id, title, role, content, match_line.
        """
        query_lower = query.lower()
        results = []

        for session_file in self.sessions_dir.glob("session-*.json"):
            try:
                data = json.loads(session_file.read_text())
                session_id = data.get("session_id", "")
                title = data.get("title", "Untitled")

                for msg in data.get("messages", []):
                    content = msg.get("content", "")
                    if query_lower in content.lower():
                        # Extract matching line
                        for line in content.splitlines():
                            if query_lower in line.lower():
                                results.append({
                                    "session_id": session_id,
                                    "title": title,
                                    "role": msg.get("role", ""),
                                    "content": line.strip()[:200],
                                })
                                if len(results) >= limit:
                                    return results
                                break
            except Exception:
                continue

        return results

    def _generate_title(self, messages: list[dict[str, str]]) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                # Take first line, limit to 50 chars
                first_line = content.split("\n")[0][:50]
                return first_line if first_line else "Untitled"
        return "Untitled"
