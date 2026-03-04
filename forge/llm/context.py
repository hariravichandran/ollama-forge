"""Context compression and summarization for long conversations.

Prevents context window overflow by intelligently compressing older messages
while preserving key information (code blocks, decisions, file paths).
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from forge.utils.logging import get_logger

if TYPE_CHECKING:
    from forge.llm.client import OllamaClient

log = get_logger("llm.context")

# Approximate tokens per character (rough heuristic for English text)
CHARS_PER_TOKEN = 3.5

SUMMARY_SYSTEM_PROMPT = """You are a precise conversation summarizer. Create a concise summary that preserves:
- All code blocks and file paths exactly as written
- Key decisions and their reasoning
- Important facts, numbers, and technical details
- Error messages and their resolutions
- Action items and outcomes

Remove:
- Greetings, pleasantries, and filler
- Redundant explanations
- Conversational back-and-forth that doesn't add information

Output a clean, factual summary in bullet-point format. Keep code blocks intact."""


class ContextCompressor:
    """Manages conversation context to fit within token limits.

    Strategies:
    - sliding_summary: Summarize older messages, keep recent ones verbatim
    - truncate: Simply drop oldest messages (fastest, least intelligent)
    - progressive: Multi-pass compression with increasing aggressiveness
    """

    def __init__(
        self,
        client: OllamaClient,
        max_tokens: int = 8192,
        strategy: str = "sliding_summary",
        keep_recent: int = 10,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.keep_recent = keep_recent
        self._summary_cache: str = ""
        self._summarized_up_to: int = 0  # index of last summarized message
        # Compression statistics for observability
        self._compression_stats = {
            "compressions": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "extractive_fallbacks": 0,
        }

    def estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count for a message list.

        Handles None content gracefully — treats as empty string.
        """
        total_chars = sum(len(m.get("content") or "") for m in messages)
        # Add overhead for role markers and formatting
        overhead = len(messages) * 10
        return int((total_chars + overhead) / CHARS_PER_TOKEN)

    def needs_compression(self, messages: list[dict[str, str]]) -> bool:
        """Check if messages exceed the token budget."""
        return self.estimate_tokens(messages) > self.max_tokens * 0.85

    @staticmethod
    def validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate and sanitize messages, fixing common issues.

        - Ensures every message has a 'role' field
        - Replaces None content with empty string
        - Removes messages with no role
        """
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if "role" not in msg:
                continue
            # Ensure content is always a string (never None)
            if msg.get("content") is None:
                msg = dict(msg, content="")
            cleaned.append(msg)
        return cleaned

    def compress(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compress message history to fit within max_tokens.

        Returns a new message list with older messages summarized.
        Validates messages before processing.
        """
        messages = self.validate_messages(messages)
        if not self.needs_compression(messages):
            return messages

        input_tokens = self.estimate_tokens(messages)

        if self.strategy == "truncate":
            result = self._truncate(messages)
        elif self.strategy == "progressive":
            result = self._progressive_compress(messages)
        else:
            result = self._sliding_summary(messages)

        output_tokens = self.estimate_tokens(result)
        self._compression_stats["compressions"] += 1
        self._compression_stats["total_input_tokens"] += input_tokens
        self._compression_stats["total_output_tokens"] += output_tokens

        return result

    def _sliding_summary(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Summarize older messages, keep recent ones verbatim."""
        if len(messages) <= self.keep_recent:
            return messages

        # Separate system messages (always keep)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.keep_recent:
            return messages

        # Split: old messages to summarize, recent to keep
        split_idx = len(non_system) - self.keep_recent
        old_messages = non_system[:split_idx]
        recent_messages = non_system[split_idx:]

        # Only re-summarize if we have new messages to process
        if split_idx > self._summarized_up_to:
            old_text = self._format_messages(old_messages)

            # Extract code blocks before summarization to preserve them
            code_blocks = self._extract_code_blocks(old_text)

            # Include previous summary if it exists
            if self._summary_cache:
                old_text = f"Previous summary:\n{self._summary_cache}\n\nNew messages to incorporate:\n{old_text}"

            summary = self._ask_for_summary(old_text)
            if summary:
                # Re-inject code blocks that may have been lost
                if code_blocks:
                    preserved = "\n\n".join(code_blocks[:5])  # Keep up to 5 most recent
                    summary = f"{summary}\n\n[Preserved code blocks]\n{preserved}"
                self._summary_cache = summary
                self._summarized_up_to = split_idx
                log.info("Compressed %d messages into summary (%d chars, %d code blocks preserved)",
                         len(old_messages), len(summary), len(code_blocks))

        # Build result: system messages + summary + recent
        result = list(system_msgs)
        if self._summary_cache:
            result.append({
                "role": "system",
                "content": f"[Conversation summary]\n{self._summary_cache}",
            })
        result.extend(recent_messages)

        return result

    def _truncate(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Simple truncation: keep system messages and most recent messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep only recent messages
        recent = non_system[-self.keep_recent:]

        result = list(system_msgs)
        if len(non_system) > self.keep_recent:
            result.append({
                "role": "system",
                "content": f"[Note: {len(non_system) - self.keep_recent} earlier messages were truncated]",
            })
        result.extend(recent)
        return result

    def _progressive_compress(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Multi-pass compression with increasing aggressiveness.

        Pass 1: Remove low-information messages (greetings, confirmations)
        Pass 2: Summarize conversation segments
        Pass 3: Extract only key facts and code
        """
        result = list(messages)

        # Pass 1: Filter low-information messages
        low_info_patterns = [
            r"^(ok|okay|thanks|thank you|got it|sure|yes|no|right|hmm|interesting)[\.\!\?]?$",
            r"^(hello|hi|hey|good morning|good afternoon)[\.\!\?]?$",
        ]
        filtered = []
        for msg in result:
            content = msg.get("content", "").strip()
            if msg.get("role") == "system":
                filtered.append(msg)
                continue
            is_low_info = any(re.match(p, content, re.IGNORECASE) for p in low_info_patterns)
            if not is_low_info or len(content) > 50:
                filtered.append(msg)

        result = filtered

        # If still too large, fall back to sliding summary
        if self.estimate_tokens(result) > self.max_tokens * 0.85:
            return self._sliding_summary(result)

        return result

    def _ask_for_summary(self, text: str) -> str:
        """Ask the LLM to summarize a conversation segment.

        Falls back to extractive summarization if the LLM is unavailable
        or returns an error, ensuring compression always succeeds.
        """
        # Truncate input if it's too long for the summary call itself
        max_input_chars = int(self.max_tokens * CHARS_PER_TOKEN * 0.6)
        if len(text) > max_input_chars:
            text = text[:max_input_chars] + "\n[... truncated for summary ...]"

        try:
            result = self.client.generate(
                prompt=f"Summarize this conversation history:\n\n{text}",
                system=SUMMARY_SYSTEM_PROMPT,
                timeout=60,
                temperature=0.1,
            )
            response = result.get("response", "").strip()
            if response and not result.get("error"):
                return response
            log.warning("LLM summarization returned empty/error, using extractive fallback")
        except Exception as e:
            log.warning("LLM summarization failed (%s), using extractive fallback", e)

        # Extractive fallback: keep lines with code, paths, decisions, errors
        self._compression_stats["extractive_fallbacks"] += 1
        return self._extractive_summary(text)

    def _extractive_summary(self, text: str) -> str:
        """Extract key information without LLM — used as fallback.

        Keeps lines containing code blocks, file paths, errors, and decisions.
        """
        important_patterns = [
            r"```",                  # code fences
            r"[\w/]+\.\w{1,5}",     # file paths (e.g., src/main.py)
            r"error|exception|fail|traceback",  # errors
            r"decided|chose|selected|using|switched",  # decisions
            r"todo|fixme|hack|note",  # annotations
            r"https?://",            # URLs
        ]
        important_re = re.compile("|".join(important_patterns), re.IGNORECASE)

        lines = text.splitlines()
        kept: list[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                kept.append(line)
            elif in_code_block:
                kept.append(line)
            elif important_re.search(stripped):
                kept.append(line)

        if not kept:
            # Nothing important found — keep first and last portions
            quarter = max(5, len(lines) // 4)
            kept = lines[:quarter] + ["[... middle portion omitted ...]"] + lines[-quarter:]

        return "\n".join(kept)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages into a readable string for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract fenced code blocks from text.

        Returns a list of code blocks (including the ``` fences) so they can
        be re-injected after summarization to prevent the LLM from mangling them.
        """
        return re.findall(r"```[\s\S]*?```", text)

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics for observability."""
        stats = dict(self._compression_stats)
        if stats["compressions"] > 0:
            total_in = stats["total_input_tokens"]
            total_out = stats["total_output_tokens"]
            stats["avg_compression_ratio"] = round(
                total_out / max(1, total_in), 2
            )
        else:
            stats["avg_compression_ratio"] = 1.0
        return stats

    def reset(self) -> None:
        """Clear cached summaries (e.g., for a new conversation)."""
        self._summary_cache = ""
        self._summarized_up_to = 0
