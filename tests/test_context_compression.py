"""Tests for context compression."""

import pytest

from forge.llm.context import ContextCompressor, CHARS_PER_TOKEN


class MockClient:
    """Mock OllamaClient for testing compression without Ollama."""

    def generate(self, prompt, system="", timeout=60, temperature=0.1):
        # Return a simple summary
        return {"response": "Summary: The conversation discussed coding tasks and file operations."}


class TestContextCompressor:
    """Tests for context compression."""

    def setup_method(self):
        self.client = MockClient()
        self.compressor = ContextCompressor(
            client=self.client,
            max_tokens=100,  # very small for testing
            keep_recent=3,
        )

    def test_no_compression_needed(self):
        """Short conversations should pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = self.compressor.compress(messages)
        assert result == messages

    def test_needs_compression(self):
        """Should detect when compression is needed."""
        short = [{"role": "user", "content": "Hi"}]
        assert self.compressor.needs_compression(short) is False

        long = [{"role": "user", "content": "x" * 1000} for _ in range(10)]
        assert self.compressor.needs_compression(long) is True

    def test_estimate_tokens(self):
        """Token estimation should be reasonable."""
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = self.compressor.estimate_tokens(messages)
        # "Hello world" = 11 chars + overhead ≈ 6 tokens
        assert 3 < tokens < 20

    def test_sliding_summary_compression(self):
        """Sliding summary should reduce message count."""
        messages = [{"role": "user", "content": f"Message {i} " * 20} for i in range(20)]
        result = self.compressor.compress(messages)
        # Should have fewer messages than original
        assert len(result) < len(messages)
        # Should keep recent messages
        assert len(result) >= self.compressor.keep_recent

    def test_truncate_strategy(self):
        """Truncate should drop oldest messages."""
        compressor = ContextCompressor(
            client=self.client,
            max_tokens=100,
            strategy="truncate",
            keep_recent=3,
        )
        messages = [{"role": "user", "content": f"Message {i} " * 20} for i in range(20)]
        result = compressor.compress(messages)
        # Should have truncation note + recent messages
        assert len(result) <= 5  # note + 3 recent

    def test_progressive_strategy(self):
        """Progressive should filter low-info messages first."""
        compressor = ContextCompressor(
            client=self.client,
            max_tokens=50,
            strategy="progressive",
            keep_recent=3,
        )
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Thanks!"},
            {"role": "user", "content": "Write a function to sort a list" * 5},
            {"role": "assistant", "content": "Here's the code..." * 5},
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "Great!" * 20},
        ]
        result = compressor.compress(messages)
        assert len(result) <= len(messages)

    def test_system_messages_preserved(self):
        """System messages should always be preserved."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Long message " * 50},
            {"role": "assistant", "content": "Long response " * 50},
            {"role": "user", "content": "Another long one " * 50},
            {"role": "assistant", "content": "More text " * 50},
            {"role": "user", "content": "Latest question"},
        ]
        result = self.compressor.compress(messages)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) >= 1

    def test_reset_clears_cache(self):
        """Reset should clear the summary cache."""
        self.compressor._summary_cache = "some cached summary"
        self.compressor._summarized_up_to = 5
        self.compressor.reset()
        assert self.compressor._summary_cache == ""
        assert self.compressor._summarized_up_to == 0
