# Context Compression

## The Problem

Local LLMs have limited context windows (4K-32K tokens typically). Long conversations quickly exceed these limits, causing either:
- Silent truncation (losing important context)
- Errors or degraded quality

## The Solution

ollama-forge provides intelligent context compression that summarizes older messages while preserving key information.

## Strategies

### 1. Sliding Summary (Default)

When the conversation exceeds 85% of the token budget:
1. Split messages into "old" and "recent"
2. Ask the LLM to summarize the old messages
3. Replace old messages with a compact summary
4. Keep recent messages verbatim

The summary preserves:
- All code blocks exactly as written
- File paths and technical details
- Key decisions and their reasoning
- Error messages and resolutions

The summary removes:
- Greetings and pleasantries
- Redundant explanations
- Conversational filler

### 2. Progressive Compression

Multi-pass compression with increasing aggressiveness:
1. **Pass 1**: Remove low-information messages (greetings, confirmations like "ok", "thanks")
2. **Pass 2**: If still too large, apply sliding summary

### 3. Truncate

Simple sliding window — drops oldest messages. Fastest but least intelligent. Use when you need maximum speed.

## Configuration

```python
# In your agent YAML
max_context: 8192  # token budget for compression

# Or via code
from forge.llm.context import ContextCompressor

compressor = ContextCompressor(
    client=client,
    max_tokens=8192,
    strategy="sliding_summary",  # or "progressive", "truncate"
    keep_recent=10,              # always keep last N messages
)
```

## How It Works

```
Messages: [sys, user1, asst1, user2, asst2, ..., user20, asst20]

Estimated tokens: 12,000 (exceeds 8,192 budget)

After compression:
[sys, summary_of_1_to_10, user11, asst11, ..., user20, asst20]

Summary preserves: code blocks, decisions, file paths
Summary removes: greetings, filler, redundant explanations
```

## Token Estimation

ollama-forge uses a heuristic of ~3.5 characters per token for English text. This is approximate but sufficient for compression decisions. The actual tokenizer varies by model, but the heuristic works well enough to prevent context overflow.
