"""Terminal UI using Textual — Phase 2 implementation.

Provides a rich TUI with:
- Chat panel with markdown rendering
- Model picker sidebar
- Status bar (hardware, model loaded, MCP status)
- Agent switcher tabs
- File tree browser (for coding agent)

Install with: pip install ollama-forge[tui]
"""

from __future__ import annotations


def launch_tui():
    """Launch the Terminal UI."""
    try:
        from textual.app import App
    except ImportError:
        print("Terminal UI requires textual. Install with:")
        print("  pip install ollama-forge[tui]")
        return

    # Placeholder — full TUI implementation in Phase 2
    print("Terminal UI coming in Phase 2.")
    print("For now, use: forge chat")
