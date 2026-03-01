"""Web UI using FastAPI + htmx — Phase 2 implementation.

Provides a browser-based interface with:
- Chat interface with streaming responses
- Model selector dropdown
- MCP toggle switches
- Agent switcher
- Hardware status panel
- Settings page

Install with: pip install ollama-forge[web]
"""

from __future__ import annotations


def launch_web_ui(port: int = 8080):
    """Launch the Web UI."""
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Web UI requires fastapi and uvicorn. Install with:")
        print("  pip install ollama-forge[web]")
        return

    # Placeholder — full Web UI implementation in Phase 2
    print(f"Web UI coming in Phase 2.")
    print(f"For now, use: forge chat")
