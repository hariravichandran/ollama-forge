"""Web UI using FastAPI + htmx — browser-based chat interface for ollama-forge.

Provides:
- Chat interface with streaming responses (SSE)
- Model selector dropdown
- Agent switcher
- Hardware status panel
- MCP status

Install with: pip install ollama-forge[web]

Usage:
    forge ui
    forge ui --port 8080
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

UI_DIR = Path(__file__).parent


def launch_web_ui(
    port: int = 8080,
    host: str = "127.0.0.1",
    model: str = "",
    agent: str = "assistant",
    working_dir: str = ".",
):
    """Launch the Web UI."""
    try:
        import uvicorn
    except ImportError:
        print("Web UI requires fastapi and uvicorn. Install with:")
        print("  pip install ollama-forge[web]")
        return

    app = create_web_app(model=model, agent=agent, working_dir=working_dir)
    print(f"ollama-forge Web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def create_web_app(
    model: str = "",
    agent: str = "assistant",
    working_dir: str = ".",
):
    """Create the FastAPI web application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    app = FastAPI(title="ollama-forge Web UI", version="0.1.0")

    # Static files and templates
    static_dir = UI_DIR / "static"
    templates_dir = UI_DIR / "templates"
    static_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # App state
    _state: dict[str, Any] = {
        "client": None,
        "orchestrator": None,
        "hw": None,
        "profile": None,
        "model": model,
        "agent": agent,
        "working_dir": working_dir,
        "initialized": False,
    }

    def _ensure_init():
        """Lazy-initialize backend on first request."""
        if _state["initialized"]:
            return

        from forge.config import load_config
        from forge.hardware import detect_hardware, select_profile
        from forge.hardware.rocm import configure_rocm_env
        from forge.llm.client import OllamaClient
        from forge.agents.orchestrator import AgentOrchestrator

        config = load_config()
        hw = detect_hardware()
        profile = select_profile(hw)
        configure_rocm_env(hw.gpu)

        model_name = model or config.default_model or profile.recommended_model

        client = OllamaClient(
            model=model_name,
            base_url=config.ollama_base_url,
            num_ctx=profile.num_ctx,
            num_thread=profile.max_threads,
            num_batch=profile.num_batch,
        )

        orchestrator = AgentOrchestrator(client=client, working_dir=working_dir)
        if agent != "assistant":
            orchestrator.switch_agent(agent)

        _state.update({
            "client": client,
            "orchestrator": orchestrator,
            "hw": hw,
            "profile": profile,
            "model": model_name,
            "initialized": True,
        })

    # ─── Routes ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main chat page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/status")
    async def status():
        """Get current status (hardware, model, agent)."""
        _ensure_init()
        hw = _state["hw"]
        profile = _state["profile"]
        client = _state["client"]

        available_models = []
        if client and client.is_available():
            available_models = [m.get("name", "") for m in client.list_models()]

        return {
            "model": _state["model"],
            "agent": _state["agent"],
            "profile": profile.name if profile else "unknown",
            "gpu": hw.gpu.name if hw else "unknown",
            "ram_gb": round(hw.ram_gb, 1) if hw else 0,
            "available_models": available_models,
            "agents": ["assistant", "coder", "researcher"],
            "ollama_available": client.is_available() if client else False,
        }

    @app.post("/api/chat")
    async def chat(request: Request):
        """Send a chat message (non-streaming)."""
        _ensure_init()
        data = await request.json()
        message = data.get("message", "").strip()

        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        orchestrator = _state["orchestrator"]
        if not orchestrator:
            return JSONResponse({"error": "Not initialized"}, status_code=503)

        response = orchestrator.chat(message)
        return {"response": response}

    @app.post("/api/chat/stream")
    async def chat_stream(request: Request):
        """Send a chat message with streaming response (SSE)."""
        _ensure_init()
        data = await request.json()
        message = data.get("message", "").strip()

        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        orchestrator = _state["orchestrator"]
        if not orchestrator:
            return JSONResponse({"error": "Not initialized"}, status_code=503)

        def generate():
            try:
                response = orchestrator.chat(message)
                # Send the complete response as a single SSE event
                yield f"data: {json.dumps({'content': response, 'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/model")
    async def switch_model(request: Request):
        """Switch to a different model."""
        _ensure_init()
        data = await request.json()
        model_name = data.get("model", "")

        client = _state["client"]
        if client and client.switch_model(model_name):
            _state["model"] = model_name
            return {"success": True, "model": model_name}
        return JSONResponse({"error": f"Failed to switch to {model_name}"}, status_code=400)

    @app.post("/api/agent")
    async def switch_agent(request: Request):
        """Switch to a different agent."""
        _ensure_init()
        data = await request.json()
        agent_name = data.get("agent", "")

        orchestrator = _state["orchestrator"]
        if orchestrator:
            result = orchestrator.switch_agent(agent_name)
            _state["agent"] = agent_name
            return {"success": True, "agent": agent_name, "message": result}
        return JSONResponse({"error": "Not initialized"}, status_code=503)

    @app.post("/api/reset")
    async def reset():
        """Reset the chat history."""
        _ensure_init()
        orchestrator = _state["orchestrator"]
        if orchestrator:
            current = orchestrator.agents.get(orchestrator.active_agent)
            if current:
                current.reset()
        return {"success": True}

    @app.get("/api/stats")
    async def stats():
        """Get usage statistics."""
        _ensure_init()
        orchestrator = _state["orchestrator"]
        if orchestrator:
            return orchestrator.get_all_stats()
        return {}

    return app
