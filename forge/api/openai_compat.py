"""OpenAI-compatible API endpoint for ollama-forge.

Provides a drop-in replacement for the OpenAI API, allowing tools like
Continue.dev, LiteLLM, and other OpenAI-compatible clients to use
ollama-forge's agent system.

Endpoints:
- POST /v1/chat/completions — Chat completions (streaming and non-streaming)
- GET  /v1/models           — List available models
- GET  /health              — Health check

Usage:
    forge api                    # Start API server on port 8000
    forge api --port 9000        # Custom port

    # In your client, set base_url to http://localhost:8000/v1
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def create_app():
    """Create the FastAPI app for the OpenAI-compatible API."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "API server requires fastapi and uvicorn. Install with:\n"
            "  pip install fastapi uvicorn"
        )

    from forge.llm.client import OllamaClient

    app = FastAPI(
        title="ollama-forge API",
        description="OpenAI-compatible API powered by local Ollama models",
        version="0.1.0",
    )

    # Lazy-initialized client
    _client: OllamaClient | None = None

    def get_client() -> OllamaClient:
        nonlocal _client
        if _client is None:
            from forge.config import load_config
            from forge.hardware import detect_hardware, select_profile
            config = load_config()
            hw = detect_hardware()
            profile = select_profile(hw)
            _client = OllamaClient(
                model=config.default_model or profile.recommended_model,
                num_ctx=profile.num_ctx,
                num_thread=profile.max_threads,
            )
        return _client

    # ─── Request/Response Models ────────────────────────────────────────

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = ""
        messages: list[ChatMessage]
        temperature: float = 0.7
        max_tokens: int | None = None
        stream: bool = False
        top_p: float = 1.0
        n: int = 1
        stop: list[str] | str | None = None

    # ─── Endpoints ──────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        client = get_client()
        available = client.is_available()
        return {
            "status": "ok" if available else "ollama_unavailable",
            "model": client.model,
        }

    @app.get("/v1/models")
    def list_models():
        client = get_client()
        models = client.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m.get("name", ""),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                }
                for m in models
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        client = get_client()

        if not client.is_available():
            raise HTTPException(503, "Ollama is not running")

        # Switch model if requested
        if request.model and request.model != client.model:
            client.switch_model(request.model)

        # Convert messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            return StreamingResponse(
                _stream_response(client, messages, request),
                media_type="text/event-stream",
            )

        # Non-streaming response
        result = client.chat(
            messages=messages,
            temperature=request.temperature,
        )

        response_text = result.get("response", "")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": client.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("tokens", 0),
            },
        }

    def _stream_response(client, messages, request):
        """Generate SSE stream in OpenAI format."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        for chunk in client.stream_chat(messages):
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": client.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Final chunk
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": client.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return app


def run_api_server(port: int = 8000, host: str = "127.0.0.1"):
    """Start the API server."""
    try:
        import uvicorn
    except ImportError:
        print("API server requires uvicorn. Install with: pip install uvicorn")
        return

    app = create_app()
    print(f"Starting ollama-forge API at http://{host}:{port}")
    print(f"OpenAI-compatible endpoint: http://{host}:{port}/v1/chat/completions")
    uvicorn.run(app, host=host, port=port)
