"""OpenAI-compatible API endpoint for ollama-forge.

Provides a drop-in replacement for the OpenAI API, allowing tools like
Continue.dev, LiteLLM, and other OpenAI-compatible clients to use
ollama-forge's agent system.

Endpoints:
- POST /v1/chat/completions — Chat completions (streaming and non-streaming)
- POST /v1/completions      — FIM completions (for tab autocomplete)
- GET  /v1/models           — List available models
- GET  /health              — Health check

Usage:
    forge api                    # Start API server on port 8000
    forge api --port 9000        # Custom port

    # In your client, set base_url to http://localhost:8000/v1
    # Compatible with Continue.dev, Cursor, and any OpenAI SDK client
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

log = logging.getLogger("forge.api.openai_compat")

# API limits
MAX_MESSAGES = 200  # max messages in a single chat request
MAX_MAX_TOKENS = 32768  # hard cap on max_tokens
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_N_COMPLETIONS = 1  # only 1 completion supported (Ollama limitation)
FIM_DEFAULT_MAX_TOKENS = 256
FIM_TIMEOUT = 30  # seconds for FIM requests


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

    class CompletionRequest(BaseModel):
        """FIM (Fill-in-the-Middle) completion request.

        Used by tab autocomplete clients. The prompt contains the code
        before the cursor, and suffix contains code after the cursor.
        """
        model: str = ""
        prompt: str
        suffix: str = ""
        max_tokens: int = 256
        temperature: float = 0.2
        stream: bool = False
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

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        """FIM (Fill-in-the-Middle) completion endpoint.

        Used by tab autocomplete clients (Continue.dev, Cursor, etc.).
        Sends prefix and suffix to the model for infill completion.
        """
        client = get_client()

        if not client.is_available():
            raise HTTPException(503, "Ollama is not running")

        if request.model and request.model != client.model:
            client.switch_model(request.model)

        if request.stream:
            return StreamingResponse(
                _stream_fim_response(client, request),
                media_type="text/event-stream",
            )

        # Non-streaming FIM
        result = _generate_fim(client, request)
        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": client.model,
            "choices": [
                {
                    "index": 0,
                    "text": result.get("response", ""),
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result.get("tokens", 0),
                "total_tokens": result.get("tokens", 0),
            },
        }

    # ─── Helpers ────────────────────────────────────────────────────────

    def _stream_response(client, messages, request):
        """Generate SSE stream in OpenAI format."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        for event in client.stream_chat(messages):
            if event.get("type") == "text":
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": client.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event["content"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
            elif event.get("type") == "done":
                break

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

    def _generate_fim(client, request):
        """Generate FIM completion using Ollama's suffix parameter."""
        import requests as req

        payload: dict[str, Any] = {
            "model": client.model,
            "prompt": request.prompt,
            "stream": False,
            "keep_alive": client.keep_alive,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": client.num_ctx,
            },
        }
        if request.suffix:
            payload["suffix"] = request.suffix
        if request.stop:
            stops = request.stop if isinstance(request.stop, list) else [request.stop]
            payload["options"]["stop"] = stops

        try:
            r = req.post(
                f"{client.base_url}/api/generate",
                json=payload,
                timeout=FIM_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning("FIM generation failed: %s", e)
        return {"response": "", "tokens": 0}

    def _stream_fim_response(client, request):
        """Stream FIM completion in OpenAI format."""
        import requests as req

        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        payload: dict[str, Any] = {
            "model": client.model,
            "prompt": request.prompt,
            "stream": True,
            "keep_alive": client.keep_alive,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": client.num_ctx,
            },
        }
        if request.suffix:
            payload["suffix"] = request.suffix

        try:
            r = req.post(
                f"{client.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=FIM_TIMEOUT,
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    text = data.get("response", "")
                    if text:
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": client.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": text,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    if data.get("done"):
                        break
        except Exception as e:
            log.warning("FIM streaming error: %s", e)

        # Final chunk
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": client.model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
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
    print(f"FIM autocomplete endpoint: http://{host}:{port}/v1/completions")
    uvicorn.run(app, host=host, port=port)
