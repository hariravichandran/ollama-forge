"""Ollama API client with streaming, tool calling, and model management.

Features:
- Non-streaming and streaming chat/generate
- Tool calling (function calling)
- Structured output (JSON schema)
- Multi-modal (image) input
- Prompt caching via keep_alive
- Model management (pull, delete, switch, list)
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import requests

from forge.utils.logging import get_logger

log = get_logger("llm.client")


@dataclass
class LLMStats:
    """Tracks LLM usage statistics."""

    total_calls: int = 0
    total_tokens: int = 0
    total_time_s: float = 0.0
    errors: int = 0

    @property
    def avg_time_s(self) -> float:
        return self.total_time_s / max(1, self.total_calls)

    @property
    def avg_tokens_per_sec(self) -> float:
        return self.total_tokens / max(0.01, self.total_time_s)


class OllamaClient:
    """Client for the Ollama REST API."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        num_thread: int | None = None,
        num_batch: int = 2048,
        keep_alive: str = "30m",
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self.num_batch = num_batch
        self.keep_alive = keep_alive  # Keep model in memory between requests
        self.max_retries = max_retries
        self.stats = LLMStats()

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            return r.status_code == 200
        except (requests.ConnectionError, requests.exceptions.RequestException, OSError):
            return False

    def get_version(self) -> str:
        """Get Ollama server version."""
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            if r.status_code == 200:
                return r.json().get("version", "unknown")
        except requests.ConnectionError:
            pass
        return "unavailable"

    def list_models(self) -> list[dict[str, Any]]:
        """List locally available models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                return r.json().get("models", [])
        except requests.ConnectionError:
            log.error("Cannot connect to Ollama at %s", self.base_url)
        return []

    def list_running(self) -> list[dict[str, Any]]:
        """List models currently loaded in memory."""
        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if r.status_code == 200:
                return r.json().get("models", [])
        except requests.ConnectionError:
            pass
        return []

    def pull_model(self, model: str, progress_cb: Callable[[str], None] | None = None) -> bool:
        """Pull a model from the Ollama registry.

        Args:
            model: Model name (e.g., 'qwen2.5-coder:7b').
            progress_cb: Called with progress string for each update.
                The string includes download percentage when available.

        Returns:
            True if pull succeeded.
        """
        log.info("Pulling model: %s", model)
        try:
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": True},
                stream=True,
                timeout=600,
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")

                    # Format progress with percentage if available
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    if total and completed:
                        pct = int(completed / total * 100)
                        size_gb = total / (1024 ** 3)
                        progress_str = f"{status} — {pct}% of {size_gb:.1f} GB"
                    else:
                        progress_str = status

                    if progress_cb:
                        progress_cb(progress_str)

                    if "error" in data:
                        log.error("Pull error: %s", data["error"])
                        return False
            log.info("Successfully pulled %s", model)
            return True
        except (requests.ConnectionError, requests.Timeout) as e:
            log.error("Failed to pull %s: %s", model, e)
            return False

    def delete_model(self, model: str) -> bool:
        """Delete a locally stored model."""
        try:
            r = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": model},
                timeout=30,
            )
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        json_schema: dict | None = None,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a completion (non-streaming).

        Args:
            prompt: The prompt text.
            system: Optional system prompt.
            json_mode: If True, forces JSON output.
            json_schema: If provided, forces output to match the given
                JSON schema (Ollama structured output). Takes precedence
                over json_mode.
            timeout: Request timeout in seconds.
            temperature: Override temperature for this call.
            model: Override model for this call.

        Returns dict with keys: response, tokens, time_s, tokens_per_sec
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if system:
            payload["system"] = system
        if json_schema:
            # Ollama structured output: pass the full JSON schema as format
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"

        last_error = ""
        for attempt in range(1 + self.max_retries):
            start = time.time()
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                elapsed = time.time() - start

                if r.status_code != 200:
                    self.stats.errors += 1
                    last_error = r.text
                    if attempt < self.max_retries:
                        log.warning("Generate failed (attempt %d), retrying: %s", attempt + 1, r.text[:100])
                        time.sleep(1)
                        continue
                    return {"response": "", "tokens": 0, "time_s": elapsed, "tokens_per_sec": 0, "error": r.text}

                data = r.json()
                tokens = data.get("eval_count", 0)
                self.stats.total_calls += 1
                self.stats.total_tokens += tokens
                self.stats.total_time_s += elapsed

                return {
                    "response": data.get("response", ""),
                    "tokens": tokens,
                    "time_s": elapsed,
                    "tokens_per_sec": tokens / max(0.01, elapsed),
                }
            except requests.Timeout:
                self.stats.errors += 1
                last_error = "timeout"
                if attempt < self.max_retries:
                    log.warning("Generate timed out (attempt %d), retrying", attempt + 1)
                    continue
                return {"response": "", "tokens": 0, "time_s": time.time() - start, "tokens_per_sec": 0, "error": "timeout"}
            except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
                self.stats.errors += 1
                last_error = str(e)
                if attempt < self.max_retries:
                    log.warning("Generate error (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                return {"response": "", "tokens": 0, "time_s": 0, "tokens_per_sec": 0, "error": str(e)}

        return {"response": "", "tokens": 0, "time_s": 0, "tokens_per_sec": 0, "error": last_error}

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict] | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
        images: list[str] | None = None,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Chat completion (non-streaming).

        Args:
            messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
            tools: Ollama tool definitions for function calling
            json_mode: If True, forces JSON output
            json_schema: If provided, forces output to match the JSON schema
            images: List of image paths or base64 strings for vision models.
                    Paths are auto-converted to base64.
            timeout: Request timeout in seconds
            temperature: Override temperature for this call
            model: Override model for this call

        Returns dict with: response, tokens, time_s, tool_calls (if any)
        """
        # If images provided, add them to the last user message
        if images:
            messages = self._inject_images(messages, images)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if tools:
            payload["tools"] = tools
        if json_schema:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"

        last_error = ""
        for attempt in range(1 + self.max_retries):
            start = time.time()
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                elapsed = time.time() - start

                if r.status_code != 200:
                    self.stats.errors += 1
                    last_error = r.text
                    if attempt < self.max_retries:
                        log.warning("Chat failed (attempt %d), retrying: %s", attempt + 1, r.text[:100])
                        time.sleep(1)
                        continue
                    return {"response": "", "tokens": 0, "time_s": elapsed, "error": r.text}

                data = r.json()
                msg = data.get("message", {})
                tokens = data.get("eval_count", 0)

                self.stats.total_calls += 1
                self.stats.total_tokens += tokens
                self.stats.total_time_s += elapsed

                result: dict[str, Any] = {
                    "response": msg.get("content", ""),
                    "tokens": tokens,
                    "time_s": elapsed,
                    "tokens_per_sec": tokens / max(0.01, elapsed),
                }
                if msg.get("tool_calls"):
                    result["tool_calls"] = msg["tool_calls"]

                return result
            except requests.Timeout:
                self.stats.errors += 1
                last_error = "timeout"
                if attempt < self.max_retries:
                    log.warning("Chat timed out (attempt %d), retrying", attempt + 1)
                    continue
                return {"response": "", "tokens": 0, "time_s": time.time() - start, "error": "timeout"}
            except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
                self.stats.errors += 1
                last_error = str(e)
                if attempt < self.max_retries:
                    log.warning("Chat error (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                return {"response": "", "tokens": 0, "time_s": 0, "error": str(e)}

        return {"response": "", "tokens": 0, "time_s": 0, "error": last_error}

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        timeout: int = 300,
        model: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat — yields event dicts as they arrive.

        Each yielded dict has a "type" key:
        - {"type": "text", "content": "..."} — text token
        - {"type": "tool_call", "tool_calls": [...]} — tool call request
        - {"type": "done", "tokens": N, "time_s": N} — generation complete
        - {"type": "error", "error": "..."} — error
        """
        if images:
            messages = self._inject_images(messages, images)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if tools:
            payload["tools"] = tools

        for attempt in range(1 + self.max_retries):
            start = time.time()
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=timeout,
                )
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        msg = data.get("message", {})

                        # Text content
                        content = msg.get("content", "")
                        if content:
                            yield {"type": "text", "content": content}

                        # Tool calls
                        if msg.get("tool_calls"):
                            yield {"type": "tool_call", "tool_calls": msg["tool_calls"]}

                        # Done signal
                        if data.get("done"):
                            total_tokens = data.get("eval_count", 0)
                            elapsed = time.time() - start
                            self.stats.total_calls += 1
                            self.stats.total_tokens += total_tokens
                            self.stats.total_time_s += elapsed
                            yield {
                                "type": "done",
                                "tokens": total_tokens,
                                "time_s": round(elapsed, 2),
                                "tokens_per_sec": round(total_tokens / max(0.01, elapsed), 1),
                            }
                return  # Stream completed successfully
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                self.stats.errors += 1
                if attempt < self.max_retries:
                    log.warning("Stream chat error (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(1)
                    continue
                yield {"type": "error", "error": str(e)}

    def show_model(self, model: str | None = None) -> dict[str, Any]:
        """Get model details (parameters, template, capabilities).

        Useful for checking if a model supports vision, tools, etc.
        """
        try:
            r = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model or self.model},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
        except (requests.ConnectionError, requests.Timeout):
            pass
        return {}

    def warmup(self, system: str = "") -> bool:
        """Pre-warm the KV cache by sending the system prompt with no generation.

        This pre-computes the KV cache for the system prompt so subsequent
        requests that share the same prefix are faster.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": system or "You are a helpful assistant.",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": 1,  # Generate minimal tokens — just warm the cache
                "num_ctx": self.num_ctx,
            },
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if r.status_code == 200:
                log.info("Warmed KV cache for %s", self.model)
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        return False

    def _inject_images(
        self, messages: list[dict[str, Any]], images: list[str],
    ) -> list[dict[str, Any]]:
        """Inject base64-encoded images into the last user message.

        Accepts file paths or raw base64 strings.
        """
        encoded = []
        for img in images:
            if Path(img).exists():
                # File path — read and encode
                encoded.append(base64.b64encode(Path(img).read_bytes()).decode("utf-8"))
            else:
                # Assume already base64
                encoded.append(img)

        if not encoded:
            return messages

        # Find last user message and add images
        messages = [dict(m) for m in messages]  # shallow copy
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["images"] = encoded
                break

        return messages

    def switch_model(self, model: str) -> bool:
        """Switch to a different model, pulling it if necessary."""
        available = [m.get("name", "") for m in self.list_models()]
        # Normalize names (strip :latest)
        available_base = [n.split(":")[0] for n in available]
        model_base = model.split(":")[0]

        if model not in available and model_base not in available_base:
            log.info("Model %s not found locally, pulling...", model)
            if not self.pull_model(model):
                return False

        self.model = model
        log.info("Switched to model: %s", model)
        return True
