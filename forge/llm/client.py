"""Ollama API client with streaming, tool calling, and model management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self.num_batch = num_batch
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
        """Pull a model from the Ollama registry."""
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
                    if progress_cb:
                        progress_cb(status)
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
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a completion (non-streaming).

        Returns dict with keys: response, tokens, time_s, tokens_per_sec
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
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
        if json_mode:
            payload["format"] = "json"

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
            return {"response": "", "tokens": 0, "time_s": time.time() - start, "tokens_per_sec": 0, "error": "timeout"}
        except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
            self.stats.errors += 1
            return {"response": "", "tokens": 0, "time_s": 0, "tokens_per_sec": 0, "error": str(e)}

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict] | None = None,
        json_mode: bool = False,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Chat completion (non-streaming).

        messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
        tools: Ollama tool definitions for function calling

        Returns dict with: response, tokens, time_s, tool_calls (if any)
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
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
        if json_mode:
            payload["format"] = "json"

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
            return {"response": "", "tokens": 0, "time_s": time.time() - start, "error": "timeout"}
        except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
            self.stats.errors += 1
            return {"response": "", "tokens": 0, "time_s": 0, "error": str(e)}

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        timeout: int = 300,
        model: str | None = None,
    ) -> Generator[str, None, None]:
        """Streaming chat — yields text chunks as they arrive."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread

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
                    content = msg.get("content", "")
                    if content:
                        yield content
        except (requests.ConnectionError, requests.Timeout) as e:
            yield f"\n[Error: {e}]"

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
