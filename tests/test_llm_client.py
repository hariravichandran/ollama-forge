"""Tests for the Ollama LLM client."""

import pytest

from forge.llm.client import OllamaClient, LLMStats
from forge.llm.models import estimate_model_size, get_models_for_category, get_models_that_fit, MODEL_CATALOGUE


class TestLLMStats:
    """Tests for LLM usage statistics."""

    def test_initial_stats(self):
        stats = LLMStats()
        assert stats.total_calls == 0
        assert stats.total_tokens == 0
        assert stats.avg_time_s == 0.0

    def test_avg_tokens_per_sec(self):
        stats = LLMStats(total_tokens=100, total_time_s=10.0)
        assert stats.avg_tokens_per_sec == 10.0


class TestOllamaClient:
    """Tests for the Ollama client (may skip if Ollama not running)."""

    def test_client_init(self):
        client = OllamaClient(model="test:7b", temperature=0.5)
        assert client.model == "test:7b"
        assert client.temperature == 0.5
        assert client.base_url == "http://localhost:11434"

    def test_client_custom_url(self):
        client = OllamaClient(base_url="http://192.168.1.100:11434")
        assert client.base_url == "http://192.168.1.100:11434"

    def test_unavailable_server(self):
        """Client should handle connection failure gracefully."""
        client = OllamaClient(base_url="http://localhost:19999")
        assert client.is_available() is False

    def test_generate_connection_error(self):
        """Generate should return error dict on connection failure."""
        client = OllamaClient(base_url="http://localhost:19999")
        result = client.generate("test", timeout=2)
        assert "error" in result
        assert result["response"] == ""

    def test_chat_connection_error(self):
        """Chat should return error dict on connection failure."""
        client = OllamaClient(base_url="http://localhost:19999")
        result = client.chat([{"role": "user", "content": "test"}], timeout=2)
        assert "error" in result

    def test_max_retries_default(self):
        """Default max_retries should be 2."""
        client = OllamaClient()
        assert client.max_retries == 2

    def test_max_retries_custom(self):
        """Should accept custom max_retries."""
        client = OllamaClient(max_retries=5)
        assert client.max_retries == 5

    def test_generate_retries_on_failure(self):
        """Generate should retry on connection failure up to max_retries."""
        client = OllamaClient(base_url="http://localhost:19999", max_retries=1)
        result = client.generate("test", timeout=2)
        assert "error" in result
        # Should have counted errors from retries
        assert client.stats.errors >= 1

    def test_generate_zero_retries(self):
        """With max_retries=0, should fail immediately."""
        client = OllamaClient(base_url="http://localhost:19999", max_retries=0)
        result = client.generate("test", timeout=2)
        assert "error" in result
        assert client.stats.errors == 1

    @pytest.mark.skipif(
        not OllamaClient().is_available(),
        reason="Ollama not running",
    )
    def test_list_models(self):
        """Should list models when Ollama is running."""
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)

    @pytest.mark.skipif(
        not OllamaClient().is_available(),
        reason="Ollama not running",
    )
    def test_get_version(self):
        """Should return version string when Ollama is running."""
        client = OllamaClient()
        version = client.get_version()
        assert version != "unavailable"


class TestModelCatalogue:
    """Tests for the model catalogue and size estimation."""

    def test_catalogue_has_models(self):
        assert len(MODEL_CATALOGUE) > 10

    def test_estimate_known_model(self):
        size = estimate_model_size("qwen2.5-coder:7b")
        assert size == 4.7

    def test_estimate_unknown_model(self):
        """Should estimate from parameter count in name."""
        size = estimate_model_size("some-model:13b")
        assert 7.0 < size < 10.0  # ~13 * 0.65

    def test_estimate_completely_unknown(self):
        """Should return default for unrecognizable names."""
        size = estimate_model_size("mystery-model")
        assert size == 5.0

    def test_get_models_for_category(self):
        coding_models = get_models_for_category("coding")
        assert len(coding_models) >= 3
        assert all(m.category == "coding" for m in coding_models)
        # Should be sorted by size
        sizes = [m.size_gb for m in coding_models]
        assert sizes == sorted(sizes)

    def test_get_models_that_fit(self):
        """Should return models that fit in GPU memory."""
        models = get_models_that_fit(10.0)
        assert len(models) > 0
        assert all(m.size_gb <= 9.0 for m in models)  # 10 - 1 GB headroom
        # Should be sorted descending by size
        sizes = [m.size_gb for m in models]
        assert sizes == sorted(sizes, reverse=True)
