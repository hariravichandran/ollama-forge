"""Model registry and size catalogue for memory-aware model selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Known model metadata."""

    name: str
    size_gb: float  # approximate VRAM usage (Q4_K_M quantization)
    category: str   # "coding", "general", "reasoning", "creative", "finance", "embedding"
    description: str
    parameters: str  # "1.5b", "7b", "14b", etc.


# Known model sizes (Q4_K_M quantization, approximate GPU memory usage)
MODEL_CATALOGUE: dict[str, ModelInfo] = {
    # Coding models
    "qwen2.5-coder:1.5b": ModelInfo("qwen2.5-coder:1.5b", 1.2, "coding", "Lightweight coding assistant", "1.5b"),
    "qwen2.5-coder:3b": ModelInfo("qwen2.5-coder:3b", 2.2, "coding", "Compact coding assistant", "3b"),
    "qwen2.5-coder:7b": ModelInfo("qwen2.5-coder:7b", 4.7, "coding", "Balanced coding model — great for most tasks", "7b"),
    "qwen2.5-coder:14b": ModelInfo("qwen2.5-coder:14b", 9.0, "coding", "Advanced coding with deep understanding", "14b"),
    "qwen2.5-coder:32b": ModelInfo("qwen2.5-coder:32b", 20.0, "coding", "Expert-level coding for complex projects", "32b"),
    "codellama:7b": ModelInfo("codellama:7b", 4.5, "coding", "Meta's code-focused Llama model", "7b"),
    "codellama:13b": ModelInfo("codellama:13b", 8.0, "coding", "Larger CodeLlama variant", "13b"),

    # General models
    "llama3.2:1b": ModelInfo("llama3.2:1b", 0.8, "general", "Ultra-lightweight chat", "1b"),
    "llama3.2:3b": ModelInfo("llama3.2:3b", 2.0, "general", "Compact general assistant", "3b"),
    "llama3.1:8b": ModelInfo("llama3.1:8b", 5.0, "general", "Solid all-around model", "8b"),
    "llama3.1:70b": ModelInfo("llama3.1:70b", 42.0, "general", "Frontier-class general model", "70b"),
    "qwen2.5:7b": ModelInfo("qwen2.5:7b", 4.7, "general", "Strong multilingual model", "7b"),
    "qwen2.5:14b": ModelInfo("qwen2.5:14b", 9.0, "general", "Advanced multilingual reasoning", "14b"),
    "qwen2.5:32b": ModelInfo("qwen2.5:32b", 20.0, "general", "Expert-level reasoning", "32b"),
    "qwen2.5:72b": ModelInfo("qwen2.5:72b", 42.5, "general", "Maximum capability Qwen", "72b"),
    "gemma3:4b": ModelInfo("gemma3:4b", 2.8, "general", "Google's compact model", "4b"),
    "gemma3:12b": ModelInfo("gemma3:12b", 8.1, "general", "Google's mid-range model", "12b"),
    "gemma3:27b": ModelInfo("gemma3:27b", 17.0, "general", "Google's large model", "27b"),

    # Reasoning models
    "deepseek-r1:1.5b": ModelInfo("deepseek-r1:1.5b", 1.1, "reasoning", "Lightweight chain-of-thought", "1.5b"),
    "deepseek-r1:7b": ModelInfo("deepseek-r1:7b", 4.7, "reasoning", "Balanced reasoning model", "7b"),
    "deepseek-r1:8b": ModelInfo("deepseek-r1:8b", 5.2, "reasoning", "Extended chain-of-thought reasoning", "8b"),
    "deepseek-r1:14b": ModelInfo("deepseek-r1:14b", 9.0, "reasoning", "Deep analytical reasoning", "14b"),
    "deepseek-r1:32b": ModelInfo("deepseek-r1:32b", 20.0, "reasoning", "Advanced deep reasoning", "32b"),

    # Finance (example domain models)
    "0xroyce/plutus": ModelInfo("0xroyce/plutus", 5.7, "finance", "Trained on 394 finance books", "7b"),

    # Embedding models
    "nomic-embed-text": ModelInfo("nomic-embed-text", 0.3, "embedding", "Text embeddings for semantic search", "137m"),
    "mxbai-embed-large": ModelInfo("mxbai-embed-large", 0.7, "embedding", "High-quality embeddings", "335m"),
}


def estimate_model_size(model_name: str) -> float:
    """Estimate model VRAM usage in GB.

    Uses the catalogue if available, otherwise estimates from parameter count.
    """
    if model_name in MODEL_CATALOGUE:
        return MODEL_CATALOGUE[model_name].size_gb

    # Try to extract parameter count from name (e.g., "model:7b" -> 7)
    import re
    match = re.search(r"(\d+\.?\d*)b", model_name.lower())
    if match:
        params_b = float(match.group(1))
        # Q4_K_M: roughly 0.65 GB per billion parameters
        return params_b * 0.65

    return 5.0  # default estimate


def get_models_for_category(category: str) -> list[ModelInfo]:
    """Get all known models in a category, sorted by size."""
    models = [m for m in MODEL_CATALOGUE.values() if m.category == category]
    return sorted(models, key=lambda m: m.size_gb)


def get_models_that_fit(gpu_gb: float) -> list[ModelInfo]:
    """Get all models that fit in available GPU memory, sorted by size descending."""
    # Leave ~1GB headroom for OS/overhead
    available = gpu_gb - 1.0
    models = [m for m in MODEL_CATALOGUE.values() if m.size_gb <= available]
    return sorted(models, key=lambda m: m.size_gb, reverse=True)
