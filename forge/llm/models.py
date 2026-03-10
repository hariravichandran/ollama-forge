"""Model registry and size catalogue for memory-aware model selection."""

from __future__ import annotations

import re
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


# Quantization format multipliers (GB per billion parameters)
# Based on empirical measurements of GGUF model sizes
QUANTIZATION_MULTIPLIERS: dict[str, float] = {
    "q2_k": 0.35,
    "q3_k_s": 0.40,
    "q3_k_m": 0.43,
    "q3_k_l": 0.46,
    "q4_0": 0.50,
    "q4_k_s": 0.55,
    "q4_k_m": 0.65,   # Most common default
    "q5_0": 0.70,
    "q5_k_s": 0.75,
    "q5_k_m": 0.78,
    "q6_k": 0.88,
    "q8_0": 1.10,
    "f16": 2.00,
    "f32": 4.00,
}

# Default quantization assumed when not specified in model name
DEFAULT_QUANTIZATION = "q4_k_m"

# Valid model name pattern: namespace/model:tag or model:tag
MODEL_NAME_PATTERN = r"^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?(:[a-zA-Z0-9_.\-]+)?$"


def estimate_model_size(model_name: str, quantization: str = "") -> float:
    """Estimate model VRAM usage in GB.

    Uses the catalogue if available, otherwise estimates from parameter count
    with quantization-aware multiplier.

    Args:
        model_name: Ollama model name (e.g., "llama3.1:8b")
        quantization: Override quantization format (e.g., "q5_k_m", "q8_0")
    """
    if model_name in MODEL_CATALOGUE and not quantization:
        return MODEL_CATALOGUE[model_name].size_gb

    # Try to extract parameter count from name (e.g., "model:7b" -> 7)
    match = re.search(r"(\d+\.?\d*)b", model_name.lower())
    if match:
        params_b = float(match.group(1))
        # Determine quantization multiplier
        quant = (quantization or _detect_quantization(model_name) or DEFAULT_QUANTIZATION).lower()
        multiplier = QUANTIZATION_MULTIPLIERS.get(quant, QUANTIZATION_MULTIPLIERS[DEFAULT_QUANTIZATION])
        return round(params_b * multiplier, 1)

    return 5.0  # default estimate


def _detect_quantization(model_name: str) -> str:
    """Try to detect quantization format from model name or tag.

    Checks for common quantization suffixes like ':q5_k_m', '-q8_0', etc.
    """
    name_lower = model_name.lower()
    for quant in sorted(QUANTIZATION_MULTIPLIERS.keys(), key=len, reverse=True):
        if quant in name_lower:
            return quant
    return ""


def validate_model_name(model_name: str) -> str:
    """Validate an Ollama model name format.

    Returns error message or empty string if valid.
    """
    if not model_name or not model_name.strip():
        return "model name cannot be empty"
    if len(model_name) > 200:
        return "model name too long (max 200 characters)"
    if not re.match(MODEL_NAME_PATTERN, model_name):
        return f"invalid model name format: '{model_name}' (expected: name:tag or namespace/name:tag)"
    return ""


def get_models_for_category(category: str) -> list[ModelInfo]:
    """Get all known models in a category, sorted by size."""
    models = [m for m in MODEL_CATALOGUE.values() if m.category == category]
    return sorted(models, key=lambda m: m.size_gb)


def get_models_that_fit(gpu_gb: float, headroom_gb: float = 0.0) -> list[ModelInfo]:
    """Get all models that fit in available GPU memory, sorted by size descending.

    Args:
        gpu_gb: Total GPU memory in GB.
        headroom_gb: Override headroom. If 0, uses dynamic calculation.
    """
    if headroom_gb <= 0:
        headroom_gb = _calculate_headroom(gpu_gb)
    available = gpu_gb - headroom_gb
    models = [m for m in MODEL_CATALOGUE.values() if m.size_gb <= available]
    return sorted(models, key=lambda m: m.size_gb, reverse=True)


def _calculate_headroom(gpu_gb: float) -> float:
    """Calculate dynamic headroom based on GPU memory size.

    Smaller GPUs need proportionally more headroom for OS/display overhead.
    """
    if gpu_gb <= 4:
        return 0.8   # Small iGPU — tight but functional
    elif gpu_gb <= 8:
        return 1.0   # Standard headroom
    elif gpu_gb <= 16:
        return 1.5   # Medium GPU
    else:
        return 2.0   # Large GPU — more overhead from display, OS, etc.
