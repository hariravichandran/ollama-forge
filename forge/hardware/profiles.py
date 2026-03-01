"""Hardware profiles: map detected hardware to optimal Ollama settings.

Supports CPU-only systems — Ollama runs models on CPU when no GPU is available.
CPU inference is slower but fully functional. The framework auto-detects this
and selects appropriately sized models with optimized settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.hardware.detect import HardwareInfo
from forge.utils.logging import get_logger

log = get_logger("hardware.profiles")


@dataclass
class HardwareProfile:
    """Optimal settings for a hardware tier."""

    name: str
    min_gpu_gb: float
    recommended_model: str
    fallback_model: str
    larger_model: str  # for complex tasks
    num_ctx: int
    num_batch: int
    max_threads: int
    description: str
    is_cpu_only: bool = False  # True when running without GPU acceleration


# Hardware profiles from compact to high-end
PROFILES = [
    HardwareProfile(
        name="compact",
        min_gpu_gb=0,
        recommended_model="qwen2.5-coder:3b",
        fallback_model="qwen2.5-coder:1.5b",
        larger_model="qwen2.5-coder:7b",
        num_ctx=4096,
        num_batch=1024,
        max_threads=8,
        description="Low-memory systems (iGPU, <8 GB usable). Small models, short context.",
    ),
    HardwareProfile(
        name="standard",
        min_gpu_gb=8,
        recommended_model="qwen2.5-coder:7b",
        fallback_model="qwen2.5-coder:3b",
        larger_model="qwen2.5-coder:14b",
        num_ctx=8192,
        num_batch=2048,
        max_threads=14,
        description="Mid-range systems (8-20 GB GPU). 7B models with good context.",
    ),
    HardwareProfile(
        name="workstation",
        min_gpu_gb=20,
        recommended_model="qwen2.5-coder:14b",
        fallback_model="qwen2.5-coder:7b",
        larger_model="qwen2.5-coder:32b",
        num_ctx=32768,
        num_batch=4096,
        max_threads=14,
        description="Workstation systems (20-60 GB GPU). 14B models with large context.",
    ),
    HardwareProfile(
        name="high_memory",
        min_gpu_gb=60,
        recommended_model="qwen2.5-coder:32b",
        fallback_model="qwen2.5-coder:14b",
        larger_model="qwen2.5:72b",
        num_ctx=65536,
        num_batch=8192,
        max_threads=16,
        description="High-memory systems (60+ GB GPU). 32B+ models with maximum context.",
    ),
]


def select_profile(hw: HardwareInfo) -> HardwareProfile:
    """Select the best hardware profile based on detected hardware.

    For CPU-only systems, uses available RAM to determine model size.
    For iGPUs, uses usable_gb (total minus OS reservation).
    For dGPUs, uses total VRAM.
    """
    is_cpu_only = hw.gpu.vendor == "none" or hw.gpu.driver == "cpu"

    if is_cpu_only:
        # CPU-only: use RAM to determine model size
        # Ollama loads models into RAM when no GPU is available
        # Leave ~4 GB for OS + other apps
        available_ram = max(0, hw.ram_gb - 4.0)
        gpu_gb = available_ram
        log.info("CPU-only mode: using %.1f GB RAM for models (%.1f GB total)", available_ram, hw.ram_gb)
    elif hw.gpu.is_igpu:
        gpu_gb = hw.gpu.usable_gb
    else:
        gpu_gb = hw.gpu.total_gb

    # Select highest profile that fits
    selected = PROFILES[0]  # default to compact
    for profile in PROFILES:
        if gpu_gb >= profile.min_gpu_gb:
            selected = profile

    # Adjust thread count to actual hardware
    selected.max_threads = min(selected.max_threads, hw.cpu.threads)

    # Mark CPU-only mode and optimize settings
    if is_cpu_only:
        selected.is_cpu_only = True
        # CPU inference benefits from more threads and smaller batches
        selected.max_threads = max(1, hw.cpu.threads - 2)  # leave 2 threads for OS
        selected.num_batch = min(selected.num_batch, 512)  # smaller batches are faster on CPU

        # For very low RAM systems (< 8 GB), force the smallest model
        if hw.ram_gb < 8:
            selected.recommended_model = "qwen2.5-coder:1.5b"
            selected.fallback_model = "qwen2.5-coder:0.5b"
            selected.num_ctx = 2048

    log.info("Selected profile: %s (%.1f GB %s, %d threads%s)",
             selected.name, gpu_gb,
             "RAM" if is_cpu_only else "GPU",
             selected.max_threads,
             ", CPU-only" if is_cpu_only else "")
    return selected


def recommend_models(hw: HardwareInfo) -> list[dict[str, str]]:
    """Recommend models based on hardware, across different categories."""
    profile = select_profile(hw)
    is_cpu_only = profile.is_cpu_only

    if is_cpu_only:
        memory_str = f"{hw.ram_gb:.0f} GB RAM, CPU-only"
    elif hw.gpu.is_igpu:
        memory_str = f"{hw.gpu.usable_gb:.0f} GB usable"
    else:
        memory_str = f"{hw.gpu.total_gb:.0f} GB GPU"

    recommendations = []

    # Coding models
    reason = f"Best coding model for your {profile.name} hardware ({memory_str})"
    if is_cpu_only:
        reason += ". Slower on CPU but fully functional"
    recommendations.append({
        "category": "Coding",
        "model": profile.recommended_model,
        "reason": reason,
    })

    # General chat
    general_models = {
        "compact": "llama3.2:3b",
        "standard": "llama3.1:8b",
        "workstation": "llama3.1:70b",
        "high_memory": "llama3.1:70b",
    }
    recommendations.append({
        "category": "General chat",
        "model": general_models.get(profile.name, "llama3.2:3b"),
        "reason": "Good all-around conversational model",
    })

    # Research / reasoning
    reasoning_models = {
        "compact": "deepseek-r1:1.5b",
        "standard": "deepseek-r1:8b",
        "workstation": "deepseek-r1:14b",
        "high_memory": "deepseek-r1:32b",
    }
    recommendations.append({
        "category": "Reasoning",
        "model": reasoning_models.get(profile.name, "deepseek-r1:8b"),
        "reason": "Chain-of-thought reasoning for complex problems",
    })

    # Creative writing
    creative_models = {
        "compact": "gemma3:4b",
        "standard": "gemma3:12b",
        "workstation": "gemma3:27b",
        "high_memory": "gemma3:27b",
    }
    recommendations.append({
        "category": "Creative writing",
        "model": creative_models.get(profile.name, "gemma3:4b"),
        "reason": "Strong creative and instructional capabilities",
    })

    return recommendations
