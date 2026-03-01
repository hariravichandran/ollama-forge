# Hardware Guide

ollama-forge auto-detects your hardware and optimizes Ollama configuration. This guide explains what happens under the hood and how to troubleshoot.

## Supported Hardware

### AMD GPUs (Primary Target)

ollama-forge is optimized for AMD GPUs using ROCm:

- **RDNA2 APUs** (Rembrandt, Barcelo): Radeon 680M, 660M — iGPU with shared memory
- **RDNA2 Discrete**: RX 6600, 6700 XT, 6800 XT, 6900 XT
- **RDNA3 APUs** (Phoenix, Hawk Point): Radeon 780M, 760M
- **RDNA3 Discrete**: RX 7600, 7700 XT, 7800 XT, 7900 XT/XTX

### NVIDIA GPUs

Detected via `nvidia-smi`. CUDA should work out of the box with Ollama.

### CPU-Only

If no GPU is detected, ollama-forge runs in CPU-only mode. Smaller models (1.5B-3B) are recommended.

## Hardware Profiles

| Profile | GPU Memory | Models | Context | Batch |
|---------|-----------|--------|---------|-------|
| **compact** | 0-8 GB | 1.5B-3B | 4K | 1K |
| **standard** | 8-20 GB | 7B | 8K | 2K |
| **workstation** | 20-60 GB | 14B | 32K | 4K |
| **high_memory** | 60+ GB | 32B+ | 64K | 8K |

Check your profile:
```bash
forge hardware
```

## AMD iGPU (UMA) Notes

AMD integrated GPUs share system RAM. The total available is:
- **VRAM**: BIOS-reserved (typically 2-4 GB)
- **GTT**: Dynamic system RAM allocation (typically 12-14 GB)
- **Total**: VRAM + GTT

For a Radeon 680M with 32 GB system RAM: ~3 GB VRAM + ~14 GB GTT = ~17 GB total.

The `compact` and `standard` profiles account for this by reserving ~4 GB for the OS.

## ROCm Environment Variables

ollama-forge automatically sets these for optimal performance:

| Variable | Purpose | Auto-detected |
|----------|---------|--------------|
| `HSA_OVERRIDE_GFX_VERSION` | GPU architecture compatibility | Yes |
| `OLLAMA_FLASH_ATTENTION` | Flash attention speedup | Set to `1` |
| `OLLAMA_MAX_LOADED_MODELS` | Prevent OOM on iGPUs | `1` for iGPU, `2` for dGPU |

### HSA_OVERRIDE_GFX_VERSION

Some AMD GPUs need this variable to work correctly with ROCm:

| GPU Architecture | GFX Version | Override Value |
|------------------|-------------|----------------|
| RDNA2 APU (gfx1035/1036) | gfx1035 | `10.3.0` |
| RDNA2 Discrete (gfx1030-1032) | gfx1030 | `10.3.0` |
| RDNA3 APU (gfx1103) | gfx1103 | `11.0.0` |
| RDNA3 Discrete (gfx1100-1102) | gfx1100 | `11.0.0` |

ollama-forge detects this automatically via `rocminfo`.

## Overriding Hardware Detection

If auto-detection doesn't work correctly, override via `.env`:

```bash
# .env
FORGE_DEFAULT_MODEL=qwen2.5-coder:14b
HSA_OVERRIDE_GFX_VERSION=10.3.0
OLLAMA_FLASH_ATTENTION=1
OLLAMA_MAX_LOADED_MODELS=2
```

## Checking GPU Status

```bash
# ROCm status
forge hardware

# Detailed ROCm info
rocminfo

# GPU memory usage
rocm-smi

# NVIDIA GPU status
nvidia-smi

# Ollama loaded models (memory usage)
curl http://localhost:11434/api/ps
```
