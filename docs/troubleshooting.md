# Troubleshooting

Common issues and solutions for ollama-forge.

## Ollama Not Running

**Symptom**: `forge doctor` shows "Ollama: Not running"

**Fix**:
```bash
# Start Ollama
ollama serve

# Or if installed via systemd:
sudo systemctl start ollama
```

## No Models Installed

**Symptom**: `forge chat` says no models available

**Fix**:
```bash
# See recommended models for your hardware
forge models recommend

# Pull a model
forge models pull qwen2.5-coder:7b
```

## AMD GPU Not Detected

**Symptom**: `forge hardware` shows "CPU only" on AMD system

**Possible causes**:
1. ROCm not installed
2. GPU firmware not loaded
3. Unsupported GPU architecture

**Fix**:
```bash
# Check if GPU is visible
ls /sys/class/drm/card*/device/mem_info_vram_total

# Install ROCm (Ubuntu)
sudo apt install rocm-hip-runtime

# Check Ollama sees the GPU
ollama run qwen2.5-coder:3b "Hello"
# Watch GPU usage: watch -n1 cat /sys/class/drm/card0/device/gpu_busy_percent
```

## Out of Memory

**Symptom**: Ollama returns errors or system becomes unresponsive

**Fix**:
- Use a smaller model: `forge config set default_model qwen2.5-coder:3b`
- Reduce context window: `forge config set max_context_tokens 4096`
- Close other GPU-intensive applications
- Check current memory: `forge hardware`

## Slow Performance

**Symptom**: Tokens per second is very low

**Diagnosis**:
```bash
forge benchmark
```

**Fixes**:
- Use a smaller model (3b instead of 7b)
- Enable flash attention: `export OLLAMA_FLASH_ATTENTION=1`
- For AMD iGPU: set `HSA_OVERRIDE_GFX_VERSION` appropriately
- Reduce batch size via Ollama config
- Close background processes using GPU memory

## Web Search Not Working

**Symptom**: Agent can't search the web

**Fix**:
```bash
# Check if web search MCP is enabled
forge mcp list

# Re-enable it
forge config set web_search_enabled true

# Verify duckduckgo-search is installed
pip install duckduckgo-search
```

## Permission Errors

**Symptom**: Agent says "Permission denied" for file/shell operations

**Fix**:
- Use `--auto-approve` flag: `forge chat --auto-approve`
- Or approve each action when prompted
- Check `.forge-rules` for project-specific restrictions

## Docker Issues

**GPU not accessible in container**:
```bash
# AMD GPU
docker compose up  # Uncomment devices section in docker-compose.yml

# NVIDIA GPU
docker compose up  # Uncomment deploy section in docker-compose.yml
# Requires nvidia-container-toolkit
```

**Can't connect to Ollama**:
- When using Docker Compose, forge connects to `http://ollama:11434` (service name)
- When running forge outside Docker, use `http://localhost:11434`

## Configuration Issues

```bash
# View current config
forge config show

# Reset to defaults
forge config reset

# Show config file location
forge config path
```

## Getting Help

1. Run `forge doctor` to diagnose common issues
2. Check the [hardware guide](hardware-guide.md) for GPU-specific setup
3. Open an issue: https://github.com/hariravichandran/ollama-forge/issues
