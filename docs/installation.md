# Installation Guide

## Automatic Installation

The quickest way to install:

```bash
git clone https://github.com/ollama-forge/ollama-forge.git
cd ollama-forge
bash install.sh
```

## Manual Installation

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install python3 python3-pip python3-venv git curl
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip git curl
```

**Arch:**
```bash
sudo pacman -S python python-pip git curl
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. AMD ROCm (Optional, for AMD GPUs)

If you have an AMD GPU, install ROCm for GPU acceleration:

```bash
bash scripts/setup_rocm.sh
```

Or follow the official guide: https://rocm.docs.amd.com/projects/install-on-linux/

After installation, add yourself to the required groups:
```bash
sudo usermod -aG render,video $USER
```
Then log out and back in.

### 4. Install ollama-forge

```bash
cd ollama-forge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 5. Pull a Model

```bash
# See recommendations for your hardware
forge models recommend

# Pull a model
forge models pull qwen2.5-coder:7b
```

### 6. Verify

```bash
forge hardware    # Should show your GPU, CPU, RAM
forge benchmark   # Run a quick performance test
forge chat        # Start chatting!
```

## Optional: TUI and Web UI

```bash
# Terminal UI (Textual-based)
pip install ollama-forge[tui]

# Web UI (FastAPI-based)
pip install ollama-forge[web]

# Everything
pip install ollama-forge[all]
```

## Troubleshooting

### Ollama not starting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start manually
ollama serve

# Check logs
journalctl --user -u ollama -f
```

### AMD GPU not detected
```bash
# Check if ROCm is installed
rocminfo | head -20

# Check groups
groups | grep -E "render|video"

# Check sysfs
ls /sys/class/drm/card*/device/mem_info_vram_total
```

### Model too slow
```bash
# Check what's running
forge models list

# Try a smaller model
forge models pull qwen2.5-coder:3b
forge chat --model qwen2.5-coder:3b
```
