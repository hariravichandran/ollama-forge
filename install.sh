#!/bin/bash
# ollama-forge installer
# One-command install: curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash
#
# What this does:
#   1. Checks Linux distro and installs system deps
#   2. Detects GPU (AMD ROCm, NVIDIA CUDA, or CPU-only)
#   3. Installs Ollama (if not present)
#   4. Configures Ollama for your GPU
#   5. Creates Python venv and installs ollama-forge
#   6. Recommends and pulls a starter model
#   7. Enables web search MCP by default

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Step 1: Check Linux ─────────────────────────────────────────────────────

info "Checking system..."
if [[ "$(uname -s)" != "Linux" ]]; then
    error "ollama-forge currently supports Linux only."
fi

# Detect distro
DISTRO="unknown"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO="$ID"
fi
info "Detected distro: $DISTRO"

# ─── Step 2: System dependencies ─────────────────────────────────────────────

info "Installing system dependencies..."
case "$DISTRO" in
    ubuntu|debian|pop|linuxmint)
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-pip python3-venv git curl
        ;;
    fedora|rhel|centos)
        sudo dnf install -y python3 python3-pip git curl
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Sy --noconfirm python python-pip git curl
        ;;
    *)
        warn "Unknown distro '$DISTRO'. Ensure python3, pip, git, and curl are installed."
        ;;
esac
success "System deps installed"

# ─── Step 3: Detect GPU ──────────────────────────────────────────────────────

GPU_TYPE="cpu"
GPU_NAME="CPU only"

# Check AMD
if [ -d /sys/class/drm ]; then
    for card in /sys/class/drm/card[0-9]*/device; do
        if [ -f "$card/vendor" ]; then
            vendor=$(cat "$card/vendor" 2>/dev/null || true)
            if [ "$vendor" = "0x1002" ]; then
                GPU_TYPE="amd"
                if command -v lspci &>/dev/null; then
                    GPU_NAME=$(lspci -d 1002: -nn 2>/dev/null | head -1 | sed 's/.*\] //' || echo "AMD GPU")
                else
                    GPU_NAME="AMD GPU"
                fi
                break
            fi
        fi
    done
fi

# Check NVIDIA
if command -v nvidia-smi &>/dev/null; then
    GPU_TYPE="nvidia"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
fi

info "GPU: $GPU_NAME ($GPU_TYPE)"

# ─── Step 4: AMD ROCm setup ──────────────────────────────────────────────────

if [ "$GPU_TYPE" = "amd" ]; then
    if [ -d /opt/rocm ]; then
        ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
        success "ROCm already installed: $ROCM_VER"
    else
        info "AMD GPU detected. ROCm is recommended for GPU acceleration."
        read -rp "Install ROCm? (y/n) " install_rocm
        if [[ "$install_rocm" =~ ^[Yy] ]]; then
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            if [ -f "$SCRIPT_DIR/scripts/setup_rocm.sh" ]; then
                bash "$SCRIPT_DIR/scripts/setup_rocm.sh"
            else
                warn "ROCm setup script not found. Install manually:"
                warn "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
            fi
        fi
    fi
fi

# ─── Step 5: Install Ollama ──────────────────────────────────────────────────

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    success "Ollama already installed: $OLLAMA_VER"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed"
fi

# ─── Step 6: Configure Ollama for GPU ────────────────────────────────────────

if [ "$GPU_TYPE" = "amd" ]; then
    info "Configuring Ollama for AMD ROCm..."

    # Create systemd override for Ollama
    OVERRIDE_DIR="$HOME/.config/systemd/user/ollama.service.d"
    mkdir -p "$OVERRIDE_DIR"
    cat > "$OVERRIDE_DIR/override.conf" << 'OLLAMA_EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
OLLAMA_EOF

    # Detect GFX version for HSA override
    if command -v rocminfo &>/dev/null; then
        GFX=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || true)
        case "$GFX" in
            gfx1035|gfx1036)
                echo 'Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"' >> "$OVERRIDE_DIR/override.conf"
                info "Set HSA_OVERRIDE_GFX_VERSION=10.3.0 for $GFX"
                ;;
            gfx1103)
                echo 'Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"' >> "$OVERRIDE_DIR/override.conf"
                info "Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for $GFX"
                ;;
            gfx1100|gfx1101|gfx1102)
                echo 'Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"' >> "$OVERRIDE_DIR/override.conf"
                info "Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for $GFX"
                ;;
            gfx1030|gfx1031|gfx1032)
                echo 'Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"' >> "$OVERRIDE_DIR/override.conf"
                info "Set HSA_OVERRIDE_GFX_VERSION=10.3.0 for $GFX"
                ;;
        esac
    fi

    # Ensure user is in render and video groups
    for grp in render video; do
        if getent group "$grp" &>/dev/null; then
            if ! groups | grep -qw "$grp"; then
                sudo usermod -aG "$grp" "$USER"
                info "Added $USER to $grp group (re-login may be needed)"
            fi
        fi
    done

    success "Ollama configured for AMD ROCm"
fi

# ─── Step 7: Create venv and install ollama-forge ─────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

info "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$REPO_DIR" -q
success "ollama-forge installed in $VENV_DIR"

# ─── Step 8: Start Ollama and pull starter model ─────────────────────────────

info "Starting Ollama..."
if ! curl -sf http://localhost:11434/api/version &>/dev/null; then
    ollama serve &>/dev/null &
    sleep 3
fi

if curl -sf http://localhost:11434/api/version &>/dev/null; then
    success "Ollama is running"

    # Detect hardware profile and recommend model
    info "Detecting hardware profile..."
    RECOMMENDED=$("$VENV_DIR/bin/python" -c "
from forge.hardware import detect_hardware, select_profile
hw = detect_hardware()
profile = select_profile(hw)
print(profile.recommended_model)
" 2>/dev/null || echo "qwen2.5-coder:7b")

    info "Recommended model: $RECOMMENDED"
    read -rp "Pull $RECOMMENDED now? (y/n) " pull_model
    if [[ "$pull_model" =~ ^[Yy] ]]; then
        ollama pull "$RECOMMENDED"
        success "Model $RECOMMENDED ready"
    fi
else
    warn "Ollama not running. Start it with: ollama serve"
fi

# ─── Step 9: Done ────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ollama-forge installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Quickstart:"
echo "    source $VENV_DIR/bin/activate"
echo "    forge hardware          # Check your hardware profile"
echo "    forge models recommend  # See recommended models"
echo "    forge chat              # Start chatting!"
echo ""
echo "  More commands:"
echo "    forge agent list        # List available agents"
echo "    forge mcp list          # List MCP servers"
echo "    forge benchmark         # Benchmark your hardware"
echo "    forge --help            # Full help"
echo ""
