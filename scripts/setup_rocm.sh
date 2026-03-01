#!/bin/bash
# ROCm installation helper for AMD GPUs
# Supports Ubuntu 22.04/24.04, Fedora 39+
#
# This script installs ROCm for Ollama GPU acceleration on AMD hardware.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    error "Do not run as root. The script will use sudo when needed."
fi

# Detect distro
if [ ! -f /etc/os-release ]; then
    error "Cannot detect Linux distribution."
fi
. /etc/os-release

info "Detected: $PRETTY_NAME"

case "$ID" in
    ubuntu|pop|linuxmint)
        info "Installing ROCm for Ubuntu/Debian-based system..."

        # Add AMD GPU repo
        sudo apt-get update -qq
        sudo apt-get install -y -qq wget gnupg2

        # Install amdgpu-install if not present
        if ! command -v amdgpu-install &>/dev/null; then
            AMDGPU_DEB="amdgpu-install_6.4.60400-1_all.deb"
            wget -q "https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/$AMDGPU_DEB" -O "/tmp/$AMDGPU_DEB" || {
                warn "Could not download ROCm installer. Check https://rocm.docs.amd.com for latest URL."
                exit 1
            }
            sudo dpkg -i "/tmp/$AMDGPU_DEB"
            sudo apt-get update -qq
        fi

        # Install ROCm (no DKMS for iGPU — kernel driver already included)
        sudo amdgpu-install -y --usecase=rocm --no-dkms

        success "ROCm installed"
        ;;

    fedora)
        info "Installing ROCm for Fedora..."

        sudo dnf install -y "https://repo.radeon.com/amdgpu-install/6.4/rhel/9.5/amdgpu-install-6.4.60400-1.el9.noarch.rpm" || {
            warn "Could not install ROCm repo. Check https://rocm.docs.amd.com for latest URL."
            exit 1
        }

        sudo dnf install -y rocm-hip-runtime rocm-smi-lib

        success "ROCm installed"
        ;;

    arch|manjaro|endeavouros)
        info "Installing ROCm for Arch Linux..."

        sudo pacman -Sy --noconfirm rocm-hip-runtime rocm-smi-lib hip-runtime-amd || {
            warn "ROCm packages not found. Try: yay -S rocm-hip-runtime"
            exit 1
        }

        success "ROCm installed"
        ;;

    *)
        error "Unsupported distro: $ID. Install ROCm manually from https://rocm.docs.amd.com"
        ;;
esac

# Add user to required groups
info "Adding $USER to render and video groups..."
for grp in render video; do
    if getent group "$grp" &>/dev/null; then
        sudo usermod -aG "$grp" "$USER"
    fi
done

# Verify installation
info "Verifying ROCm installation..."
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VER=$(cat /opt/rocm/.info/version)
    success "ROCm version: $ROCM_VER"
else
    warn "ROCm version file not found — installation may be incomplete."
fi

if command -v rocminfo &>/dev/null; then
    GFX=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
    success "GPU architecture: $GFX"
else
    warn "rocminfo not found. ROCm tools may not be fully installed."
fi

echo ""
info "You may need to log out and back in for group changes to take effect."
info "Then restart Ollama: systemctl --user restart ollama"
