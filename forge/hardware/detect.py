"""Hardware detection: GPU, CPU, and RAM via Linux sysfs and /proc."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from forge.utils.logging import get_logger

log = get_logger("hardware.detect")


@dataclass
class GPUInfo:
    """Detected GPU information."""

    vendor: str = "none"  # "amd", "nvidia", "intel", "none"
    name: str = "Unknown"
    vram_gb: float = 0.0
    gtt_gb: float = 0.0  # AMD UMA shared memory
    total_gb: float = 0.0
    driver: str = ""  # "rocm", "cuda", "cpu"
    driver_version: str = ""
    architecture: str = ""  # e.g., "gfx1035" for RDNA2
    is_igpu: bool = False

    @property
    def usable_gb(self) -> float:
        """Estimated usable memory for LLM inference."""
        if self.is_igpu:
            # iGPU shares system RAM — leave ~4GB for OS
            return max(0, self.total_gb - 4.0)
        return self.total_gb


@dataclass
class CPUInfo:
    """Detected CPU information."""

    model: str = "Unknown"
    threads: int = 1
    cores: int = 1


@dataclass
class HardwareInfo:
    """Complete hardware detection result."""

    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram_gb: float = 0.0

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            f"GPU:  {self.gpu.name} ({self.gpu.vendor})",
            f"      {self.gpu.total_gb:.1f} GB ({self.gpu.driver or 'no driver'})",
        ]
        if self.gpu.is_igpu:
            lines.append(f"      iGPU: {self.gpu.vram_gb:.1f} GB VRAM + {self.gpu.gtt_gb:.1f} GB GTT (shared)")
        lines.extend([
            f"CPU:  {self.cpu.model}",
            f"      {self.cpu.threads} threads ({self.cpu.cores} cores)",
            f"RAM:  {self.ram_gb:.1f} GB",
        ])
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """Detect GPU, CPU, and RAM. Returns HardwareInfo with all findings."""
    info = HardwareInfo()
    info.gpu = _detect_gpu()
    info.cpu = _detect_cpu()
    info.ram_gb = _detect_ram()
    log.debug("Hardware detected: GPU=%s (%.1f GB), CPU=%s (%d threads), RAM=%.1f GB",
              info.gpu.name, info.gpu.total_gb, info.cpu.model, info.cpu.threads, info.ram_gb)
    return info


def _detect_gpu() -> GPUInfo:
    """Detect GPU via sysfs (AMD) or nvidia-smi (NVIDIA)."""
    # Try AMD first (sysfs)
    amd = _detect_amd_gpu()
    if amd and amd.total_gb > 0:
        return amd

    # Try NVIDIA
    nvidia = _detect_nvidia_gpu()
    if nvidia and nvidia.total_gb > 0:
        return nvidia

    # Try Intel
    intel = _detect_intel_gpu()
    if intel and intel.total_gb > 0:
        return intel

    log.info("No GPU detected — CPU-only mode")
    return GPUInfo(vendor="none", name="CPU only", driver="cpu")


def _detect_amd_gpu() -> GPUInfo | None:
    """Detect AMD GPU via Linux sysfs."""
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return None

    best_gpu = None
    best_total = 0.0

    for card_dir in sorted(drm_path.glob("card[0-9]*")):
        device_dir = card_dir / "device"
        if not device_dir.exists():
            continue

        # Check if AMD
        vendor_path = device_dir / "vendor"
        if vendor_path.exists():
            vendor_id = vendor_path.read_text().strip()
            if vendor_id != "0x1002":  # AMD vendor ID
                continue

        vram_path = device_dir / "mem_info_vram_total"
        gtt_path = device_dir / "mem_info_gtt_total"

        if not vram_path.exists():
            continue

        vram_bytes = int(vram_path.read_text().strip())
        gtt_bytes = int(gtt_path.read_text().strip()) if gtt_path.exists() else 0

        vram_gb = vram_bytes / (1024 ** 3)
        gtt_gb = gtt_bytes / (1024 ** 3)
        total_gb = vram_gb + gtt_gb

        if total_gb > best_total:
            best_total = total_gb
            # Detect GPU name
            name = _read_amd_gpu_name(device_dir)
            # Detect if iGPU (VRAM < 8GB typically means iGPU/APU)
            is_igpu = vram_gb < 8.0 and gtt_gb > 0

            # Detect architecture
            arch = ""
            rev_path = device_dir / "gpu_id"
            if rev_path.exists():
                arch = rev_path.read_text().strip()

            # Check ROCm
            driver = "cpu"
            driver_version = ""
            rocm_ver = _detect_rocm_version()
            if rocm_ver:
                driver = "rocm"
                driver_version = rocm_ver

            best_gpu = GPUInfo(
                vendor="amd",
                name=name,
                vram_gb=vram_gb,
                gtt_gb=gtt_gb,
                total_gb=total_gb,
                driver=driver,
                driver_version=driver_version,
                architecture=arch,
                is_igpu=is_igpu,
            )

    return best_gpu


def _read_amd_gpu_name(device_dir: Path) -> str:
    """Read AMD GPU product name from sysfs or lspci."""
    # Try sysfs product name
    for name_file in ["product_name", "pp_dpm_sclk"]:
        path = device_dir / name_file
        if path.exists() and name_file == "product_name":
            name = path.read_text().strip()
            if name:
                return name

    # Fallback to lspci
    try:
        result = subprocess.run(
            ["lspci", "-d", "1002:", "-nn"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse first VGA/Display line
            for line in result.stdout.strip().splitlines():
                if "VGA" in line or "Display" in line:
                    # Extract name after the bracket section
                    match = re.search(r"\] (.+)$", line)
                    if match:
                        return match.group(1).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "AMD GPU"


def _detect_rocm_version() -> str:
    """Detect ROCm version if installed."""
    # Check rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showversion"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "ROCm" in line or "version" in line.lower():
                    match = re.search(r"(\d+\.\d+[\.\d]*)", line)
                    if match:
                        return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check /opt/rocm/.info/version
    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        return version_file.read_text().strip()

    # Check /opt/rocm/include/rocm-core/rocm_version.h
    ver_header = Path("/opt/rocm/include/rocm-core/rocm_version.h")
    if ver_header.exists():
        content = ver_header.read_text()
        major = re.search(r"ROCM_VERSION_MAJOR\s+(\d+)", content)
        minor = re.search(r"ROCM_VERSION_MINOR\s+(\d+)", content)
        if major and minor:
            patch = re.search(r"ROCM_VERSION_PATCH\s+(\d+)", content)
            return f"{major.group(1)}.{minor.group(1)}.{patch.group(1) if patch else '0'}"

    return ""


def _detect_nvidia_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        name = parts[0]
        vram_mb = float(parts[1])
        driver_version = parts[2]

        return GPUInfo(
            vendor="nvidia",
            name=name,
            vram_gb=vram_mb / 1024,
            total_gb=vram_mb / 1024,
            driver="cuda",
            driver_version=driver_version,
            is_igpu=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, (ValueError, IndexError)):
        return None


def _detect_intel_gpu() -> GPUInfo | None:
    """Basic Intel GPU detection via lspci."""
    try:
        result = subprocess.run(
            ["lspci", "-d", "8086:", "-nn"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            if "VGA" in line or "Display" in line:
                match = re.search(r"\] (.+)$", line)
                name = match.group(1).strip() if match else "Intel GPU"
                return GPUInfo(
                    vendor="intel",
                    name=name,
                    driver="cpu",  # Intel GPU compute support is limited
                    is_igpu=True,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_cpu() -> CPUInfo:
    """Detect CPU model and thread count from /proc/cpuinfo."""
    info = CPUInfo()

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text()

        # Model name (first occurrence)
        match = re.search(r"model name\s*:\s*(.+)", content)
        if match:
            info.model = match.group(1).strip()

        # Count logical processors
        info.threads = content.count("processor\t:")
        if info.threads == 0:
            info.threads = os.cpu_count() or 1

        # Core count (unique core ids)
        core_ids = set(re.findall(r"core id\s*:\s*(\d+)", content))
        # Multiply by number of physical packages
        phys_ids = set(re.findall(r"physical id\s*:\s*(\d+)", content))
        if core_ids and phys_ids:
            info.cores = len(core_ids) * len(phys_ids)
        else:
            info.cores = info.threads // 2 or 1
    else:
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1

    return info


def _detect_ram() -> float:
    """Detect total system RAM in GB from /proc/meminfo."""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        content = meminfo_path.read_text()
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", content)
        if match:
            return int(match.group(1)) / (1024 * 1024)

    # Fallback
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0
