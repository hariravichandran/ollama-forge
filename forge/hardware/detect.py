"""Hardware detection: GPU, CPU, and RAM — cross-platform.

Supports:
- Linux: sysfs (AMD), nvidia-smi (NVIDIA), lspci (Intel)
- macOS: system_profiler, sysctl (Apple Silicon, Intel Macs)
- Windows: nvidia-smi (NVIDIA), WMI/PowerShell (AMD, Intel)
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from forge.utils.logging import get_logger

log = get_logger("hardware.detect")

SYSTEM = platform.system()  # "Linux", "Darwin", "Windows"


@dataclass
class GPUInfo:
    """Detected GPU information."""

    vendor: str = "none"  # "amd", "nvidia", "intel", "apple", "none"
    name: str = "Unknown"
    vram_gb: float = 0.0
    gtt_gb: float = 0.0  # AMD UMA shared memory
    total_gb: float = 0.0
    driver: str = ""  # "rocm", "cuda", "metal", "cpu"
    driver_version: str = ""
    architecture: str = ""  # e.g., "gfx1035" for RDNA2, "arm64" for Apple Silicon
    is_igpu: bool = False

    @property
    def usable_gb(self) -> float:
        """Estimated usable memory for LLM inference."""
        if self.is_igpu:
            # iGPU shares system RAM — leave ~4GB for OS
            return max(0, self.total_gb - 4.0)
        if self.vendor == "apple":
            # Apple Silicon unified memory — leave ~4GB for OS
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
    platform: str = SYSTEM

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            f"GPU:  {self.gpu.name} ({self.gpu.vendor})",
            f"      {self.gpu.total_gb:.1f} GB ({self.gpu.driver or 'no driver'})",
        ]
        if self.gpu.is_igpu:
            lines.append(f"      iGPU: {self.gpu.vram_gb:.1f} GB VRAM + {self.gpu.gtt_gb:.1f} GB GTT (shared)")
        elif self.gpu.vendor == "apple":
            lines.append(f"      Unified memory (shared with CPU)")
        lines.extend([
            f"CPU:  {self.cpu.model}",
            f"      {self.cpu.threads} threads ({self.cpu.cores} cores)",
            f"RAM:  {self.ram_gb:.1f} GB",
            f"OS:   {self.platform}",
        ])
        return "\n".join(lines)


# In-process cache — hardware doesn't change within a single process lifetime
_hardware_cache: HardwareInfo | None = None


def detect_hardware(use_cache: bool = True) -> HardwareInfo:
    """Detect GPU, CPU, and RAM. Returns HardwareInfo with all findings.

    Results are cached in-process since hardware doesn't change at runtime.
    Pass use_cache=False to force re-detection.
    """
    global _hardware_cache
    if use_cache and _hardware_cache is not None:
        return _hardware_cache

    info = HardwareInfo()
    info.gpu = _detect_gpu()
    info.cpu = _detect_cpu()
    info.ram_gb = _detect_ram()
    log.debug("Hardware detected: GPU=%s (%.1f GB), CPU=%s (%d threads), RAM=%.1f GB",
              info.gpu.name, info.gpu.total_gb, info.cpu.model, info.cpu.threads, info.ram_gb)
    _hardware_cache = info
    return info


# ─── GPU Detection ───────────────────────────────────────────────────────────


def _detect_gpu() -> GPUInfo:
    """Detect GPU — dispatches to platform-specific detection."""
    if SYSTEM == "Darwin":
        apple = _detect_apple_gpu()
        if apple:
            return apple

    # NVIDIA works on all platforms
    nvidia = _detect_nvidia_gpu()
    if nvidia and nvidia.total_gb > 0:
        return nvidia

    if SYSTEM == "Linux":
        amd = _detect_amd_gpu_linux()
        if amd and amd.total_gb > 0:
            return amd

        intel = _detect_intel_gpu_linux()
        if intel and intel.total_gb > 0:
            return intel

    elif SYSTEM == "Windows":
        wmi = _detect_gpu_windows()
        if wmi and wmi.total_gb > 0:
            return wmi

    log.info("No GPU detected — CPU-only mode")
    return GPUInfo(vendor="none", name="CPU only", driver="cpu")


def _detect_nvidia_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU via nvidia-smi (works on Linux, Windows, macOS)."""
    if not shutil.which("nvidia-smi"):
        return None
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
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _detect_apple_gpu() -> GPUInfo | None:
    """Detect Apple Silicon GPU on macOS via sysctl and system_profiler."""
    if platform.machine() != "arm64":
        return None  # Intel Mac — no Metal GPU compute for LLMs

    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

        # Total unified memory (this IS the GPU memory on Apple Silicon)
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        total_bytes = int(result.stdout.strip()) if result.returncode == 0 else 0
        total_gb = total_bytes / (1024 ** 3)

        # Get GPU core count from system_profiler
        gpu_cores = ""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                if displays:
                    gpu_cores = displays[0].get("sppci_cores", "")
        except (json.JSONDecodeError, KeyError):
            pass

        name = chip_name
        if gpu_cores:
            name += f" ({gpu_cores}-core GPU)"

        return GPUInfo(
            vendor="apple",
            name=name,
            total_gb=total_gb,
            driver="metal",
            driver_version="Metal 3",
            architecture="arm64",
            is_igpu=False,  # Apple Silicon unified memory is not "iGPU" in the traditional sense
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _detect_amd_gpu_linux() -> GPUInfo | None:
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
            name = _read_amd_gpu_name(device_dir)
            is_igpu = vram_gb < 8.0 and gtt_gb > 0

            arch = ""
            rev_path = device_dir / "gpu_id"
            if rev_path.exists():
                arch = rev_path.read_text().strip()

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
    for name_file in ["product_name", "pp_dpm_sclk"]:
        path = device_dir / name_file
        if path.exists() and name_file == "product_name":
            name = path.read_text().strip()
            if name:
                return name

    try:
        result = subprocess.run(
            ["lspci", "-d", "1002:", "-nn"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                if "VGA" in line or "Display" in line:
                    match = re.search(r"\] (.+)$", line)
                    if match:
                        return match.group(1).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "AMD GPU"


def _detect_rocm_version() -> str:
    """Detect ROCm version if installed."""
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

    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        return version_file.read_text().strip()

    ver_header = Path("/opt/rocm/include/rocm-core/rocm_version.h")
    if ver_header.exists():
        content = ver_header.read_text()
        major = re.search(r"ROCM_VERSION_MAJOR\s+(\d+)", content)
        minor = re.search(r"ROCM_VERSION_MINOR\s+(\d+)", content)
        if major and minor:
            patch = re.search(r"ROCM_VERSION_PATCH\s+(\d+)", content)
            return f"{major.group(1)}.{minor.group(1)}.{patch.group(1) if patch else '0'}"

    return ""


def _detect_intel_gpu_linux() -> GPUInfo | None:
    """Basic Intel GPU detection via lspci (Linux only)."""
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
                    driver="cpu",
                    is_igpu=True,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_gpu_windows() -> GPUInfo | None:
    """Detect GPU on Windows via PowerShell WMI (AMD, Intel fallback)."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name, AdapterRAM, DriverVersion | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]

        for gpu in data:
            name = gpu.get("Name", "")
            adapter_ram = gpu.get("AdapterRAM", 0) or 0
            driver_ver = gpu.get("DriverVersion", "")
            total_gb = adapter_ram / (1024 ** 3) if adapter_ram else 0.0

            vendor = "none"
            if "AMD" in name or "Radeon" in name:
                vendor = "amd"
            elif "Intel" in name:
                vendor = "intel"

            if vendor != "none":
                return GPUInfo(
                    vendor=vendor,
                    name=name,
                    total_gb=total_gb,
                    driver="directx",
                    driver_version=driver_ver,
                    is_igpu=total_gb < 4.0,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass
    return None


# ─── CPU Detection ───────────────────────────────────────────────────────────


def _detect_cpu() -> CPUInfo:
    """Detect CPU model and thread count — cross-platform."""
    if SYSTEM == "Linux":
        return _detect_cpu_linux()
    elif SYSTEM == "Darwin":
        return _detect_cpu_macos()
    elif SYSTEM == "Windows":
        return _detect_cpu_windows()
    else:
        return CPUInfo(threads=os.cpu_count() or 1, cores=(os.cpu_count() or 2) // 2)


def _detect_cpu_linux() -> CPUInfo:
    """Detect CPU from /proc/cpuinfo."""
    info = CPUInfo()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text()
        match = re.search(r"model name\s*:\s*(.+)", content)
        if match:
            info.model = match.group(1).strip()
        info.threads = content.count("processor\t:")
        if info.threads == 0:
            info.threads = os.cpu_count() or 1
        core_ids = set(re.findall(r"core id\s*:\s*(\d+)", content))
        phys_ids = set(re.findall(r"physical id\s*:\s*(\d+)", content))
        if core_ids and phys_ids:
            info.cores = len(core_ids) * len(phys_ids)
        else:
            info.cores = info.threads // 2 or 1
    else:
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


def _detect_cpu_macos() -> CPUInfo:
    """Detect CPU on macOS via sysctl."""
    info = CPUInfo()
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.model = result.stdout.strip()

        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.threads = int(result.stdout.strip())

        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.cores = int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


def _detect_cpu_windows() -> CPUInfo:
    """Detect CPU on Windows via PowerShell."""
    info = CPUInfo()
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_Processor | "
             "Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list):
                data = data[0]
            info.model = data.get("Name", "Unknown").strip()
            info.cores = data.get("NumberOfCores", 1)
            info.threads = data.get("NumberOfLogicalProcessors", 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


# ─── RAM Detection ───────────────────────────────────────────────────────────


def _detect_ram() -> float:
    """Detect total system RAM in GB — cross-platform."""
    if SYSTEM == "Linux":
        return _detect_ram_linux()
    elif SYSTEM == "Darwin":
        return _detect_ram_macos()
    elif SYSTEM == "Windows":
        return _detect_ram_windows()

    # Fallback: psutil
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0


def _detect_ram_linux() -> float:
    """Detect RAM from /proc/meminfo."""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        content = meminfo_path.read_text()
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", content)
        if match:
            return int(match.group(1)) / (1024 * 1024)
    return 0.0


def _detect_ram_macos() -> float:
    """Detect RAM on macOS via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


def _detect_ram_windows() -> float:
    """Detect RAM on Windows via PowerShell."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0
