"""Tests for hardware detection module."""

import os
from pathlib import Path

import pytest

from forge.hardware.detect import (
    detect_hardware,
    HardwareInfo,
    GPUInfo,
    CPUInfo,
    _detect_cpu,
    _detect_ram,
)
from forge.hardware.profiles import select_profile, recommend_models, PROFILES, HardwareProfile


class TestHardwareDetect:
    """Tests for hardware detection."""

    def test_detect_hardware_returns_info(self):
        """detect_hardware should return a HardwareInfo object."""
        hw = detect_hardware()
        assert isinstance(hw, HardwareInfo)
        assert isinstance(hw.gpu, GPUInfo)
        assert isinstance(hw.cpu, CPUInfo)

    def test_cpu_detection(self):
        """CPU detection should find at least 1 thread."""
        cpu = _detect_cpu()
        assert cpu.threads >= 1
        assert cpu.cores >= 1
        assert isinstance(cpu.model, str)

    def test_ram_detection(self):
        """RAM detection should find at least some memory."""
        ram_gb = _detect_ram()
        assert ram_gb > 0

    def test_gpu_info_usable_gb(self):
        """iGPU usable_gb should subtract OS reservation."""
        igpu = GPUInfo(vendor="amd", total_gb=17.0, is_igpu=True)
        assert igpu.usable_gb == 13.0  # 17 - 4

        dgpu = GPUInfo(vendor="nvidia", total_gb=24.0, is_igpu=False)
        assert dgpu.usable_gb == 24.0  # no reservation for dGPU

    def test_gpu_info_defaults(self):
        """Default GPUInfo should be safe (no GPU)."""
        gpu = GPUInfo()
        assert gpu.vendor == "none"
        assert gpu.total_gb == 0.0
        assert gpu.usable_gb == 0.0

    def test_hardware_summary(self):
        """summary() should return a readable string."""
        hw = detect_hardware()
        summary = hw.summary()
        assert isinstance(summary, str)
        assert "GPU:" in summary
        assert "CPU:" in summary
        assert "RAM:" in summary


class TestHardwareProfiles:
    """Tests for hardware profile selection."""

    def test_profiles_exist(self):
        """All profiles should be defined."""
        assert len(PROFILES) == 4
        names = [p.name for p in PROFILES]
        assert "compact" in names
        assert "standard" in names
        assert "workstation" in names
        assert "high_memory" in names

    def test_select_profile_compact(self):
        """Small GPU should get compact profile."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="amd", total_gb=6.0, is_igpu=True),
            cpu=CPUInfo(threads=8),
            ram_gb=8.0,
        )
        profile = select_profile(hw)
        assert profile.name == "compact"

    def test_select_profile_standard(self):
        """Mid-range GPU should get standard profile."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="amd", total_gb=17.0, is_igpu=True),
            cpu=CPUInfo(threads=16),
            ram_gb=32.0,
        )
        profile = select_profile(hw)
        # 17 GB iGPU → 13 GB usable → standard (≥8 GB)
        assert profile.name == "standard"

    def test_select_profile_workstation(self):
        """Large GPU should get workstation profile."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="nvidia", total_gb=24.0, is_igpu=False),
            cpu=CPUInfo(threads=32),
            ram_gb=64.0,
        )
        profile = select_profile(hw)
        assert profile.name == "workstation"

    def test_select_profile_high_memory(self):
        """Very large GPU should get high_memory profile."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="nvidia", total_gb=80.0, is_igpu=False),
            cpu=CPUInfo(threads=64),
            ram_gb=128.0,
        )
        profile = select_profile(hw)
        assert profile.name == "high_memory"

    def test_thread_count_capped(self):
        """Profile thread count should not exceed actual CPU threads."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="amd", total_gb=4.0, is_igpu=True),
            cpu=CPUInfo(threads=4),
            ram_gb=8.0,
        )
        profile = select_profile(hw)
        assert profile.max_threads <= 4

    def test_recommend_models(self):
        """Should recommend models across categories."""
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="amd", total_gb=17.0, is_igpu=True),
            cpu=CPUInfo(threads=16),
            ram_gb=32.0,
        )
        recs = recommend_models(hw)
        assert len(recs) >= 3
        categories = [r["category"] for r in recs]
        assert "Coding" in categories
        assert "General chat" in categories
        assert "Reasoning" in categories
