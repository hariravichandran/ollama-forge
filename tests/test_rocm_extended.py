"""Extended tests for ROCm overrides, RDNA4/RDNA3.5 support, and Intel Arc detection."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from forge.hardware.detect import GPUInfo, _detect_intel_gpu_linux
from forge.hardware.rocm import GFX_OVERRIDES, configure_rocm_env


class TestRDNA4Overrides:
    """Tests for RDNA4 GFX override entries."""

    def test_gfx1200_in_overrides(self):
        """gfx1200 (RDNA4 Navi 48) should be in GFX_OVERRIDES."""
        assert "gfx1200" in GFX_OVERRIDES

    def test_gfx1201_in_overrides(self):
        """gfx1201 (RDNA4 Navi 44) should be in GFX_OVERRIDES."""
        assert "gfx1201" in GFX_OVERRIDES

    def test_rdna4_override_value(self):
        """RDNA4 GPUs should use GFX version 12.0.0."""
        assert GFX_OVERRIDES["gfx1200"] == "12.0.0"
        assert GFX_OVERRIDES["gfx1201"] == "12.0.0"


class TestRDNA35Overrides:
    """Tests for RDNA3.5 GFX override entries."""

    def test_gfx1150_in_overrides(self):
        """gfx1150 (RDNA3.5 Strix Point) should be in GFX_OVERRIDES."""
        assert "gfx1150" in GFX_OVERRIDES

    def test_gfx1151_in_overrides(self):
        """gfx1151 (RDNA3.5 Strix Point variant) should be in GFX_OVERRIDES."""
        assert "gfx1151" in GFX_OVERRIDES

    def test_rdna35_override_value(self):
        """RDNA3.5 APUs should map to 11.0.0 (nearest supported target)."""
        assert GFX_OVERRIDES["gfx1150"] == "11.0.0"
        assert GFX_OVERRIDES["gfx1151"] == "11.0.0"


class TestRDNA4ConfigureRocm:
    """Tests for configure_rocm_env with RDNA4 GPU."""

    def _clean_env(self):
        """Remove ROCm env vars for clean test."""
        keys = ["HSA_OVERRIDE_GFX_VERSION", "OLLAMA_FLASH_ATTENTION", "OLLAMA_MAX_LOADED_MODELS"]
        backup = {k: os.environ.pop(k, None) for k in keys}
        return backup

    def _restore_env(self, backup):
        """Restore env vars from backup."""
        for k, v in backup.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_configure_rocm_rdna4_sets_gfx_version(self):
        """configure_rocm_env should set HSA_OVERRIDE_GFX_VERSION=12.0.0 for RDNA4."""
        gpu = GPUInfo(
            vendor="amd", name="Radeon RX 9070 XT", driver="rocm",
            architecture="gfx1200", is_igpu=False, vram_gb=16.0, total_gb=16.0,
        )
        backup = self._clean_env()
        try:
            with patch("forge.hardware.rocm._get_gfx_from_rocminfo", return_value="gfx1200"):
                env_vars = configure_rocm_env(gpu)
            assert env_vars.get("HSA_OVERRIDE_GFX_VERSION") == "12.0.0"
        finally:
            self._restore_env(backup)

    def test_configure_rocm_rdna4_flash_attention(self):
        """configure_rocm_env should enable flash attention for RDNA4."""
        gpu = GPUInfo(
            vendor="amd", name="Radeon RX 9070", driver="rocm",
            architecture="gfx1201", is_igpu=False, vram_gb=12.0, total_gb=12.0,
        )
        backup = self._clean_env()
        try:
            with patch("forge.hardware.rocm._get_gfx_from_rocminfo", return_value="gfx1201"):
                env_vars = configure_rocm_env(gpu)
            assert env_vars.get("OLLAMA_FLASH_ATTENTION") == "1"
        finally:
            self._restore_env(backup)

    def test_configure_rocm_rdna4_max_loaded_models(self):
        """RDNA4 discrete GPU should allow 2 loaded models."""
        gpu = GPUInfo(
            vendor="amd", name="Radeon RX 9070 XT", driver="rocm",
            architecture="gfx1200", is_igpu=False, vram_gb=16.0, total_gb=16.0,
        )
        backup = self._clean_env()
        try:
            with patch("forge.hardware.rocm._get_gfx_from_rocminfo", return_value="gfx1200"):
                env_vars = configure_rocm_env(gpu)
            assert env_vars.get("OLLAMA_MAX_LOADED_MODELS") == "2"
        finally:
            self._restore_env(backup)


class TestIntelArcDetection:
    """Tests for Intel Arc GPU detection and GPUInfo properties."""

    def test_detect_intel_lspci_failure(self):
        """Should return None if lspci fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("forge.hardware.detect.subprocess.run", return_value=mock_result):
            gpu = _detect_intel_gpu_linux()
        assert gpu is None

    def test_detect_intel_lspci_no_gpu_lines(self):
        """Should return None if no VGA/Display lines in lspci."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "00:1f.0 ISA bridge: Intel Corporation Device [8086:7a04]\n"

        with patch("forge.hardware.detect.subprocess.run", return_value=mock_result):
            gpu = _detect_intel_gpu_linux()
        assert gpu is None

    def test_gpuinfo_intel_arc_properties(self):
        """Intel Arc discrete GPU should not be an iGPU and report full VRAM."""
        gpu = GPUInfo(
            vendor="intel", name="Intel Arc A770",
            vram_gb=16.0, total_gb=16.0, driver="xe",
            is_igpu=False,
        )
        assert gpu.vendor == "intel"
        assert gpu.is_igpu is False
        assert gpu.usable_gb == 16.0

    def test_gpuinfo_intel_integrated_properties(self):
        """Intel integrated GPU should be marked as iGPU."""
        gpu = GPUInfo(
            vendor="intel", name="Intel Iris Xe Graphics",
            driver="i915", is_igpu=True,
        )
        assert gpu.vendor == "intel"
        assert gpu.is_igpu is True


class TestExistingOverridesPreserved:
    """Verify that existing RDNA2/RDNA3 overrides are intact after additions."""

    @pytest.mark.parametrize("gfx_id,expected_version", [
        ("gfx1030", "10.3.0"),
        ("gfx1031", "10.3.0"),
        ("gfx1032", "10.3.0"),
        ("gfx1035", "10.3.0"),
        ("gfx1036", "10.3.0"),
        ("gfx1100", "11.0.0"),
        ("gfx1101", "11.0.0"),
        ("gfx1102", "11.0.0"),
        ("gfx1103", "11.0.0"),
    ])
    def test_existing_override(self, gfx_id: str, expected_version: str):
        """Existing RDNA2/RDNA3 overrides must be preserved."""
        assert gfx_id in GFX_OVERRIDES
        assert GFX_OVERRIDES[gfx_id] == expected_version


class TestGFXOverridesCompleteness:
    """Test that all expected GPU architectures are covered."""

    def test_total_override_count(self):
        """Should have entries for all known architectures."""
        # 5 RDNA2 + 4 RDNA3 + 2 RDNA3.5 + 2 RDNA4 = 13
        assert len(GFX_OVERRIDES) == 13

    def test_all_rdna2_covered(self):
        """All RDNA2 GFX IDs should be covered."""
        rdna2 = ["gfx1030", "gfx1031", "gfx1032", "gfx1035", "gfx1036"]
        for gfx in rdna2:
            assert gfx in GFX_OVERRIDES, f"Missing RDNA2 entry: {gfx}"

    def test_all_rdna3_covered(self):
        """All RDNA3 GFX IDs should be covered."""
        rdna3 = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]
        for gfx in rdna3:
            assert gfx in GFX_OVERRIDES, f"Missing RDNA3 entry: {gfx}"
