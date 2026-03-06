"""Tests for batch 21 improvements: dataclass __repr__, __post_init__, property docstrings.

Verifies that LLMStats, ExecutionResult, GPUInfo, and HardwareProfile
have proper string representations and validation.
"""

import pytest


class TestLLMStatsRepr:
    """Tests for LLMStats __repr__ and property docstrings."""

    def test_repr_default(self):
        from forge.llm.client import LLMStats
        stats = LLMStats()
        r = repr(stats)
        assert "LLMStats(" in r
        assert "calls=0" in r
        assert "tokens=0" in r
        assert "errors=0" in r

    def test_repr_with_data(self):
        from forge.llm.client import LLMStats
        stats = LLMStats(total_calls=10, total_tokens=500, total_time_s=5.0, errors=1)
        r = repr(stats)
        assert "calls=10" in r
        assert "tokens=500" in r
        assert "errors=1" in r
        assert "avg_time=0.50s" in r

    def test_avg_time_docstring(self):
        from forge.llm.client import LLMStats
        assert LLMStats.avg_time_s.fget.__doc__ is not None

    def test_avg_tokens_per_sec_docstring(self):
        from forge.llm.client import LLMStats
        assert LLMStats.avg_tokens_per_sec.fget.__doc__ is not None


class TestExecutionResultRepr:
    """Tests for ExecutionResult __repr__ and __post_init__."""

    def test_repr_success(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="hello", stderr="", return_code=0, duration_s=1.5)
        r = repr(result)
        assert "OK" in r
        assert "1.5s" in r

    def test_repr_failure(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="err", return_code=1, duration_s=0.5)
        r = repr(result)
        assert "exit=1" in r

    def test_repr_timeout(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=-1, duration_s=30.0, timed_out=True)
        r = repr(result)
        assert "TIMEOUT" in r

    def test_repr_with_memory(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=1.0, peak_memory_mb=128.5)
        r = repr(result)
        assert "128MB" in r or "129MB" in r

    def test_post_init_clamps_duration(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=-1.0)
        assert result.duration_s == 0.0

    def test_post_init_clamps_memory(self):
        from forge.tools.sandbox import ExecutionResult
        result = ExecutionResult(stdout="", stderr="", return_code=0, duration_s=0.0, peak_memory_mb=-5.0)
        assert result.peak_memory_mb == 0.0

    def test_success_property_docstring(self):
        from forge.tools.sandbox import ExecutionResult
        assert ExecutionResult.success.fget.__doc__ is not None


class TestGPUInfoRepr:
    """Tests for GPUInfo __repr__ and __post_init__."""

    def test_repr_default(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo()
        r = repr(gpu)
        assert "GPUInfo(" in r
        assert "none" in r
        assert "Unknown" in r

    def test_repr_with_data(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(vendor="amd", name="Radeon 680M", total_gb=12.0, driver="rocm", is_igpu=True)
        r = repr(gpu)
        assert "amd" in r
        assert "Radeon 680M" in r
        assert "12.0GB" in r
        assert "iGPU" in r
        assert "rocm" in r

    def test_post_init_clamps_negative_vram(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(vram_gb=-1.0)
        assert gpu.vram_gb == 0.0

    def test_post_init_clamps_negative_gtt(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(gtt_gb=-2.0)
        assert gpu.gtt_gb == 0.0

    def test_post_init_clamps_negative_total(self):
        from forge.hardware.detect import GPUInfo
        gpu = GPUInfo(total_gb=-5.0)
        assert gpu.total_gb == 0.0


class TestHardwareProfileRepr:
    """Tests for HardwareProfile __repr__ and __post_init__."""

    def test_repr(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="standard", min_gpu_gb=8, recommended_model="qwen2.5-coder:7b",
            fallback_model="qwen2.5-coder:3b", larger_model="qwen2.5-coder:14b",
            num_ctx=8192, num_batch=2048, max_threads=14, description="test",
        )
        r = repr(profile)
        assert "standard" in r
        assert "qwen2.5-coder:7b" in r
        assert "ctx=8192" in r

    def test_repr_cpu_only(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="compact", min_gpu_gb=0, recommended_model="qwen2.5-coder:3b",
            fallback_model="qwen2.5-coder:1.5b", larger_model="qwen2.5-coder:7b",
            num_ctx=4096, num_batch=512, max_threads=8, description="test",
            is_cpu_only=True,
        )
        r = repr(profile)
        assert "CPU-only" in r

    def test_post_init_clamps_min_gpu(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="test", min_gpu_gb=-5, recommended_model="m", fallback_model="m",
            larger_model="m", num_ctx=4096, num_batch=1024, max_threads=8,
            description="test",
        )
        assert profile.min_gpu_gb == 0.0

    def test_post_init_clamps_threads(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="test", min_gpu_gb=0, recommended_model="m", fallback_model="m",
            larger_model="m", num_ctx=4096, num_batch=1024, max_threads=0,
            description="test",
        )
        assert profile.max_threads >= 1

    def test_post_init_clamps_num_ctx(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="test", min_gpu_gb=0, recommended_model="m", fallback_model="m",
            larger_model="m", num_ctx=10, num_batch=1024, max_threads=8,
            description="test",
        )
        assert profile.num_ctx >= 128

    def test_post_init_clamps_num_batch(self):
        from forge.hardware.profiles import HardwareProfile
        profile = HardwareProfile(
            name="test", min_gpu_gb=0, recommended_model="m", fallback_model="m",
            larger_model="m", num_ctx=4096, num_batch=0, max_threads=8,
            description="test",
        )
        assert profile.num_batch >= 1


class TestBatch21Integration:
    """Integration tests verifying all modified modules import correctly."""

    def test_llm_stats_imports(self):
        from forge.llm.client import LLMStats

    def test_execution_result_imports(self):
        from forge.tools.sandbox import ExecutionResult

    def test_gpu_info_imports(self):
        from forge.hardware.detect import GPUInfo

    def test_hardware_profile_imports(self):
        from forge.hardware.profiles import HardwareProfile
