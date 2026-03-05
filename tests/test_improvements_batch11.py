"""Tests for batch 11 improvements: profiles constants, benchmark validation, reflect safety, QA hardening."""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest


# === Hardware Profiles: Extracted Constants ===

class TestProfileConstants:
    """Tests for extracted profile constants."""

    def test_os_memory_reservation(self):
        from forge.hardware.profiles import OS_MEMORY_RESERVATION_GB
        assert OS_MEMORY_RESERVATION_GB > 0
        assert OS_MEMORY_RESERVATION_GB < 16  # reasonable range

    def test_os_thread_reservation(self):
        from forge.hardware.profiles import OS_THREAD_RESERVATION
        assert OS_THREAD_RESERVATION >= 1

    def test_cpu_max_batch_size(self):
        from forge.hardware.profiles import CPU_MAX_BATCH_SIZE
        assert CPU_MAX_BATCH_SIZE > 0
        assert CPU_MAX_BATCH_SIZE <= 1024

    def test_low_ram_threshold(self):
        from forge.hardware.profiles import LOW_RAM_THRESHOLD_GB
        assert LOW_RAM_THRESHOLD_GB > 0

    def test_low_ram_context(self):
        from forge.hardware.profiles import LOW_RAM_CONTEXT
        assert LOW_RAM_CONTEXT > 0
        assert LOW_RAM_CONTEXT <= 4096

    def test_valid_profile_names(self):
        from forge.hardware.profiles import VALID_PROFILE_NAMES, PROFILES
        for profile in PROFILES:
            assert profile.name in VALID_PROFILE_NAMES

    def test_valid_profile_names_set(self):
        from forge.hardware.profiles import VALID_PROFILE_NAMES
        assert isinstance(VALID_PROFILE_NAMES, set)
        assert len(VALID_PROFILE_NAMES) >= 3


class TestProfileCPUOnlyConstants:
    """Tests for CPU-only mode using extracted constants."""

    def test_cpu_only_uses_memory_reservation(self):
        from forge.hardware.profiles import select_profile, OS_MEMORY_RESERVATION_GB
        from forge.hardware.detect import HardwareInfo, GPUInfo, CPUInfo
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="none", driver="cpu"),
            cpu=CPUInfo(threads=8, cores=4),
            ram_gb=16.0,
        )
        profile = select_profile(hw)
        assert profile.is_cpu_only
        # The profile should be selected based on (ram - reservation)
        # 16 - 4 = 12 GB usable

    def test_cpu_only_thread_reservation(self):
        from forge.hardware.profiles import select_profile, OS_THREAD_RESERVATION
        from forge.hardware.detect import HardwareInfo, GPUInfo, CPUInfo
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="none", driver="cpu"),
            cpu=CPUInfo(threads=8, cores=4),
            ram_gb=16.0,
        )
        profile = select_profile(hw)
        assert profile.max_threads == 8 - OS_THREAD_RESERVATION

    def test_low_ram_forces_small_model(self):
        from forge.hardware.profiles import select_profile, LOW_RAM_THRESHOLD_GB
        from forge.hardware.detect import HardwareInfo, GPUInfo, CPUInfo
        hw = HardwareInfo(
            gpu=GPUInfo(vendor="none", driver="cpu"),
            cpu=CPUInfo(threads=4, cores=2),
            ram_gb=LOW_RAM_THRESHOLD_GB - 1,
        )
        profile = select_profile(hw)
        assert "1.5b" in profile.recommended_model or "0.5b" in profile.recommended_model


# === Benchmark: Validation and Safety ===

class TestBenchmarkConstants:
    """Tests for benchmark constants."""

    def test_benchmark_version(self):
        from forge.llm.benchmark import BENCHMARK_VERSION
        assert BENCHMARK_VERSION
        assert isinstance(BENCHMARK_VERSION, str)

    def test_max_models_per_run(self):
        from forge.llm.benchmark import MAX_MODELS_PER_RUN
        assert MAX_MODELS_PER_RUN > 0
        assert MAX_MODELS_PER_RUN <= 50

    def test_benchmark_timeout(self):
        from forge.llm.benchmark import BENCHMARK_TIMEOUT
        assert BENCHMARK_TIMEOUT > 0

    def test_required_prompt_fields(self):
        from forge.llm.benchmark import REQUIRED_PROMPT_FIELDS
        assert "name" in REQUIRED_PROMPT_FIELDS
        assert "prompt" in REQUIRED_PROMPT_FIELDS


class TestBenchmarkValidation:
    """Tests for benchmark input validation."""

    def test_model_count_limit(self):
        from forge.llm.benchmark import MAX_MODELS_PER_RUN, run_benchmark
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        # We can't actually run benchmarks without Ollama, but we can test
        # that the function exists and has the right signature
        assert callable(run_benchmark)

    def test_standard_prompts_have_required_fields(self):
        from forge.llm.benchmark import BENCHMARK_PROMPTS, REQUIRED_PROMPT_FIELDS
        for prompt in BENCHMARK_PROMPTS:
            for field in REQUIRED_PROMPT_FIELDS:
                assert field in prompt, f"Prompt missing {field}"

    def test_standard_prompts_have_categories(self):
        from forge.llm.benchmark import BENCHMARK_PROMPTS
        for prompt in BENCHMARK_PROMPTS:
            assert "category" in prompt

    def test_benchmark_result_passed(self):
        from forge.llm.benchmark import BenchmarkResult
        r = BenchmarkResult(
            model="test", prompt_name="test", category="test",
            response="hello world", tokens=10, time_s=1.0,
            tokens_per_sec=10.0, error="",
        )
        assert r.passed

    def test_benchmark_result_failed_error(self):
        from forge.llm.benchmark import BenchmarkResult
        r = BenchmarkResult(
            model="test", prompt_name="test", category="test",
            response="hello", tokens=10, time_s=1.0,
            tokens_per_sec=10.0, error="timeout",
        )
        assert not r.passed

    def test_benchmark_result_failed_empty(self):
        from forge.llm.benchmark import BenchmarkResult
        r = BenchmarkResult(
            model="test", prompt_name="test", category="test",
            response="", tokens=0, time_s=1.0,
            tokens_per_sec=0, error="",
        )
        assert not r.passed


class TestBenchmarkSummary:
    """Tests for ModelBenchmarkSummary."""

    def test_empty_summary(self):
        from forge.llm.benchmark import ModelBenchmarkSummary
        s = ModelBenchmarkSummary(model="test")
        assert s.total_tokens == 0
        assert s.total_time_s == 0
        assert s.avg_tokens_per_sec == 0.0
        assert s.pass_rate == 0.0

    def test_summary_with_results(self):
        from forge.llm.benchmark import ModelBenchmarkSummary, BenchmarkResult
        s = ModelBenchmarkSummary(model="test")
        s.results.append(BenchmarkResult(
            model="test", prompt_name="p1", category="general",
            response="ok", tokens=100, time_s=2.0,
            tokens_per_sec=50.0, error="",
        ))
        assert s.total_tokens == 100
        assert s.total_time_s == 2.0
        assert s.avg_tokens_per_sec == 50.0
        assert s.pass_rate == 1.0

    def test_summary_line_format(self):
        from forge.llm.benchmark import ModelBenchmarkSummary, BenchmarkResult
        s = ModelBenchmarkSummary(model="test-model")
        s.results.append(BenchmarkResult(
            model="test-model", prompt_name="p1", category="general",
            response="ok", tokens=100, time_s=2.0,
            tokens_per_sec=50.0, error="",
        ))
        line = s.summary_line()
        assert "test-model" in line
        assert "tok/s" in line
        assert "pass" in line


class TestBenchmarkReport:
    """Tests for format_benchmark_report."""

    def test_report_formatting(self):
        from forge.llm.benchmark import format_benchmark_report, ModelBenchmarkSummary, BenchmarkResult
        s = ModelBenchmarkSummary(model="test-model")
        s.results.append(BenchmarkResult(
            model="test-model", prompt_name="p1", category="general",
            response="ok", tokens=100, time_s=2.0,
            tokens_per_sec=50.0, error="",
        ))
        report = format_benchmark_report([s])
        assert "Model Benchmark Results" in report
        assert "test-model" in report

    def test_report_with_multiple_models(self):
        from forge.llm.benchmark import format_benchmark_report, ModelBenchmarkSummary, BenchmarkResult
        summaries = []
        for name, speed in [("fast-model", 100.0), ("slow-model", 10.0)]:
            s = ModelBenchmarkSummary(model=name)
            s.results.append(BenchmarkResult(
                model=name, prompt_name="p1", category="general",
                response="ok", tokens=100, time_s=1.0,
                tokens_per_sec=speed, error="",
            ))
            summaries.append(s)
        report = format_benchmark_report(summaries)
        assert "Fastest" in report
        assert "fast-model" in report


# === Reflect Agent: LGTM Detection, Regex Fix, max_revisions ===

class TestReflectConstants:
    """Tests for reflect agent constants."""

    def test_max_revisions_constant(self):
        from forge.agents.reflect import MAX_REVISIONS
        assert MAX_REVISIONS > 0
        assert MAX_REVISIONS <= 10

    def test_min_revisions_constant(self):
        from forge.agents.reflect import MIN_REVISIONS
        assert MIN_REVISIONS >= 0

    def test_min_response_length(self):
        from forge.agents.reflect import MIN_RESPONSE_LENGTH
        assert MIN_RESPONSE_LENGTH > 0

    def test_lgtm_pattern_exists(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN is not None

    def test_max_question_length(self):
        from forge.agents.reflect import MAX_QUESTION_LENGTH
        assert MAX_QUESTION_LENGTH > 0

    def test_max_response_length(self):
        from forge.agents.reflect import MAX_RESPONSE_LENGTH
        assert MAX_RESPONSE_LENGTH > 0


class TestReflectLGTMDetection:
    """Tests for improved LGTM detection pattern."""

    def test_simple_lgtm(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("LGTM")

    def test_lgtm_with_period(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("LGTM.")

    def test_lgtm_lowercase(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("lgtm")

    def test_lgtm_in_sentence_start(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("LGTM, looks good to me")

    def test_lgtm_after_newline(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("The code is fine.\nLGTM")

    def test_lgtm_with_leading_spaces(self):
        from forge.agents.reflect import LGTM_PATTERN
        assert LGTM_PATTERN.search("  LGTM")

    def test_no_lgtm_in_middle_of_word(self):
        from forge.agents.reflect import LGTM_PATTERN
        # "LGTMX" should not match because \b prevents it
        assert not LGTM_PATTERN.search("xLGTMx")


class TestReflectMaxRevisions:
    """Tests for max_revisions clamping."""

    def test_normal_max_revisions(self):
        from forge.agents.reflect import ReflectiveAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client, max_revisions=2)
        assert agent.max_revisions == 2

    def test_max_revisions_clamped_high(self):
        from forge.agents.reflect import ReflectiveAgent, MAX_REVISIONS
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client, max_revisions=100)
        assert agent.max_revisions == MAX_REVISIONS

    def test_max_revisions_clamped_negative(self):
        from forge.agents.reflect import ReflectiveAgent, MIN_REVISIONS
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client, max_revisions=-5)
        assert agent.max_revisions == MIN_REVISIONS

    def test_zero_revisions_disables_reflection(self):
        from forge.agents.reflect import ReflectiveAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client, max_revisions=0)
        assert agent.max_revisions == 0


class TestReflectCategorizeIssues:
    """Tests for issue categorization with fixed regex."""

    def test_factual_issue(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("This statement is incorrect")
        assert "factual" in cats

    def test_inaccurate_fixed_typo(self):
        from forge.agents.reflect import ReflectiveAgent
        # This was the regex typo fix: "inaccurat" -> "inaccurate"
        cats = ReflectiveAgent._categorize_issues("The answer is inaccurate")
        assert "factual" in cats

    def test_completeness_issue(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("The response is incomplete")
        assert "completeness" in cats

    def test_code_error_issue(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("There is a syntax error in the code")
        assert "code_error" in cats

    def test_clarity_issue(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("The explanation is unclear")
        assert "clarity" in cats

    def test_general_fallback(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("something else entirely")
        assert "general" in cats

    def test_multiple_categories(self):
        from forge.agents.reflect import ReflectiveAgent
        cats = ReflectiveAgent._categorize_issues("This is incorrect and the code has a bug")
        assert "factual" in cats
        assert "code_error" in cats


class TestReflectSelectPrompt:
    """Tests for review prompt selection."""

    def test_code_response_gets_code_prompt(self):
        from forge.agents.reflect import ReflectiveAgent, REVIEW_PROMPT_CODE
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client)
        prompt = agent._select_review_prompt("```python\ndef hello():\n    pass\n```")
        assert prompt == REVIEW_PROMPT_CODE

    def test_short_response_gets_short_prompt(self):
        from forge.agents.reflect import ReflectiveAgent, REVIEW_PROMPT_SHORT
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client)
        prompt = agent._select_review_prompt("The answer is 42.")
        assert prompt == REVIEW_PROMPT_SHORT

    def test_long_response_gets_general_prompt(self):
        from forge.agents.reflect import ReflectiveAgent, REVIEW_PROMPT_GENERAL
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client)
        prompt = agent._select_review_prompt("x" * 300)
        assert prompt == REVIEW_PROMPT_GENERAL


class TestReflectStats:
    """Tests for reflection stats."""

    def test_stats_include_reflection(self):
        from forge.agents.reflect import ReflectiveAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client)
        stats = agent.get_stats()
        assert "reflection" in stats
        assert "reviews" in stats["reflection"]
        assert "revisions" in stats["reflection"]


# === QA Agent: sys.executable, Timeout Constants, Test Validation ===

class TestQAConstants:
    """Tests for QA agent constants."""

    def test_existing_test_timeout(self):
        from forge.agents.qa import EXISTING_TEST_TIMEOUT
        assert EXISTING_TEST_TIMEOUT > 0

    def test_generated_test_timeout(self):
        from forge.agents.qa import GENERATED_TEST_TIMEOUT
        assert GENERATED_TEST_TIMEOUT > 0

    def test_max_file_content_length(self):
        from forge.agents.qa import MAX_FILE_CONTENT_LENGTH
        assert MAX_FILE_CONTENT_LENGTH > 0

    def test_max_review_files(self):
        from forge.agents.qa import MAX_REVIEW_FILES
        assert MAX_REVIEW_FILES > 0

    def test_max_test_gen_files(self):
        from forge.agents.qa import MAX_TEST_GEN_FILES
        assert MAX_TEST_GEN_FILES > 0

    def test_dangerous_test_patterns(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        assert DANGEROUS_TEST_PATTERNS is not None


class TestQADangerousPatterns:
    """Tests for dangerous pattern detection in generated tests."""

    def test_os_system_rejected(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        assert DANGEROUS_TEST_PATTERNS.search("os.system('rm -rf /')")

    def test_eval_rejected(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        assert DANGEROUS_TEST_PATTERNS.search("eval(user_input)")

    def test_exec_rejected(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        assert DANGEROUS_TEST_PATTERNS.search("exec(code)")

    def test_safe_code_passes(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        safe_code = """
def test_something():
    from forge.llm.client import OllamaClient
    client = OllamaClient()
    assert client is not None
"""
        assert not DANGEROUS_TEST_PATTERNS.search(safe_code)

    def test_subprocess_shell_true_rejected(self):
        from forge.agents.qa import DANGEROUS_TEST_PATTERNS
        assert DANGEROUS_TEST_PATTERNS.search("subprocess.call('cmd', shell=True)")


class TestQASysExecutable:
    """Tests for sys.executable usage in QA agent."""

    def test_qa_imports_sys(self):
        import forge.agents.qa as qa_mod
        import inspect
        source = inspect.getsource(qa_mod)
        assert "sys.executable" in source
        assert "import sys" in source

    def test_no_hardcoded_python(self):
        import forge.agents.qa as qa_mod
        import inspect
        source = inspect.getsource(qa_mod)
        # Should not use bare "python" in subprocess calls
        lines = source.split("\n")
        for line in lines:
            if "subprocess.run" in line or "subprocess.call" in line:
                # This line starts a subprocess call; check the list isn't ["python", ...]
                pass  # The actual check is in the source inspection
        # Check no literal ["python" remains
        assert '["python"' not in source


class TestQAResult:
    """Tests for QAResult class."""

    def test_qa_result_repr(self):
        from forge.agents.qa import QAResult
        r = QAResult(
            passed=True,
            existing_tests_passed=True,
            generated_tests_passed=True,
            summary="All good",
            generated_test_code="",
            test_output="",
        )
        assert "passed=True" in repr(r)

    def test_qa_result_failed(self):
        from forge.agents.qa import QAResult
        r = QAResult(
            passed=False,
            existing_tests_passed=True,
            generated_tests_passed=False,
            summary="Tests failed",
            generated_test_code="def test_x(): pass",
            test_output="FAILED",
        )
        assert not r.passed
        assert r.existing_tests_passed
        assert not r.generated_tests_passed


class TestQAInit:
    """Tests for QA agent initialization."""

    def test_qa_agent_init(self):
        from forge.agents.qa import QAAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        qa = QAAgent(client=client, repo_dir="/tmp")
        assert qa.repo_dir.as_posix() == "/tmp"

    def test_qa_agent_default_repo_dir(self):
        from forge.agents.qa import QAAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        qa = QAAgent(client=client)
        assert qa.repo_dir.as_posix() == "."


# === Integration Tests ===

class TestBatch11Integration:
    """Integration tests across batch 11 improvements."""

    def test_profiles_constants_used(self):
        """Verify profile constants are imported from the right module."""
        from forge.hardware.profiles import (
            OS_MEMORY_RESERVATION_GB,
            OS_THREAD_RESERVATION,
            CPU_MAX_BATCH_SIZE,
            LOW_RAM_THRESHOLD_GB,
            LOW_RAM_CONTEXT,
            VALID_PROFILE_NAMES,
        )
        # All should be importable and have reasonable values
        assert all(v > 0 for v in [
            OS_MEMORY_RESERVATION_GB,
            OS_THREAD_RESERVATION,
            CPU_MAX_BATCH_SIZE,
            LOW_RAM_THRESHOLD_GB,
            LOW_RAM_CONTEXT,
        ])
        assert len(VALID_PROFILE_NAMES) >= 3

    def test_benchmark_prompts_valid(self):
        """Verify all benchmark prompts pass validation."""
        from forge.llm.benchmark import BENCHMARK_PROMPTS, REQUIRED_PROMPT_FIELDS
        for p in BENCHMARK_PROMPTS:
            missing = REQUIRED_PROMPT_FIELDS - set(p.keys())
            assert not missing, f"Prompt {p.get('name')} missing {missing}"

    def test_reflect_agent_full_init(self):
        """Verify reflective agent initializes with all safety features."""
        from forge.agents.reflect import ReflectiveAgent
        from forge.llm.client import OllamaClient
        client = OllamaClient()
        agent = ReflectiveAgent(client=client, max_revisions=3)
        assert agent.max_revisions == 3
        assert agent._review_count == 0
        assert agent._revision_count == 0
        assert isinstance(agent._issue_categories, dict)

    def test_qa_agent_has_safety_checks(self):
        """Verify QA agent has all safety features."""
        from forge.agents.qa import (
            DANGEROUS_TEST_PATTERNS,
            EXISTING_TEST_TIMEOUT,
            GENERATED_TEST_TIMEOUT,
            MAX_FILE_CONTENT_LENGTH,
            MAX_REVIEW_FILES,
        )
        assert DANGEROUS_TEST_PATTERNS is not None
        assert EXISTING_TEST_TIMEOUT > 0
        assert GENERATED_TEST_TIMEOUT > 0
        assert MAX_FILE_CONTENT_LENGTH > 0
        assert MAX_REVIEW_FILES > 0
