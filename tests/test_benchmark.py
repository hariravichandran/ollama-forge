"""Tests for model benchmarking module."""

from forge.llm.benchmark import (
    BenchmarkResult,
    ModelBenchmarkSummary,
    BENCHMARK_PROMPTS,
    format_benchmark_report,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_passed_when_response_exists(self):
        r = BenchmarkResult(
            model="test:7b",
            prompt_name="test",
            category="general",
            response="Hello",
            tokens=10,
            time_s=1.0,
            tokens_per_sec=10.0,
        )
        assert r.passed is True

    def test_failed_when_error(self):
        r = BenchmarkResult(
            model="test:7b",
            prompt_name="test",
            category="general",
            response="",
            tokens=0,
            time_s=0,
            tokens_per_sec=0,
            error="timeout",
        )
        assert r.passed is False

    def test_failed_when_empty_response(self):
        r = BenchmarkResult(
            model="test:7b",
            prompt_name="test",
            category="general",
            response="",
            tokens=0,
            time_s=1.0,
            tokens_per_sec=0,
        )
        assert r.passed is False


class TestModelBenchmarkSummary:
    """Tests for ModelBenchmarkSummary."""

    def test_empty_summary(self):
        s = ModelBenchmarkSummary(model="test:7b")
        assert s.total_tokens == 0
        assert s.total_time_s == 0.0
        assert s.avg_tokens_per_sec == 0.0
        assert s.pass_rate == 0.0
        assert s.avg_latency_s == 0.0

    def test_summary_with_results(self):
        s = ModelBenchmarkSummary(
            model="test:7b",
            results=[
                BenchmarkResult("test:7b", "p1", "general", "hello", 100, 2.0, 50.0),
                BenchmarkResult("test:7b", "p2", "coding", "world", 200, 3.0, 66.7),
            ],
        )
        assert s.total_tokens == 300
        assert s.total_time_s == 5.0
        assert s.avg_tokens_per_sec == 60.0
        assert s.pass_rate == 1.0
        assert s.avg_latency_s == 2.5

    def test_pass_rate_with_failure(self):
        s = ModelBenchmarkSummary(
            model="test:7b",
            results=[
                BenchmarkResult("test:7b", "p1", "general", "hello", 100, 2.0, 50.0),
                BenchmarkResult("test:7b", "p2", "coding", "", 0, 1.0, 0, error="timeout"),
            ],
        )
        assert s.pass_rate == 0.5

    def test_summary_line(self):
        s = ModelBenchmarkSummary(
            model="test:7b",
            results=[
                BenchmarkResult("test:7b", "p1", "general", "hello", 100, 2.0, 50.0),
            ],
        )
        line = s.summary_line()
        assert "test:7b" in line
        assert "tok/s" in line


class TestBenchmarkPrompts:
    """Tests for benchmark prompt set."""

    def test_prompts_exist(self):
        assert len(BENCHMARK_PROMPTS) >= 3

    def test_prompts_have_required_fields(self):
        for p in BENCHMARK_PROMPTS:
            assert "name" in p
            assert "prompt" in p
            assert "category" in p

    def test_prompts_cover_categories(self):
        categories = {p["category"] for p in BENCHMARK_PROMPTS}
        assert "general" in categories
        assert "coding" in categories
        assert "reasoning" in categories


class TestFormatReport:
    """Tests for benchmark report formatting."""

    def test_format_single_model(self):
        s = ModelBenchmarkSummary(
            model="test:7b",
            results=[
                BenchmarkResult("test:7b", "simple_qa", "general", "answer", 20, 1.0, 20.0),
            ],
        )
        report = format_benchmark_report([s])
        assert "test:7b" in report
        assert "Benchmark" in report

    def test_format_multiple_models_shows_fastest(self):
        summaries = [
            ModelBenchmarkSummary(
                model="slow:7b",
                results=[BenchmarkResult("slow:7b", "p1", "general", "a", 10, 5.0, 2.0)],
            ),
            ModelBenchmarkSummary(
                model="fast:3b",
                results=[BenchmarkResult("fast:3b", "p1", "general", "b", 10, 1.0, 10.0)],
            ),
        ]
        report = format_benchmark_report(summaries)
        assert "Fastest" in report
        assert "fast:3b" in report
