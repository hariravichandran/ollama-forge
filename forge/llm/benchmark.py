"""Model benchmarking: measure throughput, latency, and quality.

Runs standardized prompts against one or more Ollama models and reports
performance metrics. Useful for choosing the best model for your hardware.

Usage:
    from forge.llm.benchmark import run_benchmark
    results = run_benchmark(client, models=["qwen2.5-coder:7b", "llama3.2:3b"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forge.llm.client import OllamaClient
from forge.utils.logging import get_logger

log = get_logger("llm.benchmark")

# Standardized prompts across difficulty levels
BENCHMARK_PROMPTS = [
    {
        "name": "simple_qa",
        "category": "general",
        "prompt": "What are the three states of matter? Answer in one sentence.",
        "expected_tokens": 30,
    },
    {
        "name": "code_generation",
        "category": "coding",
        "prompt": "Write a Python function that checks if a string is a palindrome. Include a docstring.",
        "expected_tokens": 100,
    },
    {
        "name": "reasoning",
        "category": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
        "expected_tokens": 80,
    },
    {
        "name": "summarization",
        "category": "general",
        "prompt": "Explain the difference between TCP and UDP in networking. Be concise — 3 bullet points maximum.",
        "expected_tokens": 60,
    },
    {
        "name": "json_output",
        "category": "structured",
        "prompt": 'Return a JSON object with keys "name", "language", and "purpose" describing Python. Output ONLY valid JSON.',
        "expected_tokens": 40,
    },
]


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single prompt against a model."""

    model: str
    prompt_name: str
    category: str
    response: str
    tokens: int
    time_s: float
    tokens_per_sec: float
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and len(self.response) > 0


@dataclass
class ModelBenchmarkSummary:
    """Aggregated benchmark results for a single model."""

    model: str
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def total_time_s(self) -> float:
        return sum(r.time_s for r in self.results)

    @property
    def avg_tokens_per_sec(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return self.total_tokens / self.total_time_s

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def avg_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return self.total_time_s / len(self.results)

    def summary_line(self) -> str:
        """One-line summary for display."""
        return (
            f"{self.model:30s} | "
            f"{self.avg_tokens_per_sec:6.1f} tok/s | "
            f"{self.avg_latency_s:5.1f}s avg | "
            f"{self.total_tokens:4d} tokens | "
            f"{self.pass_rate * 100:5.1f}% pass"
        )


def run_benchmark(
    client: OllamaClient,
    models: list[str] | None = None,
    prompts: list[dict] | None = None,
    warmup: bool = True,
    progress_cb=None,
) -> list[ModelBenchmarkSummary]:
    """Run benchmarks across one or more models.

    Args:
        client: Ollama client instance.
        models: Models to benchmark. If None, uses the client's current model.
        prompts: Custom prompts. If None, uses the standard benchmark set.
        warmup: Whether to send a warmup request first.
        progress_cb: Called with (model, prompt_name, result_or_none) for each step.

    Returns:
        List of ModelBenchmarkSummary, one per model.
    """
    if models is None:
        models = [client.model]
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    summaries = []

    for model in models:
        log.info("Benchmarking model: %s", model)
        summary = ModelBenchmarkSummary(model=model)

        if warmup:
            log.info("Warming up %s...", model)
            client.warmup()

        for prompt_info in prompts:
            name = prompt_info["name"]
            category = prompt_info.get("category", "general")
            prompt_text = prompt_info["prompt"]

            if progress_cb:
                progress_cb(model, name, None)

            start = time.time()
            result = client.generate(
                prompt=prompt_text,
                model=model,
                timeout=120,
            )
            elapsed = time.time() - start

            bench_result = BenchmarkResult(
                model=model,
                prompt_name=name,
                category=category,
                response=result.get("response", ""),
                tokens=result.get("tokens", 0),
                time_s=round(elapsed, 2),
                tokens_per_sec=round(result.get("tokens_per_sec", 0), 1),
                error=result.get("error", ""),
            )
            summary.results.append(bench_result)

            if progress_cb:
                progress_cb(model, name, bench_result)

            log.info(
                "  %s: %d tokens in %.1fs (%.1f tok/s)%s",
                name,
                bench_result.tokens,
                bench_result.time_s,
                bench_result.tokens_per_sec,
                f" ERROR: {bench_result.error}" if bench_result.error else "",
            )

        summaries.append(summary)

    return summaries


def format_benchmark_report(summaries: list[ModelBenchmarkSummary]) -> str:
    """Format benchmark results as a readable report."""
    lines = [
        "Model Benchmark Results",
        "=" * 75,
        "",
        f"{'Model':30s} | {'Speed':>10s} | {'Latency':>7s} | {'Tokens':>7s} | {'Pass':>7s}",
        "-" * 75,
    ]

    for s in summaries:
        lines.append(s.summary_line())

    lines.append("-" * 75)
    lines.append("")

    # Detailed results per model
    for s in summaries:
        lines.append(f"\n{s.model}")
        lines.append("-" * 50)
        for r in s.results:
            status = "PASS" if r.passed else f"FAIL: {r.error[:30]}"
            lines.append(
                f"  {r.prompt_name:20s} | {r.tokens:4d} tok | "
                f"{r.time_s:5.1f}s | {r.tokens_per_sec:6.1f} tok/s | {status}"
            )

    # Winner
    if len(summaries) > 1:
        best = max(summaries, key=lambda s: s.avg_tokens_per_sec)
        lines.append(f"\nFastest: {best.model} ({best.avg_tokens_per_sec:.1f} tok/s)")

    return "\n".join(lines)
