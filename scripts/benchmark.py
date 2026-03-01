#!/usr/bin/env python3
"""Hardware benchmark for model recommendations.

Usage: python scripts/benchmark.py [--model MODEL]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from forge.hardware import detect_hardware, select_profile
from forge.llm.client import OllamaClient


def main():
    parser = argparse.ArgumentParser(description="Benchmark your hardware for LLM inference")
    parser.add_argument("--model", "-m", default="", help="Model to benchmark")
    args = parser.parse_args()

    hw = detect_hardware()
    profile = select_profile(hw)
    model = args.model or profile.recommended_model

    print(f"Hardware: {hw.gpu.name} ({hw.gpu.total_gb:.1f} GB), {hw.cpu.threads} threads, {hw.ram_gb:.1f} GB RAM")
    print(f"Profile: {profile.name}")
    print(f"Model: {model}")
    print()

    client = OllamaClient(
        model=model,
        num_ctx=profile.num_ctx,
        num_thread=profile.max_threads,
    )

    if not client.is_available():
        print("ERROR: Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    prompts = [
        ("Short generation", "What is 2+2?"),
        ("Code generation", "Write a Python function to reverse a linked list."),
        ("Explanation", "Explain how a hash table works in 3 sentences."),
        ("Creative", "Write a haiku about programming."),
    ]

    print(f"Running {len(prompts)} benchmarks...\n")

    results = []
    for name, prompt in prompts:
        start = time.time()
        result = client.generate(prompt, timeout=120)
        elapsed = time.time() - start
        tokens = result.get("tokens", 0)
        tps = tokens / max(0.01, elapsed)
        results.append((name, tokens, elapsed, tps))
        print(f"  {name:20s} {tokens:4d} tokens  {elapsed:5.1f}s  {tps:5.1f} tok/s")

    avg_tps = sum(r[3] for r in results) / len(results)
    print(f"\n  Average: {avg_tps:.1f} tokens/sec")

    if avg_tps < 3:
        print(f"\n  Recommendation: Use a smaller model ({profile.fallback_model})")
    elif avg_tps < 10:
        print(f"\n  Recommendation: Current model ({model}) is a good fit")
    else:
        print(f"\n  Recommendation: You could try a larger model ({profile.larger_model})")


if __name__ == "__main__":
    main()
