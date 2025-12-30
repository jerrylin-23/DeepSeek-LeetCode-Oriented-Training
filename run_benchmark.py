#!/usr/bin/env python3
"""CLI entry point for LeetCode benchmark."""
import argparse
from pathlib import Path

import config
from src.benchmark import run_benchmark
from src.reporter import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs on LeetCode problems using Ollama"
    )
    parser.add_argument(
        "--problems", "-n",
        type=int,
        default=config.NUM_PROBLEMS,
        help=f"Number of problems to benchmark (default: {config.NUM_PROBLEMS})"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Run benchmark for a specific model only (e.g., deepseek-coder:6.7b-instruct)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Don't save generated code in results (saves space)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results"
    )
    
    args = parser.parse_args()
    
    if args.report_only:
        # Just generate report
        results_path = args.output / "raw_results.json"
        if not results_path.exists():
            print(f"Error: No results file found at {results_path}")
            print("Run the benchmark first, then use --report-only")
            return 1
        
        report = generate_report(results_path, args.output / "report.md")
        print(report)
        return 0
    
    # Determine models to run
    models = None
    if args.model:
        # Resolve model name
        if args.model in config.MODELS:
            models = [config.MODELS[args.model]]
        else:
            models = [args.model]
    
    # Run benchmark
    try:
        results = run_benchmark(
            num_problems=args.problems,
            models=models,
            output_dir=args.output,
            save_code=not args.no_code
        )
    except ConnectionError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo start Ollama:")
        print("  1. Install: brew install ollama")
        print("  2. Start: ollama serve")
        print("  3. Pull models: ollama pull deepseek-coder:6.7b-instruct")
        return 1
    
    # Generate report
    print("\n📊 Generating report...")
    report = generate_report(
        args.output / "raw_results.json",
        args.output / "report.md"
    )
    print(report)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    print(f"\n✅ Total passed: {passed}/{len(results)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
