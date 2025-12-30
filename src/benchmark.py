"""Main benchmark orchestrator."""
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import config
from src.dataset import Problem, load_leetcode_dataset, select_problems
from src.ollama_client import OllamaClient, ensure_models_available
from src.executor import execute_code, ExecutionResult


@dataclass
class BenchmarkResult:
    """Result for a single problem-model combination."""
    problem_id: str
    problem_title: str
    difficulty: str
    model: str
    passed: bool
    total_tests: int
    passed_tests: int
    api_time: float
    execution_time: float
    error: Optional[str] = None
    generated_code: Optional[str] = None


def run_benchmark(
    num_problems: int = config.NUM_PROBLEMS,
    models: Optional[list[str]] = None,
    output_dir: Path = Path("results"),
    save_code: bool = True
) -> list[BenchmarkResult]:
    """
    Run the full benchmark.
    
    Args:
        num_problems: Number of problems to benchmark
        models: List of models to test (default: both deepseek-chat and deepseek-coder)
        output_dir: Directory to save results
        save_code: Whether to save generated code in results
    
    Returns:
        List of BenchmarkResult objects
    """
    if models is None:
        models = list(config.MODELS.values())
    
    # Initialize
    print("=" * 60)
    print("LeetCode Model Benchmark")
    print("=" * 60)
    
    # Load dataset
    print("\n📚 Loading dataset...")
    all_problems = load_leetcode_dataset()
    
    # Check for cached problem set (ensures same problems across model runs)
    problem_set_path = output_dir / "problem_set.json"
    if problem_set_path.exists():
        print(f"Loading fixed problem set from {problem_set_path}")
        import json
        with open(problem_set_path) as f:
            problem_ids = set(json.load(f))
        problems = [p for p in all_problems if p.id in problem_ids]
        print(f"Loaded {len(problems)} problems from cached set")
    else:
        problems = select_problems(all_problems, num_problems=num_problems)
        # Save problem IDs for future runs
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(problem_set_path, "w") as f:
            json.dump([p.id for p in problems], f, indent=2)
        print(f"Selected {len(problems)} problems (saved to {problem_set_path})")
    
    # Initialize client
    print("\n🔌 Connecting to Ollama...")
    client = OllamaClient()
    
    # Ensure models are available
    print("\n📦 Checking models...")
    ensure_models_available(models, client)
    
    # Run benchmark
    results: list[BenchmarkResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Running benchmark with models: {models}")
    print("-" * 60)
    
    for problem in tqdm(problems, desc="Problems"):
        for model in models:
            result = benchmark_single(client, problem, model, save_code)
            results.append(result)
            
            # Save intermediate results
            _save_results(results, output_dir / "raw_results.json")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    
    return results


def benchmark_single(
    client: OllamaClient,
    problem: Problem,
    model: str,
    save_code: bool = True
) -> BenchmarkResult:
    """Benchmark a single problem with a single model."""
    
    try:
        # Generate solution
        code, api_time = client.generate_solution(
            problem_description=problem.description,
            starter_code=problem.starter_code,
            model=model
        )
        
        if not code:
            return BenchmarkResult(
                problem_id=str(problem.id),
                problem_title=problem.title,
                difficulty=problem.difficulty,
                model=model,
                passed=False,
                total_tests=len(problem.test_cases),
                passed_tests=0,
                api_time=api_time,
                execution_time=0.0,
                error="No code generated",
                generated_code=code if save_code else None
            )
        
        # Execute and test
        exec_result = execute_code(code, problem.test_cases, problem.starter_code)
        
        return BenchmarkResult(
            problem_id=str(problem.id),
            problem_title=problem.title,
            difficulty=problem.difficulty,
            model=model,
            passed=exec_result.passed,
            total_tests=exec_result.total_tests,
            passed_tests=exec_result.passed_tests,
            api_time=api_time,
            execution_time=exec_result.execution_time,
            error=exec_result.error,
            generated_code=code if save_code else None
        )
        
    except Exception as e:
        return BenchmarkResult(
            problem_id=str(problem.id),
            problem_title=problem.title,
            difficulty=problem.difficulty,
            model=model,
            passed=False,
            total_tests=len(problem.test_cases),
            passed_tests=0,
            api_time=0.0,
            execution_time=0.0,
            error=str(e)[:500]
        )


def _save_results(results: list[BenchmarkResult], path: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "results": [asdict(r) for r in results]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Quick test with 2 problems
    results = run_benchmark(num_problems=2)
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"{status} {r.problem_title} [{r.model}] - {r.passed_tests}/{r.total_tests}")
