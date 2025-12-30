"""Generate reports from benchmark results."""
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tabulate import tabulate


@dataclass
class ModelStats:
    """Statistics for a single model."""
    model: str
    total_problems: int
    passed: int
    failed: int
    pass_rate: float
    avg_api_time: float
    avg_exec_time: float
    by_difficulty: dict


def load_results(path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("results", [])


def compute_stats(results: list[dict]) -> dict[str, ModelStats]:
    """Compute statistics for each model."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    
    stats = {}
    for model, model_results in by_model.items():
        passed = sum(1 for r in model_results if r["passed"])
        failed = len(model_results) - passed
        
        # By difficulty
        by_diff = defaultdict(lambda: {"passed": 0, "total": 0})
        for r in model_results:
            diff = r.get("difficulty", "Unknown")
            by_diff[diff]["total"] += 1
            if r["passed"]:
                by_diff[diff]["passed"] += 1
        
        # Compute averages
        api_times = [r["api_time"] for r in model_results if r["api_time"] > 0]
        exec_times = [r["execution_time"] for r in model_results if r["execution_time"] > 0]
        
        stats[model] = ModelStats(
            model=model,
            total_problems=len(model_results),
            passed=passed,
            failed=failed,
            pass_rate=passed / len(model_results) * 100 if model_results else 0,
            avg_api_time=sum(api_times) / len(api_times) if api_times else 0,
            avg_exec_time=sum(exec_times) / len(exec_times) if exec_times else 0,
            by_difficulty=dict(by_diff)
        )
    
    return stats


def generate_report(
    results_path: Path,
    output_path: Optional[Path] = None
) -> str:
    """Generate a markdown report from benchmark results."""
    
    results = load_results(results_path)
    stats = compute_stats(results)
    
    report_lines = []
    report_lines.append("# LeetCode Benchmark Results\n")
    report_lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Summary table
    report_lines.append("## Summary\n")
    summary_table = []
    for model, s in stats.items():
        summary_table.append([
            model,
            s.total_problems,
            s.passed,
            s.failed,
            f"{s.pass_rate:.1f}%",
            f"{s.avg_api_time:.2f}s",
            f"{s.avg_exec_time:.3f}s"
        ])
    
    headers = ["Model", "Problems", "Passed", "Failed", "Pass Rate", "Avg API Time", "Avg Exec Time"]
    report_lines.append(tabulate(summary_table, headers=headers, tablefmt="github"))
    report_lines.append("\n")
    
    # By difficulty
    report_lines.append("## Results by Difficulty\n")
    for model, s in stats.items():
        report_lines.append(f"### {model}\n")
        diff_table = []
        for diff in ["Easy", "Medium", "Hard"]:
            if diff in s.by_difficulty:
                d = s.by_difficulty[diff]
                rate = d["passed"] / d["total"] * 100 if d["total"] > 0 else 0
                diff_table.append([diff, d["total"], d["passed"], f"{rate:.1f}%"])
        
        if diff_table:
            report_lines.append(tabulate(diff_table, headers=["Difficulty", "Total", "Passed", "Rate"], tablefmt="github"))
            report_lines.append("\n")
    
    # Comparison
    if len(stats) > 1:
        report_lines.append("## Model Comparison\n")
        models = list(stats.keys())
        
        # Find problems where models differ
        by_problem = defaultdict(dict)
        for r in results:
            by_problem[r["problem_id"]][r["model"]] = r["passed"]
        
        both_pass = 0
        both_fail = 0
        model1_only = 0
        model2_only = 0
        
        if len(models) >= 2:
            m1, m2 = models[0], models[1]
            for problem_id, model_results in by_problem.items():
                p1 = model_results.get(m1, False)
                p2 = model_results.get(m2, False)
                
                if p1 and p2:
                    both_pass += 1
                elif not p1 and not p2:
                    both_fail += 1
                elif p1:
                    model1_only += 1
                else:
                    model2_only += 1
            
            report_lines.append(f"| Comparison | Count |")
            report_lines.append(f"|------------|-------|")
            report_lines.append(f"| Both pass | {both_pass} |")
            report_lines.append(f"| Both fail | {both_fail} |")
            report_lines.append(f"| Only {m1} passes | {model1_only} |")
            report_lines.append(f"| Only {m2} passes | {model2_only} |")
            report_lines.append("\n")
    
    # Failed problems list
    report_lines.append("## Failed Problems\n")
    for model, s in stats.items():
        failed = [r for r in results if r["model"] == model and not r["passed"]]
        if failed:
            report_lines.append(f"### {model} ({len(failed)} failures)\n")
            for r in failed[:10]:  # Limit to first 10
                error = (r.get("error") or "Unknown error")[:100]
                report_lines.append(f"- **{r['problem_title']}** [{r['difficulty']}]: {error}")
            if len(failed) > 10:
                report_lines.append(f"- ... and {len(failed) - 10} more")
            report_lines.append("\n")
    
    report = "\n".join(report_lines)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")
        
        # Also generate summary JSON
        summary_path = output_path.parent / "summary.json"
        generate_summary_json(results, stats, summary_path)
    
    return report


def generate_summary_json(results: list[dict], stats: dict, output_path: Path):
    """Generate a clean summary JSON with pass rates and completion metrics."""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for model, s in stats.items():
        # Calculate completion rate for failed problems
        failed_results = [r for r in results if r["model"] == model and not r["passed"]]
        
        completion_rates = []
        for r in failed_results:
            total = r.get("total_tests", 0)
            passed = r.get("passed_tests", 0)
            if total > 0:
                completion_rates.append(passed / total)
        
        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0
        
        # Build difficulty breakdown
        difficulty_stats = {}
        for diff in ["Easy", "Medium", "Hard"]:
            if diff in s.by_difficulty:
                d = s.by_difficulty[diff]
                difficulty_stats[diff.lower()] = {
                    "total": d["total"],
                    "passed": d["passed"],
                    "pass_rate": round(d["passed"] / d["total"] * 100, 1) if d["total"] > 0 else 0
                }
        
        summary["models"][model] = {
            "overall": {
                "total": s.total_problems,
                "passed": s.passed,
                "failed": s.failed,
                "pass_rate": round(s.pass_rate, 1)
            },
            "by_difficulty": difficulty_stats,
            "failed_problems": {
                "count": s.failed,
                "avg_completion_rate": round(avg_completion * 100, 1)
            },
            "timing": {
                "avg_generation_time": round(s.avg_api_time, 2),
                "avg_execution_time": round(s.avg_exec_time, 3)
            }
        }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    # Generate report from existing results
    results_path = Path("results/raw_results.json")
    if results_path.exists():
        report = generate_report(results_path, Path("results/report.md"))
        print(report)
    else:
        print("No results file found. Run the benchmark first.")
