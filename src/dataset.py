"""Dataset loader for LeetCode problems from HuggingFace."""
import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

import config


class Problem:
    """Represents a LeetCode problem with test cases."""
    
    def __init__(self, data: dict):
        self.id = data.get("task_id", data.get("question_id", "unknown"))
        self.title = data.get("task_id", "Unknown").replace("-", " ").title()
        self.difficulty = data.get("difficulty", "Medium")
        self.description = data.get("problem_description", "")
        self.starter_code = data.get("starter_code", "")
        self.solution = data.get("completion", "")  # The actual solution
        self.test_cases = self._extract_test_cases(data)
        self.raw_data = data
    
    def _extract_test_cases(self, data: dict) -> list:
        """Extract test cases from the problem data."""
        test_cases = []
        
        # This dataset uses input_output field
        if "input_output" in data and data["input_output"]:
            for item in data["input_output"]:
                if isinstance(item, dict):
                    test_cases.append({
                        "input": item.get("input", ""),
                        "expected": item.get("output", "")
                    })
        
        return test_cases
    
    def __repr__(self):
        return f"Problem({self.id}: {self.title} [{self.difficulty}])"


def load_leetcode_dataset(cache_path: Optional[Path] = None) -> list[Problem]:
    """Load LeetCode problems from HuggingFace dataset."""
    
    if cache_path is None:
        cache_path = Path("data/problems.json")
    
    # Check cache first
    if cache_path.exists():
        print(f"Loading cached problems from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return [Problem(p) for p in data]
    
    print("Downloading LeetCodeDataset from HuggingFace...")
    dataset = load_dataset("newfacade/LeetCodeDataset", split="train")
    
    problems = []
    for item in tqdm(dataset, desc="Processing problems"):
        problem_dict = dict(item)
        # Convert datetime to string for JSON serialization
        if 'estimated_date' in problem_dict and problem_dict['estimated_date']:
            problem_dict['estimated_date'] = str(problem_dict['estimated_date'])
        problems.append(problem_dict)
    
    # Cache the dataset
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Cached {len(problems)} problems to {cache_path}")
    
    return [Problem(p) for p in problems]


def select_problems(
    problems: list[Problem],
    num_problems: int = config.NUM_PROBLEMS,
    distribution: dict = config.DIFFICULTY_DISTRIBUTION,
    topics: list = None,
    problems_per_topic: int = None,
    seed: int = 42
) -> list[Problem]:
    """
    Select problems by topic (if configured) or by difficulty.
    
    Args:
        problems: All available problems
        num_problems: Max problems to return
        distribution: Difficulty distribution dict
        topics: List of topics to select from (uses config.TOPICS if None)
        problems_per_topic: Problems per topic (uses config.PROBLEMS_PER_TOPIC if None)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Use topic-based selection if configured
    if topics is None:
        topics = getattr(config, 'TOPICS', None)
    if problems_per_topic is None:
        problems_per_topic = getattr(config, 'PROBLEMS_PER_TOPIC', 10)
    
    # Filter to only problems with test cases
    valid_problems = [p for p in problems if p.test_cases]
    
    if topics:
        return _select_by_topic(valid_problems, topics, problems_per_topic, num_problems)
    else:
        return _select_by_difficulty(valid_problems, distribution, num_problems)


def _select_by_topic(
    problems: list[Problem],
    topics: list[str],
    per_topic: int,
    max_problems: int
) -> list[Problem]:
    """Select problems evenly distributed across topics."""
    from collections import defaultdict
    
    # Group by topic
    by_topic = defaultdict(list)
    for p in problems:
        tags = p.raw_data.get("tags", [])
        for tag in tags:
            if tag in topics:
                by_topic[tag].append(p)
    
    selected = []
    seen_ids = set()
    
    for topic in topics:
        available = [p for p in by_topic[topic] if p.id not in seen_ids]
        if len(available) < per_topic:
            print(f"Warning: Only {len(available)} {topic} problems available")
            sample = available
        else:
            sample = random.sample(available, per_topic)
        
        for p in sample:
            seen_ids.add(p.id)
            selected.append(p)
    
    print(f"Selected {len(selected)} problems across {len(topics)} topics")
    random.shuffle(selected)
    return selected[:max_problems]


def _select_by_difficulty(
    problems: list[Problem],
    distribution: dict,
    max_problems: int
) -> list[Problem]:
    """Select problems by difficulty distribution."""
    by_difficulty = {"Easy": [], "Medium": [], "Hard": []}
    for p in problems:
        if p.difficulty in by_difficulty:
            by_difficulty[p.difficulty].append(p)
    
    selected = []
    for difficulty, count in distribution.items():
        available = by_difficulty.get(difficulty, [])
        if len(available) < count:
            print(f"Warning: Only {len(available)} {difficulty} problems available")
            selected.extend(available)
        else:
            selected.extend(random.sample(available, count))
    
    random.shuffle(selected)
    return selected[:max_problems]


if __name__ == "__main__":
    # Test the dataset loader
    problems = load_leetcode_dataset()
    print(f"Loaded {len(problems)} problems")
    
    selected = select_problems(problems, num_problems=10)
    print(f"\nSelected {len(selected)} problems:")
    for p in selected:
        print(f"  - {p}")
        if p.test_cases:
            print(f"    Test cases: {len(p.test_cases)}")
