"""Configuration for LeetCode Benchmark Tool."""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration (Local inference)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Models to benchmark (Ollama model names)
MODELS = {
    "base": "deepseek-coder:6.7b-base",       # Raw pretrained model
    "instruct": "deepseek-coder:6.7b-instruct",  # Official fine-tuned
    "tuned": "deepseek-leetcode",             # Your custom fine-tuned model (1 epoch)
    "epoch3": "deepseek-leetcode-epoch3",     # 3-epoch fine-tuned model
    "p3": "deepseek-leetcode-p3",             # New properly merged 3-epoch model
}

# Benchmark Configuration
NUM_PROBLEMS = 100

# Difficulty distribution (problems per difficulty)
DIFFICULTY_DISTRIBUTION = {
    "Easy": 33,
    "Medium": 34,
    "Hard": 33,
}

# Topic-based selection (set to None to use difficulty-based)
# Each topic will get PROBLEMS_PER_TOPIC problems
TOPICS = [
    "Array",
    "String",
    "Hash Table",
    "Dynamic Programming",
    "Binary Search",
    "Tree",
    "Two Pointers",
    "Stack",
    "Greedy",
    "Math",
]
PROBLEMS_PER_TOPIC = 10  # 10 problems x 10 topics = 100 problems

# Execution Configuration  
EXECUTION_TIMEOUT = 30  # seconds for code execution
GENERATION_TIMEOUT = 300  # seconds for model generation (local is slower)
MAX_RETRIES = 2
