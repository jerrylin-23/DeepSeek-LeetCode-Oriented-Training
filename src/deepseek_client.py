"""DeepSeek API client for code generation."""
import re
import time
from typing import Optional

from openai import OpenAI

import config


class DeepSeekClient:
    """Client for interacting with DeepSeek API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.base_url = base_url or config.DEEPSEEK_BASE_URL
        
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_solution(
        self,
        problem_description: str,
        starter_code: str,
        model: str = "deepseek-chat",
        max_retries: int = config.MAX_RETRIES
    ) -> tuple[str, float]:
        """
        Generate a solution for a LeetCode problem.
        
        Args:
            problem_description: The problem statement
            starter_code: The starter code/function signature
            model: Model to use (deepseek-chat or deepseek-coder)
            max_retries: Number of retries on failure
        
        Returns:
            Tuple of (code_solution, response_time_seconds)
        """
        prompt = self._build_prompt(problem_description, starter_code)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Python programmer. "
                                "Solve the given LeetCode problem. "
                                "Return ONLY the complete solution code, no explanations. "
                                "The code should be a complete, working solution that can be executed directly."
                            )
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.0,  # Deterministic for benchmarking
                    max_tokens=2048
                )
                
                response_time = time.time() - start_time
                
                code = self._extract_code(response.choices[0].message.content)
                return code, response_time
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return "", 0.0
    
    def _build_prompt(self, problem_description: str, starter_code: str) -> str:
        """Build the prompt for code generation."""
        return f"""Solve this LeetCode problem in Python:

## Problem
{problem_description}

## Starter Code
```python
{starter_code}
```

Provide a complete, working Python solution. Return only the code, no explanations."""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from the model response."""
        if not response:
            return ""
        
        # Try to extract from markdown code blocks
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Return the longest match (likely the full solution)
            return max(matches, key=len).strip()
        
        # If no code blocks, return the raw response (might be plain code)
        return response.strip()


if __name__ == "__main__":
    # Test the client
    client = DeepSeekClient()
    
    test_problem = """
    Given an array of integers nums and an integer target, return indices of the two numbers 
    such that they add up to target. You may assume that each input would have exactly one 
    solution, and you may not use the same element twice.
    """
    
    test_starter = """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        pass
    """
    
    print("Testing with deepseek-chat...")
    code, response_time = client.generate_solution(test_problem, test_starter, "deepseek-chat")
    print(f"Response time: {response_time:.2f}s")
    print(f"Generated code:\n{code}")
