"""Ollama client for local model inference."""
import re
import time
from typing import Optional

import requests

import config


class OllamaClient:
    """Client for interacting with local Ollama models."""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
    
    def list_models(self) -> list[str]:
        """List available models."""
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    
    def pull_model(self, model: str) -> bool:
        """Pull a model if not already available."""
        print(f"Pulling model {model}... (this may take a while)")
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model},
            stream=True,
            timeout=600  # 10 minutes for large model downloads
        )
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "status" in data:
                    print(f"  {data['status']}", end="\r")
        
        print(f"\n✅ Model {model} ready")
        return True
    
    def generate_solution(
        self,
        problem_description: str,
        starter_code: str,
        model: str = "deepseek-coder:6.7b-instruct",
        max_retries: int = config.MAX_RETRIES
    ) -> tuple[str, float]:
        """
        Generate a solution for a LeetCode problem.
        
        Args:
            problem_description: The problem statement
            starter_code: The starter code/function signature
            model: Ollama model to use
            max_retries: Number of retries on failure
        
        Returns:
            Tuple of (code_solution, response_time_seconds)
        """
        prompt = self._build_prompt(problem_description, starter_code)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,  # Deterministic
                            "num_predict": 2048,  # Max tokens
                        }
                    },
                    timeout=config.GENERATION_TIMEOUT
                )
                response.raise_for_status()
                
                response_time = time.time() - start_time
                
                result = response.json()
                code = self._extract_code(result.get("response", ""))
                
                return code, response_time
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}. Retrying...")
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    raise
        
        return "", 0.0
    
    def _build_prompt(self, problem_description: str, starter_code: str) -> str:
        """Build the prompt for code generation."""
        return f"""You are an expert Python programmer. Solve the following LeetCode problem.
Return ONLY the complete Python solution code, no explanations.

## Problem
{problem_description}

## Starter Code
```python
{starter_code}
```

## Solution
```python
"""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from the model response."""
        if not response:
            return ""
        
        # The response might be just the code (we prompted with ```python)
        # Try to find code blocks first
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return max(matches, key=len).strip()
        
        # Otherwise, assume the response IS the code (common for base models)
        # Clean up any trailing ``` if present
        code = response.strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        
        return code


def ensure_models_available(models: list[str], client: Optional[OllamaClient] = None):
    """Ensure all required models are available in Ollama."""
    if client is None:
        client = OllamaClient()
    
    available = client.list_models()
    
    for model in models:
        # Check if model is available (handle tags like :latest)
        model_name = model.split(":")[0]
        if not any(model_name in m for m in available):
            print(f"Model {model} not found. Pulling...")
            client.pull_model(model)
        else:
            print(f"✅ Model {model} available")


if __name__ == "__main__":
    # Test the client
    print("Testing Ollama client...")
    
    try:
        client = OllamaClient()
        print(f"Connected to Ollama at {client.base_url}")
        
        models = client.list_models()
        print(f"Available models: {models}")
        
        # Test with a simple problem
        test_problem = """
        Given an array of integers nums and an integer target, return indices of the two 
        numbers such that they add up to target.
        """
        
        test_starter = """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        pass
"""
        
        print("\nTesting code generation...")
        model = config.MODELS.get("instruct", "deepseek-coder:6.7b-instruct")
        
        if model not in models:
            print(f"Model {model} not found. Run: ollama pull {model}")
        else:
            code, response_time = client.generate_solution(test_problem, test_starter, model)
            print(f"Response time: {response_time:.2f}s")
            print(f"Generated code:\n{code}")
            
    except ConnectionError as e:
        print(f"Error: {e}")
        print("\nTo start Ollama, run: ollama serve")
