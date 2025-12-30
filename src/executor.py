"""Sandboxed code executor for running generated solutions."""
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import config


@dataclass
class ExecutionResult:
    """Result of executing code against test cases."""
    passed: bool
    total_tests: int
    passed_tests: int
    execution_time: float
    error: Optional[str] = None
    details: Optional[list] = None


def normalize_code(code: str, starter_code: str = "") -> str:
    """
    Try to normalize generated code to match expected Solution class format.
    If code doesn't have 'class Solution', try to wrap it.
    """
    if not code:
        return code
    
    # If it already has Solution class, return as-is
    if "class Solution" in code:
        return code
    
    # Try to extract function name from starter code
    import re
    func_match = re.search(r'def\s+(\w+)\s*\(', starter_code)
    if not func_match:
        # Can't determine function name, return as-is
        return code
    
    func_name = func_match.group(1)
    
    # Check if generated code has a function definition
    if f"def {func_name}" in code or "def " in code:
        # Try to extract the function body and wrap it
        # Find the function definition in generated code
        gen_func_match = re.search(r'def\s+\w+\s*\([^)]*\):', code)
        if gen_func_match:
            # Get everything from the function def onwards
            func_start = gen_func_match.start()
            func_code = code[func_start:]
            
            # Wrap in Solution class
            wrapped = f"""class Solution:
    {func_code.replace(chr(10), chr(10) + '    ')}"""
            return wrapped
    
    # If it's just raw code/logic, try to wrap it as the function body
    # This is a last resort
    lines = code.strip().split('\n')
    indented = '\n        '.join(lines)
    
    # Use starter code as template
    if starter_code:
        wrapped = starter_code.rstrip()
        if wrapped.endswith(':'):
            wrapped += f"\n        {indented}"
        return wrapped
    
    return code


def execute_code(
    code: str,
    test_cases: list[dict],
    starter_code: str = "",
    timeout: int = config.EXECUTION_TIMEOUT
) -> ExecutionResult:
    """
    Execute generated code against test cases in a sandboxed subprocess.
    
    Args:
        code: The Python code to execute
        test_cases: List of test cases with 'input' and 'expected' keys
        starter_code: Original starter code (used to normalize output)
        timeout: Maximum execution time in seconds
    
    Returns:
        ExecutionResult with pass/fail status and timing
    """
    if not code or not test_cases:
        return ExecutionResult(
            passed=False,
            total_tests=len(test_cases),
            passed_tests=0,
            execution_time=0.0,
            error="No code or test cases provided"
        )
    
    # Try to normalize the code to expected format
    normalized_code = normalize_code(code, starter_code)
    
    # Build the test runner script
    test_script = _build_test_script(normalized_code, test_cases)
    
    start_time = time.time()
    
    try:
        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(test_script)
            temp_path = f.name
        
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        # Parse results - always check stdout for test results
        parsed = _parse_results(result.stdout, len(test_cases), execution_time)
        
        # If parsing found results, use those
        if parsed.passed_tests > 0 or "RESULT:" in result.stdout:
            # Add stderr as error info if present but tests ran
            if result.stderr and not parsed.passed:
                parsed.error = result.stderr[:500]
            return parsed
        
        # If no results found and there was an error, report it
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return ExecutionResult(
                passed=False,
                total_tests=len(test_cases),
                passed_tests=0,
                execution_time=execution_time,
                error=error_msg[:500]
            )
        
        return parsed
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        Path(temp_path).unlink(missing_ok=True)
        return ExecutionResult(
            passed=False,
            total_tests=len(test_cases),
            passed_tests=0,
            execution_time=execution_time,
            error=f"Timeout after {timeout} seconds"
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return ExecutionResult(
            passed=False,
            total_tests=len(test_cases),
            passed_tests=0,
            execution_time=execution_time,
            error=str(e)[:500]
        )


def _build_test_script(code: str, test_cases: list[dict]) -> str:
    """Build a Python script that runs the solution against test cases."""
    
    # Common imports that LeetCode problems might need
    imports = """
import sys
from typing import List, Optional, Dict, Set, Tuple, Any
from collections import defaultdict, Counter, deque, OrderedDict
from math import inf, ceil, floor, sqrt, gcd, log, log2, isqrt, comb, factorial
from functools import cache, lru_cache, reduce
from itertools import accumulate, pairwise, permutations, combinations, product, count, groupby, chain, zip_longest
from bisect import bisect_left, bisect_right, insort_left, insort_right
from string import ascii_lowercase, ascii_uppercase, ascii_letters, digits
import heapq
from heapq import heappush, heappop, heapify, heappushpop, heapreplace
import bisect
import math
import itertools
import functools
import re
import string

# For linked list / tree problems (common in LeetCode)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
    
    test_runner = """
# Test runner
import re

def parse_input_string(inp_str):
    '''Parse LeetCode-style input like 'nums = [1,2,3], target = 9' into dict of args.'''
    if not inp_str or not isinstance(inp_str, str):
        return {}
    
    args = {}
    # Split by comma, but respect brackets and strings
    parts = []
    current = ""
    depth = 0
    in_string = False
    string_char = None
    
    for char in inp_str:
        if char in '"\\'' and not in_string:
            in_string = True
            string_char = char
        elif char == string_char and in_string:
            in_string = False
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        
        if char == ',' and depth == 0 and not in_string:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        parts.append(current.strip())
    
    # Parse each part as 'name = value'
    for part in parts:
        if '=' in part:
            name, value = part.split('=', 1)
            name = name.strip()
            value = value.strip()
            try:
                args[name] = eval(value)
            except:
                args[name] = value
    
    return args

def run_tests():
    passed = 0
    total = len(TEST_CASES)
    
    for i, test in enumerate(TEST_CASES):
        try:
            result = run_single_test(test)
            if result:
                passed += 1
                print(f"PASS: Test {i+1}")
            else:
                print(f"FAIL: Test {i+1}")
        except Exception as e:
            print(f"ERROR: Test {i+1}: {e}")
    
    print(f"RESULT: {passed}/{total}")
    return passed == total

def run_single_test(test):
    inp = test.get('input', '')
    expected = test.get('expected', '')
    
    try:
        sol = Solution()
        # Find the main method (not starting with _)
        methods = [m for m in dir(sol) if not m.startswith('_') and callable(getattr(sol, m))]
        if not methods:
            print("No methods found in Solution", file=sys.stderr)
            return False
        
        method = getattr(sol, methods[0])
        
        # Parse the input string into arguments
        if isinstance(inp, str) and '=' in inp:
            # LeetCode format: 'nums = [1,2,3], target = 9'
            args_dict = parse_input_string(inp)
            # Get method parameter names (skip 'self')
            import inspect
            try:
                sig = inspect.signature(method)
                param_names = [p for p in sig.parameters.keys()]
                args = [args_dict.get(p) for p in param_names if p in args_dict]
            except:
                args = list(args_dict.values())
        elif isinstance(inp, str):
            try:
                args = [eval(inp)]
            except:
                args = [inp]
        else:
            args = [inp] if not isinstance(inp, (list, tuple)) else list(inp)
        
        result = method(*args)
        
        # Parse expected value
        if isinstance(expected, str):
            try:
                expected_val = eval(expected)
            except:
                expected_val = expected
        else:
            expected_val = expected
        
        return result == expected_val
        
    except Exception as e:
        print(f"Test execution error: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
"""
    
    # Serialize test cases
    test_cases_str = f"TEST_CASES = {repr(test_cases)}"
    
    return f"{imports}\n\n{code}\n\n{test_cases_str}\n\n{test_runner}"


def _parse_results(output: str, total_tests: int, execution_time: float) -> ExecutionResult:
    """Parse the test runner output to get results."""
    lines = output.strip().split('\n')
    
    passed_count = sum(1 for line in lines if line.startswith('PASS:'))
    
    # Look for the RESULT line
    for line in lines:
        if line.startswith('RESULT:'):
            try:
                parts = line.split(':')[1].strip().split('/')
                passed_count = int(parts[0])
                total_tests = int(parts[1])
            except:
                pass
    
    return ExecutionResult(
        passed=passed_count == total_tests and total_tests > 0,
        total_tests=total_tests,
        passed_tests=passed_count,
        execution_time=execution_time,
        details=[line for line in lines if line.startswith(('PASS:', 'FAIL:', 'ERROR:'))]
    )


if __name__ == "__main__":
    # Test the executor
    test_code = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
"""
    
    test_cases = [
        {"input": "([2, 7, 11, 15], 9)", "expected": "[0, 1]"},
        {"input": "([3, 2, 4], 6)", "expected": "[1, 2]"},
    ]
    
    result = execute_code(test_code, test_cases)
    print(f"Passed: {result.passed}")
    print(f"Tests: {result.passed_tests}/{result.total_tests}")
    print(f"Time: {result.execution_time:.3f}s")
    if result.error:
        print(f"Error: {result.error}")
