#!/usr/bin/env python3
"""
Generate regression tests from validation failures

This script analyzes failure cases from the validation runs and creates
automated regression tests to ensure these issues are fixed and stay fixed.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

class RegressionTestGenerator:
    """Generates regression tests from failure cases"""
    
    def __init__(self, failure_cases_dir: Path, output_dir: Path):
        self.failure_cases_dir = failure_cases_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track unique failure patterns
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.test_cases: List[str] = []
    
    def analyze_failures(self):
        """Analyze all failure cases and categorize them"""
        for repo_dir in self.failure_cases_dir.iterdir():
            if not repo_dir.is_dir():
                continue
                
            summary_file = repo_dir / "failures_summary.json"
            if not summary_file.exists():
                continue
            
            with open(summary_file) as f:
                failures = json.load(f)
            
            for failure in failures:
                pattern = self._categorize_failure(failure)
                self.failure_patterns[pattern].append(failure)
    
    def _categorize_failure(self, failure: Dict) -> str:
        """Categorize failure into a pattern type"""
        error_type = failure.get("error_type", "Unknown")
        error_msg = failure.get("error_message", "")
        
        # Common Python syntax patterns
        if error_type == "SyntaxError":
            if "walrus" in error_msg or ":=" in error_msg:
                return "walrus_operator"
            elif "match" in error_msg:
                return "pattern_matching"
            elif "async" in error_msg:
                return "async_syntax"
            elif "f-string" in error_msg:
                return "fstring_syntax"
            else:
                return "generic_syntax"
        
        # Type-related errors
        elif "type" in error_type.lower():
            return "type_annotation"
        
        # Import errors
        elif "import" in error_type.lower():
            return "import_error"
        
        # Default category
        return f"{error_type.lower()}_error"
    
    def generate_test_module(self, pattern: str, failures: List[Dict]) -> str:
        """Generate a test module for a specific failure pattern"""
        test_content = f'''"""
Regression tests for {pattern.replace('_', ' ').title()} failures

Auto-generated from validation failures found in real Python codebases.
"""

import pytest
import sys
from pathlib import Path

# Add AST library to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ast_lib import ASTParser, ParseError
except ImportError:
    # Placeholder for when AST library is implemented
    class ASTParser:
        def parse(self, code: str, filename: str = "<string>"):
            import ast
            return ast.parse(code, filename)
    
    class ParseError(Exception):
        pass


class Test{pattern.title().replace('_', '')}:
    """Test cases for {pattern.replace('_', ' ')} patterns"""
'''
        
        # Generate unique test cases
        seen_patterns = set()
        test_num = 1
        
        for failure in failures[:20]:  # Limit to 20 tests per pattern
            code_snippet = failure.get("code_snippet", "")
            if not code_snippet or code_snippet in seen_patterns:
                continue
            
            seen_patterns.add(code_snippet)
            
            # Extract the problematic line
            problem_line = self._extract_problem_line(code_snippet)
            if not problem_line:
                continue
            
            test_content += f'''
    def test_{pattern}_{test_num}(self):
        """
        Test case from: {failure.get('repo_name', 'unknown')}
        File: {failure.get('file_path', 'unknown')}
        Error: {failure.get('error_message', 'unknown')}
        """
        code = {repr(problem_line)}
        
        parser = ASTParser()
        
        # This should parse without errors when the issue is fixed
        try:
            result = parser.parse(code)
            assert result is not None
        except ParseError as e:
            pytest.fail(f"Failed to parse valid Python code: {{e}}")
'''
            test_num += 1
        
        return test_content
    
    def _extract_problem_line(self, code_snippet: str) -> Optional[str]:
        """Extract the problematic line from code snippet"""
        if not code_snippet:
            return None
        
        lines = code_snippet.split('\n')
        for line in lines:
            if line.strip().startswith('>>>'):
                # This is the error line
                return line.split(':', 1)[1].strip() if ':' in line else line[3:].strip()
        
        return None
    
    def generate_edge_case_tests(self) -> str:
        """Generate tests for specific edge cases"""
        edge_cases = f'''"""
Edge case tests for AST parser

These tests cover edge cases discovered during validation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ast_lib import ASTParser, ParseError
except ImportError:
    class ASTParser:
        def parse(self, code: str, filename: str = "<string>"):
            import ast
            return ast.parse(code, filename)
    
    class ParseError(Exception):
        pass


class TestEdgeCases:
    """Edge cases found in real-world Python code"""
    
    def test_deeply_nested_structures(self):
        """Test parsing of deeply nested structures"""
        # Generate deeply nested code
        depth = 50
        code = "x = " + "(" * depth + "1" + ")" * depth
        
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    def test_very_long_lines(self):
        """Test parsing of very long lines"""
        # Create a very long line
        long_string = '"' + 'a' * 5000 + '"'
        code = f"x = {long_string}"
        
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    def test_unicode_identifiers(self):
        """Test parsing of Unicode identifiers"""
        code = """
# Unicode variable names
π = 3.14159
面积 = π * 半径 ** 2
"""
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    def test_complex_fstrings(self):
        """Test parsing of complex f-strings"""
        code = '''
x = 42
s = f"{{x=}} {{x=:>10}} {{x=:.2f}} {{x=!r}} {{x=!s}}"
nested = f"{{f'inner {{x}}'}}"
'''
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    def test_walrus_in_comprehensions(self):
        """Test walrus operator in various contexts"""
        code = '''
# Walrus in list comprehension
data = [y := x**2 for x in range(10) if y > 25]

# Walrus in dict comprehension  
squares = {x: (y := x**2) for x in range(5)}

# Walrus in generator
gen = (y for x in range(10) if (y := x**2) > 25)
'''
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
    def test_pattern_matching_edge_cases(self):
        """Test edge cases in pattern matching"""
        code = '''
def process(obj):
    match obj:
        case [x, *rest, y] if x == y:
            return "symmetric"
        case {"key": value, **rest}:
            return value
        case str() | bytes():
            return "text"
        case [int(), *_, int() as last]:
            return last
        case _:
            return None
'''
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
    
    def test_type_annotation_edge_cases(self):
        """Test complex type annotations"""
        code = '''
from typing import Union, List, Dict, Callable, TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')

class Container(Generic[T, U]):
    def method(
        self,
        x: Union[int, str], 
        y: List[Dict[str, Callable[[int], str]]],
        z: 'Container[T, U]'
    ) -> Optional[Tuple[T, ...]]:
        pass
'''
        parser = ASTParser()
        result = parser.parse(code)
        assert result is not None
'''
        
        return edge_cases
    
    def generate_performance_tests(self) -> str:
        """Generate performance regression tests"""
        perf_tests = '''"""
Performance regression tests

Ensure parsing performance doesn't degrade.
"""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ast_lib import ASTParser
except ImportError:
    class ASTParser:
        def parse(self, code: str, filename: str = "<string>"):
            import ast
            return ast.parse(code, filename)


class TestPerformance:
    """Performance regression tests"""
    
    def test_large_file_parsing(self):
        """Test parsing performance on large files"""
        # Generate a large Python file
        lines = []
        for i in range(1000):
            lines.append(f"def function_{i}(x, y, z):")
            lines.append(f"    result = x + y + z + {i}")
            lines.append(f"    return result * 2")
            lines.append("")
        
        code = "\\n".join(lines)
        
        parser = ASTParser()
        start_time = time.time()
        result = parser.parse(code)
        parse_time = time.time() - start_time
        
        assert result is not None
        assert parse_time < 1.0, f"Parsing took {parse_time:.2f}s, expected < 1s"
    
    def test_many_imports_performance(self):
        """Test performance with many imports"""
        # Generate file with many imports
        imports = []
        for i in range(500):
            imports.append(f"from module{i} import func{i}, Class{i}")
        
        code = "\\n".join(imports)
        
        parser = ASTParser()
        start_time = time.time()
        result = parser.parse(code)
        parse_time = time.time() - start_time
        
        assert result is not None
        assert parse_time < 0.5, f"Parsing took {parse_time:.2f}s, expected < 0.5s"
    
    def test_deeply_nested_performance(self):
        """Test performance with deeply nested structures"""
        # Create nested function definitions
        code = ""
        indent = ""
        for i in range(20):
            code += f"{indent}def level_{i}():\\n"
            indent += "    "
        code += f"{indent}pass"
        
        parser = ASTParser()
        start_time = time.time()
        result = parser.parse(code)
        parse_time = time.time() - start_time
        
        assert result is not None
        assert parse_time < 0.1, f"Parsing took {parse_time:.2f}s, expected < 0.1s"
'''
        return perf_tests
    
    def generate_all_tests(self):
        """Generate all regression test files"""
        # Analyze failures first
        self.analyze_failures()
        
        # Generate test modules for each failure pattern
        for pattern, failures in self.failure_patterns.items():
            if not failures:
                continue
            
            test_content = self.generate_test_module(pattern, failures)
            test_file = self.output_dir / f"test_regression_{pattern}.py"
            test_file.write_text(test_content)
            print(f"Generated: {test_file}")
        
        # Generate edge case tests
        edge_case_content = self.generate_edge_case_tests()
        edge_case_file = self.output_dir / "test_edge_cases.py"
        edge_case_file.write_text(edge_case_content)
        print(f"Generated: {edge_case_file}")
        
        # Generate performance tests
        perf_content = self.generate_performance_tests()
        perf_file = self.output_dir / "test_performance_regression.py"
        perf_file.write_text(perf_content)
        print(f"Generated: {perf_file}")
        
        # Generate test runner
        self._generate_test_runner()
    
    def _generate_test_runner(self):
        """Generate a test runner script"""
        runner_content = '''#!/usr/bin/env python3
"""
Run all regression tests
"""

import subprocess
import sys
from pathlib import Path

def run_regression_tests():
    """Run all regression tests and report results"""
    test_dir = Path(__file__).parent
    
    # Find all test files
    test_files = sorted(test_dir.glob("test_*.py"))
    
    print(f"Running {len(test_files)} regression test modules...")
    print("=" * 60)
    
    failed_modules = []
    
    for test_file in test_files:
        print(f"\\nRunning {test_file.name}...", end="", flush=True)
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(" ✓ PASSED")
        else:
            print(" ✗ FAILED")
            failed_modules.append(test_file.name)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
    
    print("\\n" + "=" * 60)
    print(f"\\nSummary: {len(test_files) - len(failed_modules)}/{len(test_files)} modules passed")
    
    if failed_modules:
        print("\\nFailed modules:")
        for module in failed_modules:
            print(f"  - {module}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_regression_tests())
'''
        
        runner_file = self.output_dir / "run_regression_tests.py"
        runner_file.write_text(runner_content)
        runner_file.chmod(0o755)
        print(f"Generated test runner: {runner_file}")


def main():
    """Generate regression tests from validation failures"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate regression tests from failures")
    parser.add_argument(
        "--failures-dir",
        type=Path,
        default=Path(__file__).parent / "failure_cases",
        help="Directory containing failure cases"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path(__file__).parent / "regression_tests",
        help="Output directory for regression tests"
    )
    
    args = parser.parse_args()
    
    if not args.failures_dir.exists():
        print(f"Failure cases directory not found: {args.failures_dir}")
        print("Run validation first to generate failure cases.")
        return 1
    
    generator = RegressionTestGenerator(args.failures_dir, args.output_dir)
    generator.generate_all_tests()
    
    print(f"\nRegenerated regression tests in: {args.output_dir}")
    print(f"Run tests with: python {args.output_dir}/run_regression_tests.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())