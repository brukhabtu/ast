#!/usr/bin/env python3
"""Create a performance baseline for the AST library."""

import ast
from pathlib import Path

from astlib.benchmark import BenchmarkSuite


def create_baseline():
    """Create a performance baseline using the AST library itself."""
    suite = BenchmarkSuite("AST Library Performance Baseline")
    
    # Find all Python files in the project
    project_root = Path(__file__).parent
    python_files = []
    for py_file in project_root.rglob("*.py"):
        if not any(part.startswith('.') for part in py_file.parts):
            python_files.append(py_file)
    
    print(f"Found {len(python_files)} Python files to benchmark")
    
    # Benchmark 1: Parse all files
    def parse_all_files():
        count = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                count += 1
            except:
                pass
        return count
    
    parsed_count = suite.run_benchmark(
        parse_all_files,
        operation_name="parse_all_project_files",
        iterations=3,
        profile_memory_usage=True
    )
    
    print(f"Successfully parsed {parsed_count} files")
    
    # Benchmark 2: Count total AST nodes
    def count_all_nodes():
        total = 0
        for py_file in python_files[:5]:  # Just first 5 files for speed
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                total += sum(1 for _ in ast.walk(tree))
            except:
                pass
        return total
    
    node_count = suite.run_benchmark(
        count_all_nodes,
        operation_name="count_ast_nodes_sample",
        iterations=5,
        profile_memory_usage=False
    )
    
    print(f"Counted {node_count} AST nodes in sample files")
    
    # Benchmark 3: Extract all imports
    def extract_imports():
        imports = []
        for py_file in python_files[:10]:  # Sample of 10 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(node.module or '')
            except:
                pass
        return len(set(imports))  # Unique imports
    
    unique_imports = suite.run_benchmark(
        extract_imports,
        operation_name="extract_unique_imports",
        iterations=3,
        profile_memory_usage=True
    )
    
    print(f"Found {unique_imports} unique imports")
    
    # Save baseline
    baseline_path = project_root / "PERFORMANCE_BASELINE.json"
    suite.save_json_report(baseline_path)
    
    # Also save readable report
    report_path = project_root / "baseline_report.md"
    suite.save_markdown_report(report_path)
    
    print(f"\nBaseline created:")
    print(f"  - JSON: {baseline_path}")
    print(f"  - Report: {report_path}")
    
    # Print summary
    report = suite.generate_report()
    print("\nBaseline Summary:")
    print(f"  Total time: {report['summary']['total_time']:.2f} seconds")
    print("\nOperation averages:")
    for timing in report['timing_results']:
        print(f"  - {timing['operation']}: {timing['average_seconds']:.4f}s")


if __name__ == "__main__":
    create_baseline()