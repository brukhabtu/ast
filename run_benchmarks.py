#!/usr/bin/env python3
"""Run performance benchmarks on test repositories."""

import ast
import sys
from pathlib import Path
from typing import Dict, Any

from astlib.benchmark import BenchmarkSuite, compare_benchmarks


def count_ast_nodes(tree: ast.AST) -> int:
    """Count total number of AST nodes in a tree."""
    count = 0
    for node in ast.walk(tree):
        count += 1
    return count


def analyze_python_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single Python file and return metrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content, filename=str(file_path))
    
    # Collect metrics
    metrics = {
        'file': str(file_path),
        'lines': len(content.splitlines()),
        'size_bytes': len(content.encode('utf-8')),
        'ast_nodes': count_ast_nodes(tree),
        'functions': sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
        'classes': sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
        'imports': sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
    }
    
    return metrics


def benchmark_repository(repo_path: Path, suite: BenchmarkSuite) -> Dict[str, Any]:
    """Benchmark AST operations on a repository."""
    repo_name = repo_path.name
    print(f"\nBenchmarking {repo_name}...")
    
    # Collect all Python files
    python_files = list(repo_path.rglob("*.py"))
    print(f"  Found {len(python_files)} Python files")
    
    if not python_files:
        return {'error': 'No Python files found'}
    
    # Benchmark parsing all files
    def parse_all_files():
        results = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                results.append(str(py_file))
            except Exception as e:
                pass
        return results
    
    parsed_files = suite.run_benchmark(
        parse_all_files,
        operation_name=f"{repo_name}_parse_all",
        iterations=3,
        profile_memory_usage=True
    )
    
    print(f"  Successfully parsed {len(parsed_files)} files")
    
    # Benchmark detailed analysis on a subset
    sample_files = python_files[:10]  # Analyze first 10 files
    
    def analyze_sample_files():
        results = []
        for py_file in sample_files:
            try:
                metrics = analyze_python_file(py_file)
                results.append(metrics)
            except:
                pass
        return results
    
    if sample_files:
        analysis_results = suite.run_benchmark(
            analyze_sample_files,
            operation_name=f"{repo_name}_analyze_sample",
            iterations=2,
            profile_memory_usage=True
        )
        
        # Calculate aggregate metrics
        if analysis_results:
            total_lines = sum(m['lines'] for m in analysis_results)
            total_nodes = sum(m['ast_nodes'] for m in analysis_results)
            print(f"  Sample analysis: {total_lines} lines, {total_nodes} AST nodes")
    
    return {
        'repository': repo_name,
        'total_files': len(python_files),
        'parsed_files': len(parsed_files),
        'sample_size': len(sample_files)
    }


def main():
    """Main benchmark execution."""
    # Setup paths
    project_root = Path(__file__).parent
    test_repos_dir = project_root / "test_repos"
    reports_dir = project_root / "benchmark_reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Check if test repos exist
    if not test_repos_dir.exists():
        print("Test repositories not found!")
        print("Please run: python download_test_repos.py")
        sys.exit(1)
    
    # Find available repositories
    repos = [d for d in test_repos_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    if not repos:
        print("No repositories found in test_repos/")
        sys.exit(1)
    
    print(f"Found {len(repos)} repositories to benchmark")
    
    # Create benchmark suite
    suite = BenchmarkSuite("AST Library Performance Benchmark")
    
    # Benchmark each repository
    repo_results = []
    for repo_path in repos:
        result = benchmark_repository(repo_path, suite)
        repo_results.append(result)
    
    # Generate reports
    print("\nGenerating reports...")
    
    timestamp = suite.start_time.strftime("%Y%m%d_%H%M%S")
    json_report = reports_dir / f"benchmark_{timestamp}.json"
    md_report = reports_dir / f"benchmark_{timestamp}.md"
    
    suite.save_json_report(json_report)
    suite.save_markdown_report(md_report)
    
    print(f"  JSON report: {json_report}")
    print(f"  Markdown report: {md_report}")
    
    # Save as baseline if it doesn't exist
    baseline_path = project_root / "PERFORMANCE_BASELINE.json"
    if not baseline_path.exists():
        suite.save_json_report(baseline_path)
        print(f"  Saved as baseline: {baseline_path}")
    else:
        # Compare with baseline
        print("\nComparing with baseline...")
        comparison = compare_benchmarks(baseline_path, json_report)
        
        print(f"  Operations compared: {comparison['summary']['total_operations']}")
        print(f"  Regressions found: {comparison['summary']['regressions_count']}")
        print(f"  Improvements found: {comparison['summary']['improvements_count']}")
        
        if comparison['regressions']:
            print("\n  Performance Regressions:")
            for reg in comparison['regressions']:
                print(f"    - {reg['operation']}: {reg['change_percent']:.1f}% slower")
        
        if comparison['improvements']:
            print("\n  Performance Improvements:")
            for imp in comparison['improvements']:
                print(f"    - {imp['operation']}: {abs(imp['change_percent']):.1f}% faster")
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    report = suite.generate_report()
    print(f"Total operations: {report['summary']['total_operations']}")
    print(f"Total time: {report['summary']['total_time']:.2f} seconds")
    
    for repo_result in repo_results:
        if 'error' not in repo_result:
            print(f"\n{repo_result['repository']}:")
            print(f"  Files: {repo_result['total_files']}")
            print(f"  Successfully parsed: {repo_result['parsed_files']}")


if __name__ == "__main__":
    main()