# AST Library Performance Benchmarking

This document describes the performance benchmarking framework for the AST library.

**Note**: The `ast` package was renamed to `astlib` to avoid conflicts with Python's built-in `ast` module.

## Overview

The benchmarking framework provides:
- High-precision timing utilities
- Memory profiling with tracemalloc
- JSON and Markdown report generation
- Regression detection between benchmark runs
- Integration with popular Python repositories for real-world testing

## Components

### Core Module: `astlib/benchmark.py`

Key classes and functions:
- `Timer`: High-precision timer for measuring execution time
- `BenchmarkSuite`: Collection of benchmarks with reporting
- `time_operation`: Context manager for timing code blocks
- `profile_memory`: Context manager for memory profiling
- `benchmark_function`: Complete benchmarking with timing and memory
- `compare_benchmarks`: Regression detection between runs

### Scripts

1. **`download_test_repos.py`**: Downloads test repositories (Django, Flask, FastAPI, Requests)
2. **`run_benchmarks.py`**: Runs benchmarks on downloaded repositories
3. **`dogfood_demo.py`**: Demonstrates tracking codebase exploration patterns

## Usage

### Basic Benchmarking

```python
from astlib.benchmark import BenchmarkSuite

suite = BenchmarkSuite("My Benchmarks")

# Benchmark a function
result = suite.run_benchmark(
    my_function,
    arg1, arg2,
    operation_name="my_operation",
    iterations=5,
    profile_memory_usage=True
)

# Save reports
suite.save_json_report("benchmark.json")
suite.save_markdown_report("benchmark.md")
```

### Memory Profiling

```python
from astlib.benchmark import profile_memory

with profile_memory("operation_name") as mem:
    # Your code here
    data = process_large_dataset()

print(f"Peak memory: {mem.peak_memory_mb:.2f} MB")
print(f"Memory delta: {mem.memory_delta_mb:.2f} MB")
```

### Regression Detection

```python
from astlib.benchmark import compare_benchmarks

comparison = compare_benchmarks("baseline.json", "current.json")

for regression in comparison['regressions']:
    print(f"{regression['operation']}: {regression['change_percent']:.1f}% slower")
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v
```

## Benchmark Reports

Reports are saved in `benchmark_reports/` directory:
- JSON format for programmatic analysis
- Markdown format for human reading

The first benchmark run creates `PERFORMANCE_BASELINE.json` for future comparisons.

## Test Repositories

The framework tests against real-world Python projects:
- Django: Web framework with complex codebase
- Flask: Lightweight web framework
- FastAPI: Modern async web framework
- Requests: Popular HTTP library

## Dogfooding Insights

The `dogfood_demo.py` script tracks how we explore codebases manually:
- Time spent finding entry points
- Effort to understand package structure
- Identifying key modules
- Tracing import relationships

This helps identify what AST operations need optimization.