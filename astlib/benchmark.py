"""Performance benchmarking utilities for AST library.

This module provides timing utilities, memory profiling, and performance
reporting capabilities for benchmarking AST operations on various codebases.
"""

import json
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics
import gc


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    operation: str
    duration_seconds: float
    iterations: int = 1
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    @property
    def average_duration(self) -> float:
        """Average duration per iteration."""
        return self.duration_seconds / self.iterations


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    current_mb: float
    peak_mb: float
    allocated_blocks: int
    
    @classmethod
    def from_tracemalloc(cls) -> "MemorySnapshot":
        """Create snapshot from current tracemalloc state."""
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        return cls(
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            allocated_blocks=len(snapshot.traces)
        )


@dataclass
class MemoryProfile:
    """Memory profiling result."""
    operation: str
    start_memory: MemorySnapshot
    end_memory: MemorySnapshot
    peak_memory_mb: float
    memory_delta_mb: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class BenchmarkResult:
    """Complete benchmark result including timing and memory."""
    operation: str
    timing: TimingResult
    memory: Optional[MemoryProfile] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Timer:
    """High-precision timer for benchmarking operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
        
    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
        
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed
        
    def __enter__(self) -> "Timer":
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


@contextmanager
def time_operation(operation: str, iterations: int = 1):
    """Context manager for timing operations.
    
    Args:
        operation: Name of the operation being timed
        iterations: Number of iterations (for averaging)
        
    Yields:
        TimingResult object that will be populated after the context exits
    """
    timer = Timer()
    result = TimingResult(operation=operation, duration_seconds=0.0, iterations=iterations)
    
    timer.start()
    try:
        yield result
    finally:
        result.duration_seconds = timer.stop()


@contextmanager
def profile_memory(operation: str):
    """Context manager for memory profiling.
    
    Args:
        operation: Name of the operation being profiled
        
    Yields:
        MemoryProfile object that will be populated after the context exits
    """
    gc.collect()  # Clean up before measuring
    
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        
    tracemalloc.clear_traces()
    start_memory = MemorySnapshot.from_tracemalloc()
    peak_start = tracemalloc.get_traced_memory()[1]
    
    profile = MemoryProfile(
        operation=operation,
        start_memory=start_memory,
        end_memory=start_memory,  # Will be updated
        peak_memory_mb=0.0,
        memory_delta_mb=0.0
    )
    
    try:
        yield profile
    finally:
        gc.collect()  # Force collection to get accurate memory usage
        end_memory = MemorySnapshot.from_tracemalloc()
        peak_end = tracemalloc.get_traced_memory()[1]
        
        profile.end_memory = end_memory
        profile.peak_memory_mb = (peak_end - peak_start) / 1024 / 1024
        profile.memory_delta_mb = end_memory.current_mb - start_memory.current_mb


def benchmark_function(
    func: Callable,
    *args,
    operation_name: Optional[str] = None,
    iterations: int = 1,
    profile_memory_usage: bool = True,
    **kwargs
) -> Tuple[Any, BenchmarkResult]:
    """Benchmark a function with timing and optional memory profiling.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        operation_name: Name for the benchmark (defaults to function name)
        iterations: Number of times to run the function
        profile_memory_usage: Whether to profile memory usage
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (function result, benchmark result)
    """
    if operation_name is None:
        operation_name = func.__name__
        
    result = None
    timing_results = []
    
    # Memory profiling for the entire run
    if profile_memory_usage:
        with profile_memory(operation_name) as memory_profile:
            for i in range(iterations):
                with time_operation(f"{operation_name}_iter_{i}", 1) as timing:
                    result = func(*args, **kwargs)
                timing_results.append(timing.duration_seconds)
    else:
        for i in range(iterations):
            with time_operation(f"{operation_name}_iter_{i}", 1) as timing:
                result = func(*args, **kwargs)
            timing_results.append(timing.duration_seconds)
    
    # Calculate aggregate timing
    total_duration = sum(timing_results)
    timing = TimingResult(
        operation=operation_name,
        duration_seconds=total_duration,
        iterations=iterations
    )
    
    benchmark = BenchmarkResult(
        operation=operation_name,
        timing=timing,
        memory=memory_profile if profile_memory_usage else None,
        metadata={
            "iterations": iterations,
            "individual_times": timing_results,
            "min_time": min(timing_results),
            "max_time": max(timing_results),
            "median_time": statistics.median(timing_results),
            "stdev_time": statistics.stdev(timing_results) if len(timing_results) > 1 else 0.0
        }
    )
    
    return result, benchmark


class BenchmarkSuite:
    """Collection of benchmarks with reporting capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []
        self.start_time = datetime.utcnow()
        
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
        
    def run_benchmark(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        iterations: int = 1,
        profile_memory_usage: bool = True,
        **kwargs
    ) -> Any:
        """Run a benchmark and add it to the suite.
        
        Returns:
            The result of the function call
        """
        result, benchmark = benchmark_function(
            func, *args,
            operation_name=operation_name,
            iterations=iterations,
            profile_memory_usage=profile_memory_usage,
            **kwargs
        )
        self.add_result(benchmark)
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        timing_summary = []
        memory_summary = []
        
        for result in self.results:
            timing_data = {
                "operation": result.operation,
                "total_seconds": result.timing.duration_seconds,
                "iterations": result.timing.iterations,
                "average_seconds": result.timing.average_duration,
                "metadata": result.metadata
            }
            timing_summary.append(timing_data)
            
            if result.memory:
                memory_data = {
                    "operation": result.operation,
                    "start_memory_mb": result.memory.start_memory.current_mb,
                    "end_memory_mb": result.memory.end_memory.current_mb,
                    "peak_memory_mb": result.memory.peak_memory_mb,
                    "memory_delta_mb": result.memory.memory_delta_mb,
                    "allocated_blocks": result.memory.end_memory.allocated_blocks
                }
                memory_summary.append(memory_data)
        
        return {
            "suite_name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": duration,
            "timing_results": timing_summary,
            "memory_results": memory_summary,
            "summary": {
                "total_operations": len(self.results),
                "total_time": sum(r.timing.duration_seconds for r in self.results),
                "operations_with_memory": len([r for r in self.results if r.memory])
            }
        }
    
    def save_json_report(self, filepath: Path) -> None:
        """Save benchmark report as JSON."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def save_markdown_report(self, filepath: Path) -> None:
        """Save benchmark report as Markdown."""
        report = self.generate_report()
        
        lines = [
            f"# Benchmark Report: {report['suite_name']}",
            "",
            f"**Start Time:** {report['start_time']}  ",
            f"**End Time:** {report['end_time']}  ",
            f"**Total Duration:** {report['total_duration_seconds']:.2f} seconds",
            "",
            "## Timing Results",
            "",
            "| Operation | Total (s) | Iterations | Average (s) | Min (s) | Max (s) | Median (s) |",
            "|-----------|-----------|------------|-------------|---------|---------|------------|"
        ]
        
        for timing in report['timing_results']:
            meta = timing['metadata']
            lines.append(
                f"| {timing['operation']} | "
                f"{timing['total_seconds']:.4f} | "
                f"{timing['iterations']} | "
                f"{timing['average_seconds']:.4f} | "
                f"{meta.get('min_time', 0):.4f} | "
                f"{meta.get('max_time', 0):.4f} | "
                f"{meta.get('median_time', 0):.4f} |"
            )
        
        if report['memory_results']:
            lines.extend([
                "",
                "## Memory Results",
                "",
                "| Operation | Start (MB) | End (MB) | Peak (MB) | Delta (MB) | Blocks |",
                "|-----------|------------|----------|-----------|------------|--------|"
            ])
            
            for memory in report['memory_results']:
                lines.append(
                    f"| {memory['operation']} | "
                    f"{memory['start_memory_mb']:.2f} | "
                    f"{memory['end_memory_mb']:.2f} | "
                    f"{memory['peak_memory_mb']:.2f} | "
                    f"{memory['memory_delta_mb']:.2f} | "
                    f"{memory['allocated_blocks']} |"
                )
        
        lines.extend([
            "",
            "## Summary",
            "",
            f"- Total Operations: {report['summary']['total_operations']}",
            f"- Total Time: {report['summary']['total_time']:.2f} seconds",
            f"- Operations with Memory Profiling: {report['summary']['operations_with_memory']}"
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


def compare_benchmarks(baseline_path: Path, current_path: Path) -> Dict[str, Any]:
    """Compare two benchmark runs for regression detection.
    
    Args:
        baseline_path: Path to baseline benchmark JSON
        current_path: Path to current benchmark JSON
        
    Returns:
        Comparison report with regression analysis
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)
    
    # Create lookup maps
    baseline_ops = {t['operation']: t for t in baseline['timing_results']}
    current_ops = {t['operation']: t for t in current['timing_results']}
    
    comparisons = []
    regressions = []
    improvements = []
    
    for op_name, current_timing in current_ops.items():
        if op_name in baseline_ops:
            baseline_timing = baseline_ops[op_name]
            
            baseline_avg = baseline_timing['average_seconds']
            current_avg = current_timing['average_seconds']
            
            if baseline_avg > 0:
                change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
            else:
                change_percent = 0
            
            comparison = {
                'operation': op_name,
                'baseline_avg': baseline_avg,
                'current_avg': current_avg,
                'change_percent': change_percent,
                'regression': change_percent > 10,  # 10% slower is a regression
                'improvement': change_percent < -10  # 10% faster is an improvement
            }
            
            comparisons.append(comparison)
            
            if comparison['regression']:
                regressions.append(comparison)
            elif comparison['improvement']:
                improvements.append(comparison)
    
    return {
        'baseline_file': str(baseline_path),
        'current_file': str(current_path),
        'comparisons': comparisons,
        'regressions': regressions,
        'improvements': improvements,
        'summary': {
            'total_operations': len(comparisons),
            'regressions_count': len(regressions),
            'improvements_count': len(improvements),
            'stable_count': len(comparisons) - len(regressions) - len(improvements)
        }
    }