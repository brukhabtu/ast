"""
Performance testing helpers for benchmarking and profiling.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional


@dataclass
class PerformanceResult:
    """Container for performance measurement results."""
    name: str
    duration: float  # seconds
    iterations: int
    mean: float
    min: float
    max: float
    memory_before: Optional[int] = None  # bytes
    memory_after: Optional[int] = None  # bytes
    memory_peak: Optional[int] = None  # bytes
    
    @property
    def memory_used(self) -> Optional[int]:
        """Calculate memory used during operation."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None
    
    def __str__(self) -> str:
        """Format performance results for display."""
        lines = [
            f"Performance: {self.name}",
            f"  Total time: {self.duration:.4f}s ({self.iterations} iterations)",
            f"  Mean: {self.mean:.6f}s",
            f"  Min: {self.min:.6f}s",
            f"  Max: {self.max:.6f}s",
        ]
        
        if self.memory_used is not None:
            lines.append(f"  Memory used: {self.memory_used / 1024 / 1024:.2f} MB")
        
        return "\n".join(lines)


def measure_performance(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    name: Optional[str] = None,
    **kwargs: Any
) -> PerformanceResult:
    """
    Measure performance of a function.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        name: Optional name for the measurement
        **kwargs: Keyword arguments for func
    
    Returns:
        PerformanceResult with timing information
    """
    if name is None:
        name = func.__name__
    
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Actual measurements
    times: list[float] = []
    
    start_total = time.perf_counter()
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    end_total = time.perf_counter()
    
    return PerformanceResult(
        name=name,
        duration=end_total - start_total,
        iterations=iterations,
        mean=sum(times) / len(times),
        min=min(times),
        max=max(times),
    )


@contextmanager
def benchmark_context(name: str = "Operation") -> Generator[list[float], None, None]:
    """
    Context manager for benchmarking code blocks.
    
    Usage:
        with benchmark_context("My operation") as times:
            for i in range(100):
                # Code to benchmark
                with times:
                    result = expensive_operation()
    
    Args:
        name: Name of the operation being benchmarked
    
    Yields:
        List that will contain timing measurements
    """
    measurements: list[float] = []
    
    class Timer:
        def __enter__(self) -> None:
            self.start = time.perf_counter()
        
        def __exit__(self, *args: Any) -> None:
            end = time.perf_counter()
            measurements.append(end - self.start)
    
    timer = Timer()
    
    # Allow both patterns:
    # 1. with benchmark_context() as times: with times: ...
    # 2. Direct timing storage in measurements list
    setattr(measurements, '__enter__', timer.__enter__)
    setattr(measurements, '__exit__', timer.__exit__)
    
    yield measurements
    
    if measurements:
        result = PerformanceResult(
            name=name,
            duration=sum(measurements),
            iterations=len(measurements),
            mean=sum(measurements) / len(measurements),
            min=min(measurements),
            max=max(measurements),
        )
        print(result)


def assert_performance(
    func: Callable[..., Any],
    *args: Any,
    max_duration: float,
    iterations: int = 100,
    percentile: float = 0.95,
    **kwargs: Any
) -> None:
    """
    Assert that a function performs within specified limits.
    
    Args:
        func: Function to test
        *args: Positional arguments for func
        max_duration: Maximum allowed duration in seconds
        iterations: Number of iterations to run
        percentile: Percentile to check (default 95th)
        **kwargs: Keyword arguments for func
    
    Raises:
        AssertionError: If performance requirements not met
    """
    result = measure_performance(func, *args, iterations=iterations, **kwargs)
    
    # Sort times to calculate percentile
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    times.sort()
    percentile_index = int(len(times) * percentile)
    percentile_time = times[percentile_index]
    
    if percentile_time > max_duration:
        raise AssertionError(
            f"{func.__name__} performance requirement not met:\n"
            f"  {percentile * 100}th percentile: {percentile_time:.6f}s\n"
            f"  Required: < {max_duration:.6f}s\n"
            f"  Mean: {result.mean:.6f}s"
        )


class PerformanceTracker:
    """Track performance across multiple operations."""
    
    def __init__(self, name: str = "Performance Test"):
        self.name = name
        self.results: list[PerformanceResult] = []
    
    def measure(
        self,
        func: Callable[..., Any],
        *args: Any,
        name: Optional[str] = None,
        iterations: int = 100,
        **kwargs: Any
    ) -> Any:
        """
        Measure and track a function's performance.
        
        Returns:
            The function's return value
        """
        result = measure_performance(
            func, *args,
            name=name,
            iterations=iterations,
            **kwargs
        )
        self.results.append(result)
        
        # Return the actual function result
        return func(*args, **kwargs)
    
    def report(self) -> str:
        """Generate a performance report."""
        if not self.results:
            return f"{self.name}: No measurements recorded"
        
        lines = [
            f"Performance Report: {self.name}",
            "=" * 50,
        ]
        
        for result in self.results:
            lines.append(str(result))
            lines.append("")
        
        # Summary
        total_time = sum(r.duration for r in self.results)
        lines.extend([
            "Summary:",
            f"  Total operations: {len(self.results)}",
            f"  Total time: {total_time:.4f}s",
        ])
        
        return "\n".join(lines)
    
    def assert_all_under(self, max_duration: float) -> None:
        """Assert all operations completed under the specified duration."""
        failures = []
        
        for result in self.results:
            if result.mean > max_duration:
                failures.append(
                    f"{result.name}: {result.mean:.6f}s > {max_duration:.6f}s"
                )
        
        if failures:
            raise AssertionError(
                f"Performance requirements not met:\n" + "\n".join(failures)
            )


def compare_performance(
    implementations: dict[str, Callable[..., Any]],
    *args: Any,
    iterations: int = 1000,
    **kwargs: Any
) -> dict[str, PerformanceResult]:
    """
    Compare performance of multiple implementations.
    
    Args:
        implementations: Dict of name -> function
        *args: Arguments to pass to each function
        iterations: Number of iterations per implementation
        **kwargs: Keyword arguments for functions
    
    Returns:
        Dict of name -> PerformanceResult
    """
    results = {}
    
    for name, func in implementations.items():
        results[name] = measure_performance(
            func, *args,
            name=name,
            iterations=iterations,
            **kwargs
        )
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("=" * 50)
    
    # Sort by mean time
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean)
    baseline = sorted_results[0][1].mean
    
    for name, result in sorted_results:
        ratio = result.mean / baseline
        print(f"{name:20} {result.mean:.6f}s ({ratio:.2f}x)")
    
    return results