"""Unit tests for benchmark.py module."""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest

from astlib.benchmark import (
    Timer, TimingResult, MemorySnapshot, MemoryProfile, BenchmarkResult,
    time_operation, profile_memory, benchmark_function, BenchmarkSuite,
    compare_benchmarks
)


class TestTimer:
    """Test the Timer class."""
    
    def test_timer_basic_usage(self):
        """Test basic timer start and stop."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = timer.stop()
        
        assert elapsed > 0.01
        assert elapsed < 0.02  # Should not take much longer than 10ms
        
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = Timer()
        with timer:
            time.sleep(0.01)
        
        assert timer.elapsed > 0.01
        assert timer.elapsed < 0.02
        
    def test_timer_not_started_error(self):
        """Test error when stopping timer that wasn't started."""
        timer = Timer()
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()


class TestTimingResult:
    """Test the TimingResult dataclass."""
    
    def test_timing_result_creation(self):
        """Test creating a timing result."""
        result = TimingResult(operation="test_op", duration_seconds=1.5, iterations=3)
        
        assert result.operation == "test_op"
        assert result.duration_seconds == 1.5
        assert result.iterations == 3
        assert result.average_duration == 0.5
        assert result.timestamp != ""
        
    def test_timing_result_custom_timestamp(self):
        """Test timing result with custom timestamp."""
        custom_time = "2024-01-01T00:00:00"
        result = TimingResult(
            operation="test_op", 
            duration_seconds=1.0,
            timestamp=custom_time
        )
        assert result.timestamp == custom_time


class TestMemorySnapshot:
    """Test the MemorySnapshot class."""
    
    @patch('tracemalloc.get_traced_memory')
    @patch('tracemalloc.take_snapshot')
    def test_memory_snapshot_from_tracemalloc(self, mock_snapshot, mock_memory):
        """Test creating memory snapshot from tracemalloc."""
        # Mock tracemalloc data
        mock_memory.return_value = (1024 * 1024, 2048 * 1024)  # 1MB current, 2MB peak
        mock_trace = Mock()
        mock_trace.traces = [Mock()] * 100  # 100 memory blocks
        mock_snapshot.return_value = mock_trace
        
        snapshot = MemorySnapshot.from_tracemalloc()
        
        assert snapshot.current_mb == 1.0
        assert snapshot.peak_mb == 2.0
        assert snapshot.allocated_blocks == 100


class TestTimeOperation:
    """Test the time_operation context manager."""
    
    def test_time_operation_basic(self):
        """Test basic time_operation usage."""
        with time_operation("test_op") as result:
            time.sleep(0.01)
            
        assert result.operation == "test_op"
        assert result.duration_seconds > 0.01
        assert result.iterations == 1
        
    def test_time_operation_with_iterations(self):
        """Test time_operation with multiple iterations."""
        with time_operation("test_op", iterations=5) as result:
            time.sleep(0.01)
            
        assert result.iterations == 5
        assert result.average_duration == result.duration_seconds / 5


class TestProfileMemory:
    """Test the profile_memory context manager."""
    
    @patch('tracemalloc.is_tracing', return_value=True)
    @patch('tracemalloc.clear_traces')
    @patch('tracemalloc.get_traced_memory')
    @patch('gc.collect')
    @patch('astlib.benchmark.MemorySnapshot.from_tracemalloc')
    def test_profile_memory_basic(
        self, mock_snapshot, mock_collect, mock_memory, mock_clear, mock_tracing
    ):
        """Test basic memory profiling."""
        # Mock memory snapshots
        start_snapshot = MemorySnapshot(current_mb=1.0, peak_mb=1.0, allocated_blocks=100)
        end_snapshot = MemorySnapshot(current_mb=2.0, peak_mb=3.0, allocated_blocks=150)
        mock_snapshot.side_effect = [start_snapshot, end_snapshot]
        
        # Mock peak memory
        mock_memory.side_effect = [(1024*1024, 1024*1024), (1024*1024, 3*1024*1024)]
        
        with profile_memory("test_op") as profile:
            pass
            
        assert profile.operation == "test_op"
        assert profile.start_memory.current_mb == 1.0
        assert profile.end_memory.current_mb == 2.0
        assert profile.memory_delta_mb == 1.0
        assert profile.peak_memory_mb == 2.0  # (3MB - 1MB) / 1MB
        assert mock_collect.call_count == 2  # Called before and after


class TestBenchmarkFunction:
    """Test the benchmark_function utility."""
    
    def test_benchmark_function_basic(self):
        """Test basic function benchmarking."""
        def test_func(x, y):
            return x + y
            
        result, benchmark = benchmark_function(
            test_func, 1, 2, 
            operation_name="addition",
            iterations=3,
            profile_memory_usage=False
        )
        
        assert result == 3
        assert benchmark.operation == "addition"
        assert benchmark.timing.iterations == 3
        assert len(benchmark.metadata['individual_times']) == 3
        assert benchmark.memory is None
        
    @patch('astlib.benchmark.profile_memory')
    def test_benchmark_function_with_memory(self, mock_profile_memory):
        """Test function benchmarking with memory profiling."""
        # Mock memory profiling context manager
        mock_memory_profile = Mock()
        mock_profile_memory.return_value.__enter__ = Mock(return_value=mock_memory_profile)
        mock_profile_memory.return_value.__exit__ = Mock(return_value=None)
        
        def test_func():
            return "result"
            
        result, benchmark = benchmark_function(
            test_func,
            iterations=1,
            profile_memory_usage=True
        )
        
        assert result == "result"
        assert benchmark.memory == mock_memory_profile
        mock_profile_memory.assert_called_once_with("test_func")
        
    def test_benchmark_function_statistics(self):
        """Test that benchmark function calculates statistics correctly."""
        def test_func():
            time.sleep(0.001)  # 1ms sleep
            return True
            
        _, benchmark = benchmark_function(
            test_func,
            iterations=5,
            profile_memory_usage=False
        )
        
        meta = benchmark.metadata
        assert meta['iterations'] == 5
        assert len(meta['individual_times']) == 5
        assert meta['min_time'] > 0
        assert meta['max_time'] >= meta['min_time']
        assert meta['median_time'] > 0
        assert 'stdev_time' in meta


class TestBenchmarkSuite:
    """Test the BenchmarkSuite class."""
    
    def test_suite_creation(self):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite("test_suite")
        assert suite.name == "test_suite"
        assert suite.results == []
        assert suite.start_time is not None
        
    def test_add_result(self):
        """Test adding results to suite."""
        suite = BenchmarkSuite("test_suite")
        timing = TimingResult("op1", 1.0)
        result = BenchmarkResult("op1", timing)
        
        suite.add_result(result)
        assert len(suite.results) == 1
        assert suite.results[0] == result
        
    def test_run_benchmark(self):
        """Test running benchmark through suite."""
        suite = BenchmarkSuite("test_suite")
        
        def test_func(x):
            return x * 2
            
        result = suite.run_benchmark(
            test_func, 5,
            operation_name="multiply",
            iterations=2,
            profile_memory_usage=False
        )
        
        assert result == 10
        assert len(suite.results) == 1
        assert suite.results[0].operation == "multiply"
        
    def test_generate_report(self):
        """Test generating benchmark report."""
        suite = BenchmarkSuite("test_suite")
        
        # Add some results
        timing1 = TimingResult("op1", 1.0, iterations=2)
        timing2 = TimingResult("op2", 2.0, iterations=1)
        
        suite.add_result(BenchmarkResult("op1", timing1))
        suite.add_result(BenchmarkResult("op2", timing2))
        
        report = suite.generate_report()
        
        assert report['suite_name'] == "test_suite"
        assert report['summary']['total_operations'] == 2
        assert report['summary']['total_time'] == 3.0
        assert len(report['timing_results']) == 2
        
    def test_save_json_report(self, tmp_path):
        """Test saving JSON report."""
        suite = BenchmarkSuite("test_suite")
        timing = TimingResult("op1", 1.0)
        suite.add_result(BenchmarkResult("op1", timing))
        
        report_path = tmp_path / "report.json"
        suite.save_json_report(report_path)
        
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
            
        assert data['suite_name'] == "test_suite"
        assert len(data['timing_results']) == 1
        
    def test_save_markdown_report(self, tmp_path):
        """Test saving Markdown report."""
        suite = BenchmarkSuite("test_suite")
        timing = TimingResult("op1", 1.0, iterations=3)
        result = BenchmarkResult(
            "op1", timing,
            metadata={
                'individual_times': [0.3, 0.35, 0.35],
                'min_time': 0.3,
                'max_time': 0.35,
                'median_time': 0.35,
                'stdev_time': 0.024
            }
        )
        suite.add_result(result)
        
        report_path = tmp_path / "report.md"
        suite.save_markdown_report(report_path)
        
        assert report_path.exists()
        content = report_path.read_text()
        
        assert "# Benchmark Report: test_suite" in content
        assert "## Timing Results" in content
        assert "| op1 |" in content
        assert "Total Operations: 1" in content


class TestCompareBenchmarks:
    """Test benchmark comparison functionality."""
    
    def test_compare_benchmarks_no_regression(self, tmp_path):
        """Test comparing benchmarks with no regression."""
        baseline_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 1.0},
                {'operation': 'op2', 'average_seconds': 2.0}
            ]
        }
        
        current_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 1.05},  # 5% slower
                {'operation': 'op2', 'average_seconds': 1.9}   # 5% faster
            ]
        }
        
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        with open(current_path, 'w') as f:
            json.dump(current_data, f)
            
        comparison = compare_benchmarks(baseline_path, current_path)
        
        assert comparison['summary']['total_operations'] == 2
        assert comparison['summary']['regressions_count'] == 0
        assert comparison['summary']['improvements_count'] == 0
        assert comparison['summary']['stable_count'] == 2
        
    def test_compare_benchmarks_with_regression(self, tmp_path):
        """Test comparing benchmarks with regression detected."""
        baseline_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 1.0}
            ]
        }
        
        current_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 1.2}  # 20% slower - regression!
            ]
        }
        
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        with open(current_path, 'w') as f:
            json.dump(current_data, f)
            
        comparison = compare_benchmarks(baseline_path, current_path)
        
        assert comparison['summary']['regressions_count'] == 1
        assert len(comparison['regressions']) == 1
        assert comparison['regressions'][0]['operation'] == 'op1'
        assert abs(comparison['regressions'][0]['change_percent'] - 20.0) < 0.001
        
    def test_compare_benchmarks_with_improvement(self, tmp_path):
        """Test comparing benchmarks with improvement detected."""
        baseline_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 2.0}
            ]
        }
        
        current_data = {
            'timing_results': [
                {'operation': 'op1', 'average_seconds': 1.5}  # 25% faster - improvement!
            ]
        }
        
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        with open(current_path, 'w') as f:
            json.dump(current_data, f)
            
        comparison = compare_benchmarks(baseline_path, current_path)
        
        assert comparison['summary']['improvements_count'] == 1
        assert len(comparison['improvements']) == 1
        assert comparison['improvements'][0]['operation'] == 'op1'
        assert comparison['improvements'][0]['change_percent'] == -25.0