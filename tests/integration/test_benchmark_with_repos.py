"""Integration tests for benchmarking with test repositories."""

import ast
import tempfile
from pathlib import Path
import pytest

from astlib.benchmark import BenchmarkSuite, compare_benchmarks
from tests.fixtures.mock_repos import create_all_mock_repos


class TestBenchmarkIntegration:
    """Integration tests for benchmark functionality with mock repos."""
    
    @pytest.fixture
    def mock_repos_dir(self):
        """Create temporary directory with mock repositories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            create_all_mock_repos(temp_path)
            yield temp_path
    
    def parse_python_files(self, repo_path: Path) -> int:
        """Parse all Python files in a repository.
        
        Returns:
            Number of files successfully parsed
        """
        count = 0
        for py_file in repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
                count += 1
            except:
                pass
        return count
    
    def test_benchmark_mock_repos(self, mock_repos_dir):
        """Test benchmarking AST parsing on mock repositories."""
        suite = BenchmarkSuite("Mock Repository Benchmark")
        
        for repo_path in mock_repos_dir.iterdir():
            if repo_path.is_dir() and repo_path.name != "__pycache__":
                # Benchmark parsing all Python files in the repo
                count = suite.run_benchmark(
                    self.parse_python_files,
                    repo_path,
                    operation_name=f"parse_{repo_path.name}",
                    iterations=3,
                    profile_memory_usage=True
                )
                
                assert count > 0, f"Should have parsed files in {repo_path.name}"
        
        # Verify we benchmarked all repos
        assert len(suite.results) == 4  # django, flask, fastapi, requests
        
        # Check that all benchmarks completed successfully
        for result in suite.results:
            assert result.timing.duration_seconds > 0
            assert result.timing.iterations == 3
            assert result.memory is not None
            assert result.memory.memory_delta_mb >= 0
    
    def test_generate_performance_reports(self, mock_repos_dir, tmp_path):
        """Test generating both JSON and Markdown reports."""
        suite = BenchmarkSuite("Performance Report Test")
        
        # Run a simple benchmark
        suite.run_benchmark(
            self.parse_python_files,
            mock_repos_dir / "django",
            operation_name="parse_django",
            iterations=2,
            profile_memory_usage=True
        )
        
        # Generate reports
        json_path = tmp_path / "benchmark.json"
        md_path = tmp_path / "benchmark.md"
        
        suite.save_json_report(json_path)
        suite.save_markdown_report(md_path)
        
        # Verify JSON report
        assert json_path.exists()
        import json
        with open(json_path) as f:
            json_data = json.load(f)
        
        assert json_data['suite_name'] == "Performance Report Test"
        assert len(json_data['timing_results']) == 1
        assert len(json_data['memory_results']) == 1
        assert json_data['timing_results'][0]['operation'] == "parse_django"
        
        # Verify Markdown report
        assert md_path.exists()
        md_content = md_path.read_text()
        
        assert "# Benchmark Report: Performance Report Test" in md_content
        assert "## Timing Results" in md_content
        assert "## Memory Results" in md_content
        assert "parse_django" in md_content
    
    def test_regression_detection(self, mock_repos_dir, tmp_path):
        """Test regression detection between benchmark runs."""
        # First run - baseline
        baseline_suite = BenchmarkSuite("Baseline")
        baseline_suite.run_benchmark(
            lambda: sum(range(1000000)),  # Simple computation
            operation_name="computation",
            iterations=3,
            profile_memory_usage=False
        )
        
        baseline_path = tmp_path / "baseline.json"
        baseline_suite.save_json_report(baseline_path)
        
        # Second run - current (simulate slower performance)
        current_suite = BenchmarkSuite("Current")
        
        def slower_computation():
            """Simulate a regression with slower computation."""
            result = sum(range(1000000))
            # Add some overhead to simulate regression
            for _ in range(10):
                _ = [i**2 for i in range(1000)]
            return result
        
        current_suite.run_benchmark(
            slower_computation,
            operation_name="computation",
            iterations=3,
            profile_memory_usage=False
        )
        
        current_path = tmp_path / "current.json"
        current_suite.save_json_report(current_path)
        
        # Compare benchmarks
        comparison = compare_benchmarks(baseline_path, current_path)
        
        assert comparison['summary']['total_operations'] == 1
        assert len(comparison['comparisons']) == 1
        
        # The slower computation should show increased time
        comp = comparison['comparisons'][0]
        assert comp['operation'] == "computation"
        assert comp['current_avg'] > comp['baseline_avg']
        
    def test_memory_profiling_accuracy(self, mock_repos_dir):
        """Test that memory profiling provides reasonable results."""
        suite = BenchmarkSuite("Memory Test")
        
        def memory_intensive_operation():
            """Operation that allocates significant memory."""
            # Create a large list
            data = [i for i in range(1000000)]
            # Create a dictionary
            mapping = {i: str(i) for i in range(100000)}
            return len(data) + len(mapping)
        
        result = suite.run_benchmark(
            memory_intensive_operation,
            operation_name="memory_test",
            iterations=1,
            profile_memory_usage=True
        )
        
        assert result > 0
        
        # Check memory profiling results
        benchmark = suite.results[0]
        assert benchmark.memory is not None
        assert benchmark.memory.peak_memory_mb > 0
        assert benchmark.memory.end_memory.allocated_blocks > benchmark.memory.start_memory.allocated_blocks
    
    def test_benchmark_with_exceptions(self, mock_repos_dir):
        """Test benchmarking handles exceptions gracefully."""
        suite = BenchmarkSuite("Exception Test")
        
        def failing_operation():
            """Operation that raises an exception."""
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            suite.run_benchmark(
                failing_operation,
                operation_name="failing_op",
                iterations=1,
                profile_memory_usage=False
            )
        
        # Suite should not have recorded the failed benchmark
        assert len(suite.results) == 0