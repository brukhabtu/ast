"""Integration tests for cache performance improvements."""

import time
import tempfile
from pathlib import Path
import pytest

from astlib.parser import parse_file
from astlib.symbols import extract_symbols
from astlib.cache import get_cache_manager, clear_caches, get_cache_stats
from astlib.benchmark import benchmark_function, BenchmarkSuite


class TestCachePerformance:
    """Test cache performance improvements."""
    
    @pytest.fixture(autouse=True)
    def setup(self, request):
        """Clear caches before each test."""
        # Skip setup for test_real_world_scenario
        if request.node.name != "test_real_world_scenario":
            clear_caches()
        yield
        # Always clear after test
        clear_caches()
    
    def test_parse_file_caching(self, tmp_path):
        """Test that parse_file caching improves performance."""
        # Create a test file with substantial content
        test_file = tmp_path / "test_module.py"
        test_file.write_text('''
import os
import sys
from typing import List, Dict, Optional

class DataProcessor:
    """A complex class for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.data: List[Dict[str, Any]] = []
        
    def process(self, items: List[str]) -> Dict[str, int]:
        """Process items and return counts."""
        result = {}
        for item in items:
            result[item] = len(item)
        return result
    
    def analyze(self, data: Dict[str, Any]) -> Optional[str]:
        """Analyze data and return summary."""
        if not data:
            return None
        
        total = sum(data.values())
        average = total / len(data)
        
        return f"Total: {total}, Average: {average:.2f}"

def helper_function(x: int, y: int) -> int:
    """Helper function for calculations."""
    return x * y + x - y

def main():
    """Main entry point."""
    processor = DataProcessor("test")
    data = processor.process(["hello", "world", "test"])
    summary = processor.analyze(data)
    print(summary)

if __name__ == "__main__":
    main()
''' * 10)  # Make it larger for more noticeable timing differences
        
        # Benchmark without cache (first call)
        _, first_call_bench = benchmark_function(
            parse_file,
            test_file,
            operation_name="parse_file_first_call",
            iterations=1,
            profile_memory_usage=False
        )
        
        # Benchmark with cache (subsequent calls)
        _, cached_calls_bench = benchmark_function(
            parse_file,
            test_file,
            operation_name="parse_file_cached",
            iterations=10,
            profile_memory_usage=False
        )
        
        # Calculate speedup
        first_call_time = first_call_bench.timing.duration_seconds
        avg_cached_time = cached_calls_bench.timing.average_duration
        speedup = first_call_time / avg_cached_time
        
        # Check cache stats
        stats = get_cache_stats()
        ast_stats = stats['ast_cache']
        
        # Assertions
        assert ast_stats['hits'] >= 10  # At least 10 cache hits
        assert ast_stats['misses'] == 1  # Only one miss (first call)
        assert speedup > 5.0  # At least 5x speedup with cache
        
        print(f"\nParse file caching results:")
        print(f"  First call: {first_call_time*1000:.2f}ms")
        print(f"  Cached call avg: {avg_cached_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Cache hit rate: {ast_stats['hit_rate']:.1%}")
    
    def test_symbol_extraction_caching(self, tmp_path):
        """Test that symbol extraction caching improves performance."""
        # Create test file
        test_file = tmp_path / "symbols_test.py"
        test_file.write_text('''
class BaseClass:
    def base_method(self): pass

class DerivedClass(BaseClass):
    def __init__(self):
        self.attribute = 42
        
    def method1(self, x: int) -> int:
        return x * 2
        
    def method2(self, y: str) -> str:
        return y.upper()
        
    @property
    def computed_value(self) -> int:
        return self.attribute * 2

def function1(a, b, c):
    return a + b + c

def function2(x: List[int]) -> int:
    return sum(x)

async def async_function():
    await something()

CONSTANT = 100
variable = "test"
''' * 5)  # Repeat for more symbols
        
        # Parse file once
        parse_result = parse_file(test_file)
        ast_tree = parse_result.tree
        
        # Benchmark symbol extraction
        def extract_symbols_wrapper():
            return extract_symbols(ast_tree, str(test_file))
        
        # First extraction (no cache)
        _, first_extract_bench = benchmark_function(
            extract_symbols_wrapper,
            operation_name="extract_symbols_first",
            iterations=1,
            profile_memory_usage=False
        )
        
        # Subsequent extractions (cached)
        _, cached_extract_bench = benchmark_function(
            extract_symbols_wrapper,
            operation_name="extract_symbols_cached",
            iterations=20,
            profile_memory_usage=False
        )
        
        # Calculate speedup
        first_time = first_extract_bench.timing.duration_seconds
        avg_cached_time = cached_extract_bench.timing.average_duration
        speedup = first_time / avg_cached_time
        
        # Check cache stats
        stats = get_cache_stats()
        symbol_stats = stats['symbol_cache']
        
        # Assertions
        assert symbol_stats['hits'] >= 20
        assert symbol_stats['misses'] == 1
        assert speedup > 3.0  # Symbol extraction should have good speedup
        
        print(f"\nSymbol extraction caching results:")
        print(f"  First extraction: {first_time*1000:.2f}ms")
        print(f"  Cached extraction avg: {avg_cached_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Cache hit rate: {symbol_stats['hit_rate']:.1%}")
    
    def test_file_modification_invalidation(self, tmp_path):
        """Test that cache is invalidated when files are modified."""
        test_file = tmp_path / "mod_test.py"
        test_file.write_text("x = 1")
        
        # First parse
        result1 = parse_file(test_file)
        
        # Should hit cache
        result2 = parse_file(test_file)
        assert result1 is result2  # Same object from cache
        
        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("x = 2")
        
        # Should miss cache and get new result
        result3 = parse_file(test_file)
        assert result3 is not result1  # Different object
        
        # Verify content changed
        assert "x = 1" in result1.source
        assert "x = 2" in result3.source
        
        # Check stats
        stats = get_cache_stats()['ast_cache']
        assert stats['hits'] == 1  # One hit before modification
        assert stats['misses'] == 2  # Initial miss + miss after modification
    
    def test_memory_limits(self, tmp_path):
        """Test that cache respects memory limits."""
        # Get current manager and set small limits
        manager = get_cache_manager()
        manager.ast_cache.max_size_bytes = 1024 * 100  # 100KB
        manager.ast_cache.max_entries = 5
        
        # Create multiple files
        files = []
        for i in range(10):
            test_file = tmp_path / f"file_{i}.py"
            # Create files with different content to avoid deduplication
            test_file.write_text(f'''
def function_{i}():
    """Function {i} with unique content."""
    data = "{i}" * 1000  # Make it larger
    return data * {i}
''')
            files.append(test_file)
        
        # Parse all files
        for f in files:
            parse_file(f)
        
        # Check that cache has evicted some entries
        stats = get_cache_stats()['ast_cache']
        assert stats['entry_count'] <= 5  # Should respect max entries
        assert stats['evictions'] > 0  # Should have evicted some entries
        
        print(f"\nMemory limit test results:")
        print(f"  Files parsed: {len(files)}")
        print(f"  Cache entries: {stats['entry_count']}")
        print(f"  Evictions: {stats['evictions']}")
        print(f"  Cache size: {stats['total_size_mb']:.2f}MB")
    
    def test_concurrent_access(self, tmp_path):
        """Test cache performance with concurrent access."""
        import threading
        import queue
        
        # Create test files
        num_files = 5
        files = []
        for i in range(num_files):
            test_file = tmp_path / f"concurrent_{i}.py"
            test_file.write_text(f'''
def function_{i}(x):
    return x * {i}
    
class Class_{i}:
    pass
''')
            files.append(test_file)
        
        # Thread worker function
        results_queue = queue.Queue()
        
        def worker(file_path, thread_id):
            start = time.perf_counter()
            for _ in range(10):
                result = parse_file(file_path)
                if result.tree:
                    symbols = extract_symbols(result.tree, str(file_path))
            duration = time.perf_counter() - start
            results_queue.put((thread_id, duration))
        
        # Run threads
        threads = []
        start_time = time.perf_counter()
        
        for i in range(num_files):
            t = threading.Thread(target=worker, args=(files[i], i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        total_duration = time.perf_counter() - start_time
        
        # Collect results
        thread_times = []
        while not results_queue.empty():
            thread_id, duration = results_queue.get()
            thread_times.append(duration)
        
        # Check cache effectiveness
        stats = get_cache_stats()
        total_hits = stats['ast_cache']['hits'] + stats['symbol_cache']['hits']
        total_accesses = (stats['ast_cache']['hits'] + stats['ast_cache']['misses'] +
                         stats['symbol_cache']['hits'] + stats['symbol_cache']['misses'])
        
        # With caching, should be much faster than serial execution
        avg_thread_time = sum(thread_times) / len(thread_times)
        
        print(f"\nConcurrent access results:")
        print(f"  Threads: {num_files}")
        print(f"  Total duration: {total_duration:.3f}s")
        print(f"  Avg thread time: {avg_thread_time:.3f}s")
        print(f"  Total cache hits: {total_hits}")
        print(f"  Overall hit rate: {total_hits/total_accesses:.1%}")
        
        # Should have good cache hit rate
        assert total_hits / total_accesses > 0.8  # 80%+ hit rate
    
    def test_real_world_scenario(self):
        """Test cache performance in a real-world scenario."""
        # Use the AST library's own files
        project_root = Path(__file__).parent.parent.parent
        astlib_path = project_root / "astlib"
        
        if not astlib_path.exists():
            pytest.skip("astlib directory not found")
        
        # Find Python files
        python_files = list(astlib_path.glob("*.py"))[:10]
        
        if not python_files:
            pytest.skip("No Python files found")
        
        # Measure time without cache
        clear_caches()
        start_no_cache = time.time()
        for f in python_files:
            result = parse_file(f)
            if result.tree:
                extract_symbols(result.tree, str(f))
        time_no_cache = time.time() - start_no_cache
        
        # Measure time with cache (second pass)
        start_with_cache = time.time()
        for f in python_files:
            result = parse_file(f)
            if result.tree:
                extract_symbols(result.tree, str(f))
        time_with_cache = time.time() - start_with_cache
        
        # Get cache stats after operations
        final_stats = get_cache_stats()
        
        # Calculate speedup
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 1.0
        
        print(f"\nReal-world scenario results:")
        print(f"  Files processed: {len(python_files)}")
        print(f"  Time without cache: {time_no_cache:.3f}s")
        print(f"  Time with cache: {time_with_cache:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  AST cache hit rate: {final_stats['ast_cache']['hit_rate']:.1%}")
        print(f"  Symbol cache hit rate: {final_stats['symbol_cache']['hit_rate']:.1%}")
        
        # Should have speedup
        assert speedup > 2.0  # At least 2x speedup
        # Note: Due to test isolation, cache stats might be cleared by fixture
        # The important metric is the speedup which proves caching works


def test_property_based_cache_consistency():
    """Property-based test for cache consistency."""
    try:
        from hypothesis import given, strategies as st
    except ImportError:
        pytest.skip("hypothesis not installed")
    import tempfile
    
    @given(
        content=st.text(min_size=1, max_size=100),
        modifications=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=5)
    )
    def test_cache_consistency(content, modifications):
        """Test that cache maintains consistency through modifications."""
        clear_caches()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            
            # Initial content
            test_file.write_text(f"# {content}")
            result1 = parse_file(test_file)
            
            # Should get same result from cache
            result2 = parse_file(test_file)
            assert result1 is result2
            
            # Apply modifications
            for i, mod in enumerate(modifications):
                time.sleep(0.01)  # Ensure mtime changes
                test_file.write_text(f"# {mod}")
                
                # Should get new result after modification
                result_new = parse_file(test_file)
                assert result_new is not result1
                assert f"# {mod}" in result_new.source
                
                # Update reference
                result1 = result_new
    
    # Run the property test
    test_cache_consistency()