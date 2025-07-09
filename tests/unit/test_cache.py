"""Unit tests for the caching system."""

import os
import tempfile
import time
import threading
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

from astlib.cache import (
    CacheEntry, CacheStats, LRUCache, CacheManager,
    cached_operation, get_cache_manager, clear_caches, get_cache_stats
)


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            size_bytes=100,
            created_at=time.time(),
            last_accessed=time.time(),
            file_mtime=123.456
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.size_bytes == 100
        assert entry.access_count == 0
        assert entry.file_mtime == 123.456
    
    def test_cache_entry_staleness(self, tmp_path):
        """Test cache entry staleness checking."""
        # Create a temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        
        # Create entry with current mtime
        mtime = test_file.stat().st_mtime
        entry = CacheEntry(
            key="test",
            value="data",
            size_bytes=10,
            created_at=time.time(),
            last_accessed=time.time(),
            file_mtime=mtime
        )
        
        # Should not be stale initially
        assert not entry.is_stale(test_file)
        
        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("# modified")
        
        # Should now be stale
        assert entry.is_stale(test_file)
    
    def test_cache_entry_access(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(
            key="test",
            value="data",
            size_bytes=10,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        initial_time = entry.last_accessed
        initial_count = entry.access_count
        
        time.sleep(0.01)
        entry.access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestCacheStats:
    """Test CacheStats functionality."""
    
    def test_cache_stats_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75
        
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0
    
    def test_cache_stats_size_mb(self):
        """Test size conversion to MB."""
        stats = CacheStats(total_size_bytes=1024 * 1024 * 10)  # 10 MB
        assert stats.size_mb == 10.0
    
    def test_cache_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(
            hits=100,
            misses=50,
            evictions=10,
            total_size_bytes=1024 * 1024,
            entry_count=20
        )
        
        data = stats.to_dict()
        assert data['hits'] == 100
        assert data['misses'] == 50
        assert data['evictions'] == 10
        assert data['hit_rate'] == 100 / 150
        assert data['total_size_mb'] == 1.0
        assert data['entry_count'] == 20


class TestLRUCache:
    """Test LRUCache functionality."""
    
    def test_basic_get_put(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size_mb=1.0, max_entries=10)
        
        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Cache miss
        assert cache.get("key2") is None
        
        # Stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count == 1
    
    def test_lru_eviction_by_count(self):
        """Test LRU eviction when max entries reached."""
        cache = LRUCache(max_size_mb=100.0, max_entries=3)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new entry, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New entry
        
        stats = cache.get_stats()
        assert stats.evictions == 1
    
    def test_lru_eviction_by_size(self):
        """Test LRU eviction when max size reached."""
        cache = LRUCache(max_size_mb=0.0001, max_entries=100)  # Very small size limit
        
        # Put large objects
        large_data = "x" * 1000
        cache.put("key1", large_data)
        cache.put("key2", large_data)
        
        # Should have evicted key1
        assert cache.get("key1") is None
        assert cache.get("key2") == large_data
        
        stats = cache.get_stats()
        assert stats.evictions >= 1
    
    def test_file_staleness_checking(self, tmp_path):
        """Test cache invalidation on file changes."""
        cache = LRUCache()
        
        # Create temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("# original")
        
        # Put with file path
        cache.put("key1", "parsed_data", test_file)
        
        # Should get from cache
        assert cache.get("key1", test_file) == "parsed_data"
        
        # Modify file
        time.sleep(0.01)
        test_file.write_text("# modified")
        
        # Should return None due to staleness
        assert cache.get("key1", test_file) is None
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache = LRUCache()
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.put(key, f"value_{i}")
                    value = cache.get(key)
                    assert value == f"value_{i}"
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert cache.get_stats().entry_count > 0
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = LRUCache()
        
        # Add entries
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
        
        assert cache.get_stats().entry_count == 5
        
        # Clear
        cache.clear()
        
        # Check cleared
        stats = cache.get_stats()
        assert stats.entry_count == 0
        assert stats.total_size_bytes == 0
        
        # Old entries should be gone
        assert cache.get("key0") is None


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_singleton_pattern(self):
        """Test that CacheManager is a singleton."""
        manager1 = CacheManager()
        manager2 = CacheManager()
        assert manager1 is manager2
    
    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation."""
        manager = CacheManager()
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        
        # AST cache key should be deterministic
        key1 = manager.get_ast_cache_key(test_file)
        key2 = manager.get_ast_cache_key(test_file)
        assert key1 == key2
        
        # Symbol cache key should include AST ID
        ast_obj1 = Mock()
        ast_obj2 = Mock()
        
        sym_key1 = manager.get_symbol_cache_key(test_file, id(ast_obj1))
        sym_key2 = manager.get_symbol_cache_key(test_file, id(ast_obj2))
        assert sym_key1 != sym_key2
    
    def test_ast_caching(self, tmp_path):
        """Test AST caching functionality."""
        manager = CacheManager()
        manager.clear_all_caches()
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        # Cache AST
        parse_result = {"tree": "mock_ast", "errors": []}
        manager.cache_ast(test_file, parse_result)
        
        # Retrieve from cache
        cached = manager.get_cached_ast(test_file)
        assert cached == parse_result
        
        # Modify file
        time.sleep(0.01)
        test_file.write_text("x = 2")
        
        # Should be invalidated
        assert manager.get_cached_ast(test_file) is None
    
    def test_symbol_caching(self, tmp_path):
        """Test symbol caching functionality."""
        manager = CacheManager()
        manager.clear_all_caches()
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")
        
        # Mock AST
        ast_obj = Mock()
        symbols = [{"name": "foo", "type": "function"}]
        
        # Cache symbols
        manager.cache_symbols(test_file, ast_obj, symbols)
        
        # Retrieve from cache
        cached = manager.get_cached_symbols(test_file, ast_obj)
        assert cached == symbols
        
        # Different AST object should miss
        ast_obj2 = Mock()
        assert manager.get_cached_symbols(test_file, ast_obj2) is None
    
    def test_get_all_stats(self):
        """Test getting stats for all caches."""
        manager = CacheManager()
        manager.clear_all_caches()
        
        # Add some data
        manager.ast_cache.put("test1", "value1")
        manager.symbol_cache.put("test2", "value2")
        
        stats = manager.get_all_stats()
        
        assert 'ast_cache' in stats
        assert 'symbol_cache' in stats
        assert stats['ast_cache']['entry_count'] == 1
        assert stats['symbol_cache']['entry_count'] == 1


class TestCachedOperationDecorator:
    """Test cached_operation decorator."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_caches()
        yield
        clear_caches()
    
    def test_ast_cache_decorator(self, tmp_path):
        """Test caching AST operations."""
        manager = get_cache_manager()
        manager.clear_all_caches()
        
        call_count = 0
        
        @cached_operation(cache_type="ast")
        def parse_mock_file(filepath):
            nonlocal call_count
            call_count += 1
            return {"ast": "mock", "count": call_count}
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        
        # First call - should execute
        result1 = parse_mock_file(test_file)
        assert result1["count"] == 1
        
        # Second call - should use cache
        result2 = parse_mock_file(test_file)
        assert result2["count"] == 1  # Same result
        assert call_count == 1  # Function only called once
    
    def test_symbol_cache_decorator(self, tmp_path):
        """Test caching symbol operations."""
        manager = get_cache_manager()
        manager.clear_all_caches()
        
        call_count = 0
        
        @cached_operation(cache_type="symbol")
        def extract_mock_symbols(ast_tree, file_path=None):
            nonlocal call_count
            call_count += 1
            return [{"symbol": "mock", "count": call_count}]
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")
        
        # Use a stable object (not Mock which creates new instances)
        ast_obj = {"type": "ast", "id": 12345}  # Stable object
        
        # First call - should execute
        result1 = extract_mock_symbols(ast_obj, file_path=str(test_file))
        assert result1[0]["count"] == 1
        
        # Second call - should use cache
        result2 = extract_mock_symbols(ast_obj, file_path=str(test_file))
        assert result2[0]["count"] == 1  # Same result
        assert call_count == 1  # Function only called once


class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_caches()
        yield
        clear_caches()
    
    def test_get_cache_manager(self):
        """Test getting cache manager."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2
        assert isinstance(manager1, CacheManager)
    
    def test_clear_caches(self):
        """Test clearing all caches."""
        manager = get_cache_manager()
        
        # Add some data
        manager.ast_cache.put("test1", "value1")
        manager.symbol_cache.put("test2", "value2")
        
        # Clear
        clear_caches()
        
        # Check cleared
        stats = get_cache_stats()
        assert stats['ast_cache']['entry_count'] == 0
        assert stats['symbol_cache']['entry_count'] == 0
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        clear_caches()
        
        # Add some data
        manager = get_cache_manager()
        manager.ast_cache.put("test1", "value1")
        _ = manager.ast_cache.get("test1")  # Hit
        _ = manager.ast_cache.get("test2")  # Miss
        
        stats = get_cache_stats()
        assert stats['ast_cache']['hits'] == 1
        assert stats['ast_cache']['misses'] == 1
        assert stats['ast_cache']['entry_count'] == 1