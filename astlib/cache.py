"""Caching system for AST library to improve performance.

This module provides:
- LRU cache with memory limits
- File modification time tracking
- Thread-safe operations
- Cache statistics and monitoring
"""

import functools
import hashlib
import os
import pickle
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import weakref

# Type variable for generic cache
T = TypeVar('T')


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    file_mtime: Optional[float] = None
    
    def is_stale(self, file_path: Optional[Path] = None) -> bool:
        """Check if cache entry is stale based on file modification time."""
        if file_path and file_path.exists() and self.file_mtime is not None:
            current_mtime = file_path.stat().st_mtime
            return current_mtime > self.file_mtime
        return False
    
    def access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'total_size_mb': self.size_mb,
            'entry_count': self.entry_count
        }


class LRUCache:
    """Thread-safe LRU cache with memory limits."""
    
    def __init__(self, max_size_mb: float = 100.0, max_entries: int = 1000):
        """Initialize LRU cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        # Use sys.getsizeof for basic estimation
        # For complex objects, this is an approximation
        try:
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            else:
                # Rough estimation using pickle
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback to basic size
            return sys.getsizeof(obj)
    
    def get(self, key: str, file_path: Optional[Path] = None) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            file_path: Optional file path to check staleness
            
        Returns:
            Cached value or None if not found/stale
        """
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if stale
            if entry.is_stale(file_path):
                self._remove(key)
                self.stats.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.access()
            
            self.stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, file_path: Optional[Path] = None) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            file_path: Optional file path for staleness checking
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self._remove(key)
            
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                file_mtime=file_path.stat().st_mtime if file_path and file_path.exists() else None
            )
            
            # Evict if necessary
            while (self.stats.total_size_bytes + size_bytes > self.max_size_bytes or
                   len(self.cache) >= self.max_entries) and self.cache:
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self.cache)
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self.cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            # First item is LRU
            key, entry = self.cache.popitem(last=False)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
            self.stats.entry_count = len(self.cache)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()  # Reset all stats
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size_bytes=self.stats.total_size_bytes,
                entry_count=self.stats.entry_count
            )


class CacheManager:
    """Global cache manager for AST operations."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize cache manager."""
        if not hasattr(self, '_initialized'):
            self.ast_cache = LRUCache(max_size_mb=50.0, max_entries=500)
            self.symbol_cache = LRUCache(max_size_mb=25.0, max_entries=1000)
            self._initialized = True
    
    def get_ast_cache_key(self, file_path: Path) -> str:
        """Generate cache key for AST parsing."""
        # Include file path and modification time in key
        mtime = file_path.stat().st_mtime if file_path.exists() else 0
        key_data = f"{file_path.absolute()}:{mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get_symbol_cache_key(self, file_path: Path, ast_id: int) -> str:
        """Generate cache key for symbol extraction."""
        # Include file path and AST object ID
        key_data = f"{file_path.absolute()}:{ast_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def cache_ast(self, file_path: Path, parse_result: Any) -> None:
        """Cache AST parse result."""
        key = self.get_ast_cache_key(file_path)
        self.ast_cache.put(key, parse_result, file_path)
    
    def get_cached_ast(self, file_path: Path) -> Optional[Any]:
        """Get cached AST parse result."""
        key = self.get_ast_cache_key(file_path)
        return self.ast_cache.get(key, file_path)
    
    def cache_symbols(self, file_path: Path, ast_tree: Any, symbols: Any) -> None:
        """Cache symbol extraction result."""
        key = self.get_symbol_cache_key(file_path, id(ast_tree))
        self.symbol_cache.put(key, symbols, file_path)
    
    def get_cached_symbols(self, file_path: Path, ast_tree: Any) -> Optional[Any]:
        """Get cached symbol extraction result."""
        key = self.get_symbol_cache_key(file_path, id(ast_tree))
        return self.symbol_cache.get(key, file_path)
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.ast_cache.clear()
        self.symbol_cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            'ast_cache': self.ast_cache.get_stats().to_dict(),
            'symbol_cache': self.symbol_cache.get_stats().to_dict()
        }


def cached_operation(cache_type: str = "ast"):
    """Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use ("ast" or "symbol")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            manager = CacheManager()
            
            # Determine cache and key generation
            if cache_type == "ast":
                # First arg should be file path
                if args and isinstance(args[0], (str, Path)):
                    file_path = Path(args[0])
                    cache_key = manager.get_ast_cache_key(file_path)
                    
                    # Try cache
                    cached = manager.ast_cache.get(cache_key, file_path)
                    if cached is not None:
                        return cached
                    
                    # Call function
                    result = func(*args, **kwargs)
                    
                    # Cache result
                    manager.ast_cache.put(cache_key, result, file_path)
                    return result
            
            elif cache_type == "symbol":
                # Need file path and AST tree
                if len(args) >= 1:  # Changed from >= 2
                    ast_tree = args[0]
                    file_path = kwargs.get('file_path') or (args[1] if len(args) > 1 else None)
                    
                    if file_path:
                        file_path = Path(file_path)
                        cache_key = manager.get_symbol_cache_key(file_path, id(ast_tree))
                        
                        # Try cache
                        cached = manager.symbol_cache.get(cache_key, file_path)
                        if cached is not None:
                            return cached
                        
                        # Call function
                        result = func(*args, **kwargs)
                        
                        # Cache result
                        manager.symbol_cache.put(cache_key, result, file_path)
                        return result
            
            # Fallback to uncached call
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Create module-level functions for easy access
def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager()


def clear_caches() -> None:
    """Clear all caches."""
    get_cache_manager().clear_all_caches()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get cache statistics."""
    return get_cache_manager().get_all_stats()