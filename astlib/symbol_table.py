"""
Symbol table for fast symbol lookup.

This module provides an in-memory symbol table with fast lookup capabilities
for navigating Python codebases.
"""

from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict
from dataclasses import dataclass, field
import time
from .symbols import Symbol, SymbolType


@dataclass
class LookupResult:
    """Result of a symbol lookup operation."""
    symbol: Symbol
    lookup_time_ms: float
    
    
@dataclass 
class SymbolQuery:
    """Query parameters for symbol lookups."""
    name: Optional[str] = None
    type: Optional[SymbolType] = None
    file_path: Optional[str] = None
    parent_name: Optional[str] = None
    has_decorator: Optional[str] = None
    is_private: Optional[bool] = None
    is_async: Optional[bool] = None


class SymbolTable:
    """
    In-memory symbol table for fast lookups.
    
    Provides multiple indexes for different query patterns:
    - By name (exact and prefix)
    - By type
    - By file
    - By qualified name
    """
    
    def __init__(self):
        # Primary storage
        self._symbols: List[Symbol] = []
        
        # Indexes for fast lookup
        self._by_name: Dict[str, List[Symbol]] = defaultdict(list)
        self._by_type: Dict[SymbolType, List[Symbol]] = defaultdict(list)
        self._by_file: Dict[str, List[Symbol]] = defaultdict(list)
        self._by_qualified_name: Dict[str, Symbol] = {}
        
        # Additional indexes
        self._by_decorator: Dict[str, List[Symbol]] = defaultdict(list)
        self._private_symbols: Set[Symbol] = set()
        self._async_symbols: Set[Symbol] = set()
        
        # Cache for common queries
        self._query_cache: Dict[str, List[Symbol]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the table."""
        self._symbols.append(symbol)
        
        # Update indexes
        self._by_name[symbol.name].append(symbol)
        self._by_type[symbol.type].append(symbol)
        
        if symbol.file_path:
            self._by_file[symbol.file_path].append(symbol)
        
        self._by_qualified_name[symbol.qualified_name] = symbol
        
        # Update decorator index
        for decorator in symbol.decorators:
            self._by_decorator[decorator].append(symbol)
        
        # Update special indexes
        if symbol.is_private:
            self._private_symbols.add(symbol)
        
        if symbol.is_async:
            self._async_symbols.add(symbol)
        
        # Clear cache on update
        self._query_cache.clear()
    
    def add_symbols(self, symbols: List[Symbol]) -> None:
        """Add multiple symbols to the table."""
        for symbol in symbols:
            self.add_symbol(symbol)
    
    def find_by_name(self, name: str, exact: bool = True) -> List[LookupResult]:
        """
        Find symbols by name.
        
        Args:
            name: The name to search for
            exact: If True, only exact matches. If False, prefix matches.
            
        Returns:
            List of matching symbols with lookup timing
        """
        start_time = time.perf_counter()
        
        if exact:
            results = self._by_name.get(name, [])
        else:
            # Prefix search
            results = []
            for sym_name, symbols in self._by_name.items():
                if sym_name.startswith(name):
                    results.extend(symbols)
        
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def find_by_type(self, symbol_type: SymbolType) -> List[LookupResult]:
        """Find all symbols of a specific type."""
        start_time = time.perf_counter()
        results = self._by_type.get(symbol_type, [])
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def find_by_file(self, file_path: str) -> List[LookupResult]:
        """Find all symbols in a specific file."""
        start_time = time.perf_counter()
        results = self._by_file.get(file_path, [])
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def find_by_qualified_name(self, qualified_name: str) -> Optional[LookupResult]:
        """Find a symbol by its fully qualified name."""
        start_time = time.perf_counter()
        symbol = self._by_qualified_name.get(qualified_name)
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return LookupResult(symbol, lookup_time_ms) if symbol else None
    
    def find_by_decorator(self, decorator_name: str) -> List[LookupResult]:
        """Find all symbols with a specific decorator."""
        start_time = time.perf_counter()
        results = self._by_decorator.get(decorator_name, [])
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def find_private_symbols(self) -> List[LookupResult]:
        """Find all private symbols (starting with _)."""
        start_time = time.perf_counter()
        results = list(self._private_symbols)
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def find_async_symbols(self) -> List[LookupResult]:
        """Find all async functions/methods."""
        start_time = time.perf_counter()
        results = list(self._async_symbols)
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in results]
    
    def query(self, query: SymbolQuery) -> List[LookupResult]:
        """
        Execute a complex query with multiple filters.
        
        Args:
            query: SymbolQuery object with filter parameters
            
        Returns:
            List of matching symbols
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query)
        
        # Check cache
        if cache_key in self._query_cache:
            self._cache_hits += 1
            start_time = time.perf_counter()
            cached_results = self._query_cache[cache_key]
            lookup_time_ms = (time.perf_counter() - start_time) * 1000
            return [LookupResult(symbol, lookup_time_ms) for symbol in cached_results]
        
        self._cache_misses += 1
        start_time = time.perf_counter()
        
        # Start with all symbols
        results = set(self._symbols)
        
        # Apply filters
        if query.name:
            name_matches = set(self._by_name.get(query.name, []))
            results &= name_matches
        
        if query.type:
            type_matches = set(self._by_type.get(query.type, []))
            results &= type_matches
        
        if query.file_path:
            file_matches = set(self._by_file.get(query.file_path, []))
            results &= file_matches
        
        if query.parent_name:
            parent_matches = {s for s in results if s.parent and s.parent.name == query.parent_name}
            results &= parent_matches
        
        if query.has_decorator:
            decorator_matches = set(self._by_decorator.get(query.has_decorator, []))
            results &= decorator_matches
        
        if query.is_private is not None:
            if query.is_private:
                results &= self._private_symbols
            else:
                results -= self._private_symbols
        
        if query.is_async is not None:
            if query.is_async:
                results &= self._async_symbols
            else:
                results -= self._async_symbols
        
        # Convert to list and cache
        result_list = list(results)
        self._query_cache[cache_key] = result_list
        
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        return [LookupResult(symbol, lookup_time_ms) for symbol in result_list]
    
    def find_by_qualified_name(self, qualified_name: str) -> List[LookupResult]:
        """Find symbols by qualified name."""
        start_time = time.perf_counter()
        symbol = self._by_qualified_name.get(qualified_name)
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        if symbol:
            return [LookupResult(symbol, lookup_time_ms)]
        return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self._query_cache),
        }
    
    def _generate_cache_key(self, query: SymbolQuery) -> str:
        """Generate a cache key for a query."""
        parts = []
        if query.name:
            parts.append(f"n:{query.name}")
        if query.type:
            parts.append(f"t:{query.type.value}")
        if query.file_path:
            parts.append(f"f:{query.file_path}")
        if query.parent_name:
            parts.append(f"p:{query.parent_name}")
        if query.has_decorator:
            parts.append(f"d:{query.has_decorator}")
        if query.is_private is not None:
            parts.append(f"priv:{query.is_private}")
        if query.is_async is not None:
            parts.append(f"async:{query.is_async}")
        
        return "|".join(parts)
    
    def get_children(self, symbol: Symbol) -> List[Symbol]:
        """Get all child symbols of a given symbol."""
        return symbol.children
    
    def get_methods(self, class_name: str) -> List[Symbol]:
        """Get all methods of a class."""
        class_symbols = self._by_name.get(class_name, [])
        methods = []
        
        for class_symbol in class_symbols:
            if class_symbol.type == SymbolType.CLASS:
                for child in class_symbol.children:
                    if child.type in (SymbolType.METHOD, SymbolType.PROPERTY):
                        methods.append(child)
        
        return methods
    
    def clear(self) -> None:
        """Clear all symbols and indexes."""
        self._symbols.clear()
        self._by_name.clear()
        self._by_type.clear()
        self._by_file.clear()
        self._by_qualified_name.clear()
        self._by_decorator.clear()
        self._private_symbols.clear()
        self._async_symbols.clear()
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the symbol table."""
        return {
            "total_symbols": len(self._symbols),
            "by_type": {t.value: len(symbols) for t, symbols in self._by_type.items()},
            "unique_files": len(self._by_file),
            "private_symbols": len(self._private_symbols),
            "async_symbols": len(self._async_symbols),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) 
                            if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def __len__(self) -> int:
        """Return the number of symbols in the table."""
        return len(self._symbols)
    
    def __contains__(self, name: str) -> bool:
        """Check if a symbol name exists in the table."""
        return name in self._by_name