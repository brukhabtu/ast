"""
Unit tests for symbol table functionality.
"""

import pytest
import time
from typing import List

from astlib.symbols import Symbol, SymbolType, Position
from astlib.symbol_table import SymbolTable, SymbolQuery, LookupResult


class TestSymbolTable:
    """Test symbol table basic functionality."""
    
    @pytest.fixture
    def empty_table(self):
        """Create an empty symbol table."""
        return SymbolTable()
    
    @pytest.fixture
    def sample_symbols(self):
        """Create sample symbols for testing."""
        return [
            Symbol(
                name="function1",
                type=SymbolType.FUNCTION,
                position=Position(line=10, column=0),
                file_path="module1.py",
                signature="(x: int) -> int"
            ),
            Symbol(
                name="MyClass",
                type=SymbolType.CLASS,
                position=Position(line=20, column=0),
                file_path="module1.py",
                bases=["BaseClass"]
            ),
            Symbol(
                name="async_func",
                type=SymbolType.ASYNC_FUNCTION,
                position=Position(line=30, column=0),
                file_path="module2.py",
                is_async=True
            ),
            Symbol(
                name="_private_func",
                type=SymbolType.FUNCTION,
                position=Position(line=40, column=0),
                file_path="module2.py",
                is_private=True
            ),
            Symbol(
                name="decorated_func",
                type=SymbolType.FUNCTION,
                position=Position(line=50, column=0),
                file_path="module3.py",
                decorators=["cache", "timer"]
            ),
        ]
    
    @pytest.fixture
    def populated_table(self, sample_symbols):
        """Create a symbol table with sample data."""
        table = SymbolTable()
        table.add_symbols(sample_symbols)
        return table
    
    def test_add_single_symbol(self, empty_table):
        """Test adding a single symbol."""
        symbol = Symbol(
            name="test_func",
            type=SymbolType.FUNCTION,
            position=Position(line=1, column=0)
        )
        
        empty_table.add_symbol(symbol)
        
        assert len(empty_table) == 1
        assert "test_func" in empty_table
    
    def test_add_multiple_symbols(self, empty_table, sample_symbols):
        """Test adding multiple symbols."""
        empty_table.add_symbols(sample_symbols)
        
        assert len(empty_table) == len(sample_symbols)
        for symbol in sample_symbols:
            assert symbol.name in empty_table
    
    def test_find_by_name_exact(self, populated_table):
        """Test exact name lookup."""
        results = populated_table.find_by_name("function1", exact=True)
        
        assert len(results) == 1
        assert results[0].symbol.name == "function1"
        assert results[0].lookup_time_ms < 10  # Should be fast
    
    def test_find_by_name_prefix(self, populated_table):
        """Test prefix name lookup."""
        # Add more functions with similar prefixes
        populated_table.add_symbol(
            Symbol(name="function2", type=SymbolType.FUNCTION, 
                  position=Position(line=60, column=0))
        )
        populated_table.add_symbol(
            Symbol(name="func_other", type=SymbolType.FUNCTION,
                  position=Position(line=70, column=0))
        )
        
        results = populated_table.find_by_name("func", exact=False)
        
        # Should find function1, function2, func_other
        assert len(results) >= 3
        names = {r.symbol.name for r in results}
        assert "function1" in names
        assert "function2" in names
        assert "func_other" in names
    
    def test_find_by_type(self, populated_table):
        """Test finding symbols by type."""
        functions = populated_table.find_by_type(SymbolType.FUNCTION)
        
        # Should find regular functions only
        func_names = {r.symbol.name for r in functions}
        assert "function1" in func_names
        assert "_private_func" in func_names
        assert "decorated_func" in func_names
        assert "async_func" not in func_names  # async has different type
        assert "MyClass" not in func_names
    
    def test_find_by_file(self, populated_table):
        """Test finding symbols by file."""
        results = populated_table.find_by_file("module1.py")
        
        assert len(results) == 2
        names = {r.symbol.name for r in results}
        assert "function1" in names
        assert "MyClass" in names
    
    def test_find_by_qualified_name(self, populated_table):
        """Test finding by qualified name."""
        # Add nested symbol
        class_sym = next(s for s in populated_table._symbols if s.name == "MyClass")
        method_sym = Symbol(
            name="method1",
            type=SymbolType.METHOD,
            position=Position(line=22, column=4),
            parent=class_sym
        )
        class_sym.children.append(method_sym)
        populated_table.add_symbol(method_sym)
        
        results = populated_table.find_by_qualified_name("MyClass.method1")
        
        assert results is not None
        assert len(results) > 0
        assert results[0].symbol.name == "method1"
        assert results[0].symbol.parent.name == "MyClass"
    
    def test_find_by_decorator(self, populated_table):
        """Test finding symbols by decorator."""
        cache_results = populated_table.find_by_decorator("cache")
        timer_results = populated_table.find_by_decorator("timer")
        
        assert len(cache_results) == 1
        assert cache_results[0].symbol.name == "decorated_func"
        
        assert len(timer_results) == 1
        assert timer_results[0].symbol.name == "decorated_func"
    
    def test_find_private_symbols(self, populated_table):
        """Test finding private symbols."""
        results = populated_table.find_private_symbols()
        
        assert len(results) == 1
        assert results[0].symbol.name == "_private_func"
    
    def test_find_async_symbols(self, populated_table):
        """Test finding async symbols."""
        results = populated_table.find_async_symbols()
        
        assert len(results) == 1
        assert results[0].symbol.name == "async_func"
    
    def test_complex_query(self, populated_table):
        """Test complex queries with multiple filters."""
        # Query for functions in module2.py
        query = SymbolQuery(
            type=SymbolType.FUNCTION,
            file_path="module2.py"
        )
        results = populated_table.query(query)
        
        assert len(results) == 1
        assert results[0].symbol.name == "_private_func"
        
        # Query for non-private functions
        query2 = SymbolQuery(
            type=SymbolType.FUNCTION,
            is_private=False
        )
        results2 = populated_table.query(query2)
        
        names = {r.symbol.name for r in results2}
        assert "function1" in names
        assert "decorated_func" in names
        assert "_private_func" not in names
    
    def test_get_children(self, populated_table):
        """Test getting child symbols."""
        class_sym = next(s for s in populated_table._symbols if s.name == "MyClass")
        
        # Add some methods
        for i in range(3):
            method = Symbol(
                name=f"method{i}",
                type=SymbolType.METHOD,
                position=Position(line=25 + i, column=4),
                parent=class_sym
            )
            class_sym.children.append(method)
        
        children = populated_table.get_children(class_sym)
        
        assert len(children) == 3
        assert all(c.type == SymbolType.METHOD for c in children)
        assert all(c.parent == class_sym for c in children)
    
    def test_get_methods(self, populated_table):
        """Test getting methods of a class."""
        # Add a class with methods
        class_sym = next(s for s in populated_table._symbols if s.name == "MyClass")
        
        methods = [
            Symbol(name="__init__", type=SymbolType.METHOD,
                  position=Position(line=21, column=4), parent=class_sym),
            Symbol(name="method1", type=SymbolType.METHOD,
                  position=Position(line=25, column=4), parent=class_sym),
            Symbol(name="property1", type=SymbolType.PROPERTY,
                  position=Position(line=30, column=4), parent=class_sym),
        ]
        
        for method in methods:
            class_sym.children.append(method)
            populated_table.add_symbol(method)
        
        class_methods = populated_table.get_methods("MyClass")
        
        assert len(class_methods) == 3
        method_names = {m.name for m in class_methods}
        assert "__init__" in method_names
        assert "method1" in method_names
        assert "property1" in method_names
    
    def test_clear_table(self, populated_table):
        """Test clearing the symbol table."""
        assert len(populated_table) > 0
        
        populated_table.clear()
        
        assert len(populated_table) == 0
        assert len(populated_table._by_name) == 0
        assert len(populated_table._by_type) == 0
        assert populated_table._cache_hits == 0
        assert populated_table._cache_misses == 0
    
    def test_stats(self, populated_table):
        """Test getting statistics."""
        stats = populated_table.get_stats()
        
        assert stats["total_symbols"] == len(populated_table)
        assert stats["by_type"][SymbolType.FUNCTION.value] == 3
        assert stats["by_type"][SymbolType.CLASS.value] == 1
        assert stats["private_symbols"] == 1
        assert stats["async_symbols"] == 1
        assert "cache_hit_rate" in stats


class TestSymbolTablePerformance:
    """Test performance characteristics of symbol table."""
    
    def create_large_symbol_set(self, count: int) -> List[Symbol]:
        """Create a large set of symbols for performance testing."""
        symbols = []
        
        for i in range(count):
            symbol_type = SymbolType.FUNCTION if i % 3 == 0 else SymbolType.CLASS
            
            symbol = Symbol(
                name=f"symbol_{i}",
                type=symbol_type,
                position=Position(line=i * 10, column=0),
                file_path=f"module_{i % 10}.py",
                decorators=["decorator"] if i % 5 == 0 else [],
                is_private=i % 7 == 0,
                is_async=i % 11 == 0 and symbol_type == SymbolType.FUNCTION
            )
            symbols.append(symbol)
        
        return symbols
    
    def test_lookup_performance(self):
        """Test that lookups are fast even with many symbols."""
        table = SymbolTable()
        symbols = self.create_large_symbol_set(10000)
        table.add_symbols(symbols)
        
        # Test name lookup
        results = table.find_by_name("symbol_5000", exact=True)
        assert len(results) == 1
        assert results[0].lookup_time_ms < 10  # Should be under 10ms
        
        # Test type lookup
        results = table.find_by_type(SymbolType.FUNCTION)
        assert len(results) > 0
        assert results[0].lookup_time_ms < 10
        
        # Test file lookup
        results = table.find_by_file("module_5.py")
        assert len(results) > 0
        assert results[0].lookup_time_ms < 10
    
    def test_query_caching(self):
        """Test that query caching improves performance."""
        table = SymbolTable()
        symbols = self.create_large_symbol_set(1000)
        table.add_symbols(symbols)
        
        query = SymbolQuery(
            type=SymbolType.FUNCTION,
            is_private=False,
            file_path="module_1.py"
        )
        
        # First query - cache miss
        results1 = table.query(query)
        first_time = results1[0].lookup_time_ms if results1 else 0
        
        # Second query - should hit cache
        results2 = table.query(query)
        second_time = results2[0].lookup_time_ms if results2 else 0
        
        # Cache hit should be faster
        assert second_time <= first_time
        assert table._cache_hits > 0
        
        # Verify results are the same
        assert len(results1) == len(results2)
    
    def test_prefix_search_performance(self):
        """Test prefix search performance."""
        table = SymbolTable()
        
        # Add symbols with common prefixes
        for i in range(100):
            table.add_symbol(Symbol(
                name=f"test_function_{i}",
                type=SymbolType.FUNCTION,
                position=Position(line=i, column=0)
            ))
            table.add_symbol(Symbol(
                name=f"test_class_{i}",
                type=SymbolType.CLASS,
                position=Position(line=i * 2, column=0)
            ))
        
        # Prefix search should still be fast
        results = table.find_by_name("test_", exact=False)
        assert len(results) == 200
        assert all(r.lookup_time_ms < 10 for r in results)


class TestSymbolTableEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_queries(self):
        """Test queries on empty table."""
        table = SymbolTable()
        
        assert len(table.find_by_name("anything")) == 0
        assert len(table.find_by_type(SymbolType.FUNCTION)) == 0
        assert len(table.find_by_file("any.py")) == 0
        assert table.find_by_qualified_name("any.name") is None
    
    def test_duplicate_names(self):
        """Test handling of duplicate symbol names."""
        table = SymbolTable()
        
        # Add symbols with same name but different files
        for i in range(3):
            table.add_symbol(Symbol(
                name="duplicate_func",
                type=SymbolType.FUNCTION,
                position=Position(line=10, column=0),
                file_path=f"module{i}.py"
            ))
        
        results = table.find_by_name("duplicate_func")
        assert len(results) == 3
        
        # All should have same name but different files
        files = {r.symbol.file_path for r in results}
        assert len(files) == 3
    
    def test_deeply_nested_symbols(self):
        """Test handling of deeply nested symbols."""
        table = SymbolTable()
        
        # Create deeply nested structure
        root = Symbol(
            name="RootClass",
            type=SymbolType.CLASS,
            position=Position(line=1, column=0)
        )
        table.add_symbol(root)
        
        current = root
        for i in range(10):
            nested = Symbol(
                name=f"Nested{i}",
                type=SymbolType.CLASS,
                position=Position(line=i * 10, column=i * 4),
                parent=current
            )
            current.children.append(nested)
            table.add_symbol(nested)
            current = nested
        
        # Should be able to find deeply nested symbol
        qualified_name = ".".join([f"Nested{i}" for i in range(10)])
        qualified_name = f"RootClass.{qualified_name}"
        
        result = table.find_by_qualified_name(qualified_name)
        assert result is not None
        assert result.symbol.name == "Nested9"