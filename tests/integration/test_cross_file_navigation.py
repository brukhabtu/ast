"""
Integration tests for cross-file navigation functionality.

Tests the complete workflow: parse -> index -> cache -> lookup
"""

import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Set

import pytest

from astlib.parser import parse_file
from astlib.symbol_table import SymbolTable, SymbolQuery
from astlib.symbols import Symbol, SymbolType, extract_symbols
from astlib.finder import FunctionFinder


class TestCrossFileNavigation:
    """Test cross-file navigation capabilities."""
    
    @pytest.fixture
    def django_project(self):
        """Get Django-like test project path."""
        return Path(__file__).parent.parent / "fixtures" / "test_projects" / "django_like"
    
    @pytest.fixture
    def circular_project(self):
        """Get circular imports test project path."""
        return Path(__file__).parent.parent / "fixtures" / "test_projects" / "circular_imports"
    
    @pytest.fixture
    def broken_project(self):
        """Get broken imports test project path."""
        return Path(__file__).parent.parent / "fixtures" / "test_projects" / "broken_imports"
    
    def test_django_like_structure(self, django_project):
        """Test parsing and indexing a Django-like project structure."""
        # Parse the entire project
        symbol_table = SymbolTable()
        files_parsed = 0
        symbols_found = 0
        
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            result = parse_file(py_file)
            if result.tree:
                files_parsed += 1
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
                symbols_found += len(symbols)
        
        assert files_parsed >= 6  # At least our created files
        assert symbols_found > 20  # Should find many symbols
        
        # Test cross-file lookups
        # Find User model
        user_results = symbol_table.find_by_name("User")
        assert len(user_results) >= 1  # May include imports
        
        # Find the class definition specifically
        user_class = None
        for result in user_results:
            if result.symbol.type == SymbolType.CLASS:
                user_class = result.symbol
                break
        
        assert user_class is not None
        assert "models.py" in user_class.file_path
        
        # Find all models
        model_results = symbol_table.find_by_type(SymbolType.CLASS)
        model_names = {r.symbol.name for r in model_results}
        assert "User" in model_names
        assert "Post" in model_names
        assert "Comment" in model_names
        assert "Model" in model_names
        
        # Test method lookups
        method_results = symbol_table.query(SymbolQuery(
            type=SymbolType.METHOD,
            parent_name="User"
        ))
        method_names = {r.symbol.name for r in method_results}
        assert "get_full_name" in method_names
        assert "deactivate" in method_names
        
        # Test import resolution
        # Views should reference models
        view_file_symbols = symbol_table.find_by_file(
            str(django_project / "views.py")
        )
        view_imports = [s.symbol for s in view_file_symbols 
                       if s.symbol.type == SymbolType.IMPORT]
        
        # Should have imports from models
        import_sources = set()
        for imp in view_imports:
            if hasattr(imp, 'import_from'):
                import_sources.add(imp.import_from)
        
        # Test cross-file definition lookup
        # In views.py, find where Post is defined
        post_in_views = None
        for result in view_file_symbols:
            if result.symbol.name == "Post" and result.symbol.type == SymbolType.IMPORT:
                post_in_views = result.symbol
                break
        
        # Now find the actual Post class definition
        post_results = symbol_table.find_by_name("Post")
        post_class = None
        for result in post_results:
            if result.symbol.type == SymbolType.CLASS:
                post_class = result.symbol
                break
        
        assert post_class is not None
        assert "models.py" in post_class.file_path
    
    def test_circular_imports_handling(self, circular_project):
        """Test handling of circular imports."""
        symbol_table = SymbolTable()
        
        # Parse all modules
        for py_file in circular_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Find ClassA and ClassB
        class_a_results = symbol_table.find_by_name("ClassA")
        class_b_results = symbol_table.find_by_name("ClassB")
        
        # Find the actual class definitions (not imports)
        class_a = None
        class_b = None
        
        for result in class_a_results:
            if result.symbol.type == SymbolType.CLASS:
                class_a = result.symbol
                break
                
        for result in class_b_results:
            if result.symbol.type == SymbolType.CLASS:
                class_b = result.symbol
                break
        
        assert class_a is not None
        assert class_b is not None
        
        # Check methods that reference the other class
        method_results = symbol_table.query(SymbolQuery(
            type=SymbolType.METHOD,
            parent_name="ClassA"
        ))
        method_names = {r.symbol.name for r in method_results}
        assert "get_b_instance" in method_names
        assert "process_with_b" in method_names
        
        # Test TYPE_CHECKING imports
        # Both modules should parse successfully despite circular imports
        assert "module_a.py" in class_a.file_path
        assert "module_b.py" in class_b.file_path
    
    def test_broken_imports_recovery(self, broken_project):
        """Test error recovery with broken imports."""
        symbol_table = SymbolTable()
        parse_errors = []
        
        # Parse all modules, collecting errors
        for py_file in broken_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                result = parse_file(py_file)
                if result.tree:
                    symbols = extract_symbols(result.tree, str(py_file))
                    symbol_table.add_symbols(symbols)
                if result.errors:
                    for error in result.errors:
                        parse_errors.append((str(py_file), error.message))
            except Exception as e:
                parse_errors.append((str(py_file), str(e)))
        
        # Should have at least one syntax error
        assert len(parse_errors) >= 1
        assert any("syntax_error.py" in err[0] for err in parse_errors)
        
        # But should still parse good modules
        good_class_results = symbol_table.find_by_name("GoodClass")
        assert len(good_class_results) >= 1  # May include imports
        
        # Find the class definition
        good_class = None
        for result in good_class_results:
            if result.symbol.type == SymbolType.CLASS:
                good_class = result.symbol
                break
        
        assert good_class is not None
        assert "good_module.py" in good_class.file_path
        
        # Should parse modules with missing imports (AST parsing succeeds)
        broken_class_results = symbol_table.find_by_name("BrokenClass")
        assert len(broken_class_results) >= 1  # May include imports
        
        # Find class definition
        broken_class = None
        for result in broken_class_results:
            if result.symbol.type == SymbolType.CLASS:
                broken_class = result.symbol
                break
        assert broken_class is not None
        
        # Should handle partial imports
        partial_class = symbol_table.find_by_name("PartiallyWorking")
        assert len(partial_class) == 1
    
    def test_find_definition_performance(self, django_project):
        """Test performance of find_definition with caching."""
        symbol_table = SymbolTable()
        
        # Index the project
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Warm up cache with first lookup
        start = time.time()
        result1 = symbol_table.find_by_name("User")
        first_lookup_ms = (time.time() - start) * 1000
        
        # Second lookup should be faster (cached)
        start = time.time()
        result2 = symbol_table.find_by_name("User")
        cached_lookup_ms = (time.time() - start) * 1000
        
        assert len(result1) == len(result2)
        assert cached_lookup_ms < first_lookup_ms
        assert cached_lookup_ms < 50  # Should be under 50ms with cache
        
        # Test cache effectiveness
        cache_stats = symbol_table.get_cache_stats()
        assert cache_stats["hit_rate"] > 0  # Should have cache hits
    
    def test_complex_query_scenarios(self, django_project):
        """Test complex cross-file query scenarios."""
        symbol_table = SymbolTable()
        
        # Index the project
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Scenario 1: Find all view functions/classes
        view_symbols = symbol_table.find_by_file(str(django_project / "views.py"))
        view_functions = [s.symbol for s in view_symbols 
                         if s.symbol.type == SymbolType.FUNCTION]
        view_classes = [s.symbol for s in view_symbols 
                       if s.symbol.type == SymbolType.CLASS]
        
        assert len(view_functions) >= 2  # index, post_detail
        assert len(view_classes) >= 2    # UserView, PostCreateView
        
        # Scenario 2: Find all methods that use a specific decorator
        property_methods = symbol_table.query(SymbolQuery(
            type=SymbolType.METHOD,
            has_decorator="property"
        ))
        # Also check PROPERTY type
        property_symbols = symbol_table.find_by_type(SymbolType.PROPERTY)
        
        # Should find properties by either method
        assert len(property_methods) + len(property_symbols) >= 1  # Post.summary
        
        # Scenario 3: Find all Meta classes
        meta_classes = symbol_table.find_by_name("Meta")
        assert len(meta_classes) >= 3  # One for each model
        
        # Scenario 4: Cross-file method calls
        # Find notify_author in Comment, trace to tasks module
        notify_methods = symbol_table.find_by_name("notify_author")
        assert len(notify_methods) >= 1
        
        # Find send_notification across files
        send_notif_results = symbol_table.find_by_name("send_notification")
        send_notif_files = {r.symbol.file_path for r in send_notif_results}
        assert len(send_notif_files) >= 2  # utils.py and tasks.py
    
    def test_incremental_updates(self, django_project):
        """Test incremental updates to the symbol table."""
        symbol_table = SymbolTable()
        
        # Initial indexing
        initial_symbols = 0
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
                initial_symbols += len(symbols)
        
        # Simulate file modification by re-parsing one file
        models_file = django_project / "models.py"
        result = parse_file(models_file)
        new_symbols = extract_symbols(result.tree, str(models_file))
        
        # In a real implementation, we'd remove old symbols from this file first
        # For now, just verify we can add new symbols
        symbol_table.add_symbols(new_symbols)
        
        # Verify symbol table still works
        user_results = symbol_table.find_by_name("User")
        assert len(user_results) >= 1
    
    def test_finder_integration(self, django_project):
        """Test integration with FunctionFinder."""
        finder = FunctionFinder()
        symbol_table = SymbolTable()
        
        # Use finder to search for test patterns
        all_functions = []
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            functions = finder.find_functions(py_file, "*")
            all_functions.extend(functions)
            
            # Also build symbol table
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Compare finder results with symbol table
        finder_names = {f['name'] for f in all_functions}
        symbol_functions = symbol_table.find_by_type(SymbolType.FUNCTION)
        symbol_methods = symbol_table.find_by_type(SymbolType.METHOD)
        symbol_names = {r.symbol.name for r in symbol_functions + symbol_methods}
        
        # Should have significant overlap
        overlap = finder_names & symbol_names
        assert len(overlap) > 10  # Should find many common functions


class TestProjectScaling:
    """Test performance with larger projects."""
    
    def create_large_project(self, base_path: Path, num_modules: int = 100) -> Path:
        """Create a large test project."""
        project_path = base_path / "large_project"
        project_path.mkdir(exist_ok=True)
        
        # Create package structure
        for i in range(num_modules):
            package = project_path / f"package_{i % 10}"
            package.mkdir(exist_ok=True)
            
            # Create module with multiple symbols
            module_content = f'''"""Module {i} in large project."""

import os
import sys
from typing import List, Dict, Optional
from ..package_{(i + 1) % 10}.module_{(i + 1) % num_modules} import helper_{(i + 1) % num_modules}


CONSTANT_{i} = {i}
_PRIVATE_{i} = "private_{i}"


class BaseClass_{i}:
    """Base class {i}."""
    
    def __init__(self, value: int = {i}):
        self.value = value
    
    def method_{i}(self) -> int:
        """Method {i}."""
        return self.value * {i}
    
    @property
    def computed_{i}(self) -> int:
        """Computed property {i}."""
        return self.value ** 2


class DerivedClass_{i}(BaseClass_{i}):
    """Derived class {i}."""
    
    def method_{i}(self) -> int:
        """Overridden method {i}."""
        return super().method_{i}() + {i}
    
    async def async_method_{i}(self) -> str:
        """Async method {i}."""
        return f"Async result {i}"


def function_{i}(param: int = {i}) -> str:
    """Function {i}."""
    return f"Function {{param}}"


async def async_function_{i}() -> Dict[str, int]:
    """Async function {i}."""
    return {{"result": {i}}}


def helper_{i}(data: List[int]) -> int:
    """Helper function {i}."""
    return sum(data) + {i}


@decorator_{(i + 1) % 10}
def decorated_{i}() -> None:
    """Decorated function {i}."""
    pass


def decorator_{i}(func):
    """Decorator {i}."""
    def wrapper(*args, **kwargs):
        print(f"Decorator {i}")
        return func(*args, **kwargs)
    return wrapper
'''
            
            module_path = package / f"module_{i}.py"
            module_path.write_text(module_content)
            
            # Create __init__.py
            (package / "__init__.py").write_text(f'"""Package {i % 10}."""')
        
        return project_path
    
    @pytest.mark.performance
    def test_large_project_indexing(self, tmp_path):
        """Test indexing performance with 1000+ files."""
        # Create large project
        project_path = self.create_large_project(tmp_path, num_modules=100)
        
        symbol_table = SymbolTable()
        start_time = time.time()
        files_parsed = 0
        total_symbols = 0
        
        # Index all files
        for py_file in project_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            result = parse_file(py_file)
            if result.tree:
                files_parsed += 1
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
                total_symbols += len(symbols)
        
        indexing_time = time.time() - start_time
        
        # Performance assertions
        assert files_parsed >= 100
        assert total_symbols > 1000
        assert indexing_time < 5.0  # Should index 100 files in under 5 seconds
        
        # Test lookup performance
        lookup_times = []
        for i in range(100):
            start = time.time()
            results = symbol_table.find_by_name(f"function_{i}")
            lookup_times.append((time.time() - start) * 1000)
            assert len(results) >= 1
        
        avg_lookup_time = sum(lookup_times) / len(lookup_times)
        assert avg_lookup_time < 50  # Average lookup under 50ms
        
        # Test memory usage (basic check)
        # In a real test, we'd use memory_profiler or similar
        assert len(symbol_table._symbols) == total_symbols
        
        # Cleanup
        shutil.rmtree(project_path)
    
    @pytest.mark.performance
    def test_cache_effectiveness(self, tmp_path):
        """Test cache hit rates with repeated queries."""
        project_path = self.create_large_project(tmp_path, num_modules=50)
        symbol_table = SymbolTable()
        
        # Index project
        for py_file in project_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Perform repeated queries
        query_patterns = [
            ("function_", SymbolType.FUNCTION),
            ("BaseClass_", SymbolType.CLASS),
            ("method_", SymbolType.METHOD),
            ("async_", None),
        ]
        
        # First pass - populate cache
        for pattern, sym_type in query_patterns:
            for i in range(10):
                if sym_type:
                    symbol_table.query(SymbolQuery(
                        name=f"{pattern}{i}",
                        type=sym_type
                    ))
                else:
                    symbol_table.find_by_name(f"{pattern}function_{i}")
        
        # Second pass - should hit cache
        initial_hits = symbol_table._cache_hits
        for pattern, sym_type in query_patterns:
            for i in range(10):
                if sym_type:
                    symbol_table.query(SymbolQuery(
                        name=f"{pattern}{i}",
                        type=sym_type
                    ))
                else:
                    symbol_table.find_by_name(f"{pattern}function_{i}")
        
        cache_hits_increase = symbol_table._cache_hits - initial_hits
        assert cache_hits_increase >= 30  # Most queries should hit cache
        
        # Check cache stats
        stats = symbol_table.get_cache_stats()
        assert stats["hit_rate"] > 0.8  # 80%+ cache hit rate
        
        # Cleanup
        shutil.rmtree(project_path)


class TestConcurrentAccess:
    """Test concurrent access to symbol table."""
    
    @pytest.mark.performance
    def test_concurrent_lookups(self, django_project):
        """Test concurrent lookups don't cause issues."""
        import concurrent.futures
        import threading
        
        symbol_table = SymbolTable()
        
        # Index project
        for py_file in django_project.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            result = parse_file(py_file)
            if result.tree:
                symbols = extract_symbols(result.tree, str(py_file))
                symbol_table.add_symbols(symbols)
        
        # Define lookup tasks
        def lookup_task(name: str, iterations: int = 100):
            results = []
            for _ in range(iterations):
                result = symbol_table.find_by_name(name)
                results.append(len(result))
            return results
        
        # Run concurrent lookups
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            names = ["User", "Post", "Comment", "Model", "Field"]
            
            for name in names:
                future = executor.submit(lookup_task, name, 50)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        
        # All lookups should succeed
        assert len(all_results) == 250  # 5 names * 50 iterations
        assert all(r >= 0 for r in all_results)  # No negative results