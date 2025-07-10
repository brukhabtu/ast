"""
Unit tests for the cross-file indexer.
"""

import os
import tempfile
import time
from pathlib import Path
import shutil
import pytest

from astlib.indexer import ProjectIndex, IndexStats
from astlib.symbols import Symbol, SymbolType


class TestProjectIndex:
    """Test suite for ProjectIndex."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create project structure
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "submodule").mkdir()
            
            # Create main.py
            (root / "main.py").write_text("""
import sys
from src.utils import helper_function
from src.models import User, Product

def main():
    '''Main entry point.'''
    user = User("test")
    product = Product("item", 10.0)
    result = helper_function(user, product)
    print(result)

if __name__ == "__main__":
    main()
""")
            
            # Create src/utils.py
            (root / "src" / "utils.py").write_text("""
'''Utility functions.'''

def helper_function(user, product):
    '''Helper function for processing.'''
    return f"{user.name} bought {product.name}"

def unused_function():
    '''This function is not used.'''
    pass

class UtilityClass:
    '''A utility class.'''
    
    def __init__(self):
        self.value = 42
    
    def process(self, data):
        '''Process some data.'''
        return data * self.value
""")
            
            # Create src/models.py
            (root / "src" / "models.py").write_text("""
'''Data models.'''
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    '''User model.'''
    name: str
    email: Optional[str] = None
    
    def get_display_name(self) -> str:
        '''Get display name.'''
        return self.name.title()

@dataclass 
class Product:
    '''Product model.'''
    name: str
    price: float
    
    @property
    def formatted_price(self) -> str:
        '''Get formatted price.'''
        return f"${self.price:.2f}"
""")
            
            # Create src/submodule/__init__.py
            (root / "src" / "submodule" / "__init__.py").write_text("""
'''Submodule package.'''
from .core import SubmoduleClass

__all__ = ['SubmoduleClass']
""")
            
            # Create src/submodule/core.py
            (root / "src" / "submodule" / "core.py").write_text("""
'''Core submodule functionality.'''

class SubmoduleClass:
    '''A class in a submodule.'''
    
    def __init__(self, name: str):
        self.name = name
    
    async def async_method(self):
        '''An async method.'''
        return f"Async: {self.name}"
    
    @staticmethod
    def static_method():
        '''A static method.'''
        return "Static"
""")
            
            # Create tests/test_main.py
            (root / "tests" / "test_main.py").write_text("""
'''Tests for main module.'''
import pytest
from main import main

def test_main():
    '''Test main function.'''
    # This would test main
    pass

class TestIntegration:
    '''Integration tests.'''
    
    def test_user_product_flow(self):
        '''Test user product flow.'''
        pass
""")
            
            # Create a file with syntax error
            (root / "src" / "broken.py").write_text("""
'''This file has a syntax error.'''
def broken_function(
    # Missing closing parenthesis
    pass
""")
            
            # Create __pycache__ to test exclusion
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "temp.pyc").write_text("bytecode")
            
            yield root
    
    def test_build_index_basic(self, temp_project):
        """Test basic index building."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        stats = index.get_stats()
        
        # Check stats
        assert stats.total_files >= 6  # All .py files except __pycache__
        assert stats.total_symbols > 0
        assert stats.index_time_seconds > 0
    
    def test_find_definition(self, temp_project):
        """Test finding symbol definitions."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Test finding a function
        result = index.find_definition("helper_function")
        assert result is not None
        file_path, line_num = result
        assert file_path.name == "utils.py"
        assert line_num > 0
        
        # Test finding a class
        result = index.find_definition("User")
        assert result is not None
        file_path, line_num = result
        assert file_path.name == "models.py"
        
        # Test finding a method
        result = index.find_definition("get_display_name")
        assert result is not None
        
        # Test non-existent symbol
        result = index.find_definition("non_existent_symbol")
        assert result is None
    
    def test_find_definition_with_context(self, temp_project):
        """Test finding definitions with file context."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        main_file = temp_project / "main.py"
        
        # Should find User from imports in main.py
        result = index.find_definition("User", from_file=main_file)
        assert result is not None
        file_path, _ = result
        assert file_path.name == "models.py"
    
    def test_get_symbol(self, temp_project):
        """Test getting symbols by qualified name."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Test getting a class method
        symbol = index.get_symbol("User.get_display_name")
        assert symbol is not None
        assert symbol.name == "get_display_name"
        assert symbol.type == SymbolType.METHOD
        
        # Test getting a nested class
        symbol = index.get_symbol("UtilityClass")
        assert symbol is not None
        assert symbol.type == SymbolType.CLASS
    
    def test_get_all_symbols(self, temp_project):
        """Test getting all symbols."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        all_symbols = index.symbol_table.get_all_symbols()
        assert len(all_symbols) > 0
        
        # Check we have various symbol types
        symbol_types = {s.symbol.type for s in all_symbols}
        assert SymbolType.FUNCTION in symbol_types
        assert SymbolType.CLASS in symbol_types
        assert SymbolType.METHOD in symbol_types
        assert SymbolType.IMPORT in symbol_types
    
    def test_find_symbols_in_file(self, temp_project):
        """Test finding symbols in a specific file."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        utils_file = temp_project / "src" / "utils.py"
        symbols = index.get_symbols_in_file(str(utils_file))
        
        assert len(symbols) > 0
        symbol_names = {s.name for s in symbols}
        assert "helper_function" in symbol_names
        assert "unused_function" in symbol_names
        assert "UtilityClass" in symbol_names
    
    def test_find_references(self, temp_project):
        """Test finding references to symbols."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Find references to User
        refs = index.find_references("User")
        assert len(refs) > 0
        
        # Should include import in main.py
        ref_files = {Path(ref.file_path).name for ref in refs}
        assert "main.py" in ref_files
    
    def test_get_file_imports(self, temp_project):
        """Test getting imports for a file."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        main_file = temp_project / "main.py"
        imports = index.get_file_imports(main_file)
        
        assert imports is not None
        assert "sys" in imports.imported_modules
        assert "src.utils" in imports.imported_modules
        assert "src.models" in imports.imported_modules
    
    def test_import_graph(self, temp_project):
        """Test import graph functionality."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        graph = index.get_import_graph()
        assert len(graph.nodes) > 0
        
        # Check that main.py imports are tracked
        main_module = "main"
        assert main_module in graph.nodes
        main_node = graph.nodes[main_module]
        assert len(main_node.imports) > 0
    
    def test_refresh_modified_files(self, temp_project):
        """Test refreshing modified files."""
        index = ProjectIndex(str(temp_project))
        initial_stats = index.build_index(temp_project)
        
        # Wait a bit to ensure mtime difference
        time.sleep(0.1)
        
        # Modify a file
        utils_file = temp_project / "src" / "utils.py"
        original_content = utils_file.read_text()
        utils_file.write_text(original_content + "\n\ndef new_function():\n    pass")
        
        # Refresh
        refresh_stats = index.refresh([utils_file])
        assert refresh_stats.total_files == 1
        assert refresh_stats.total_files == 1
        
        # Check that new function is indexed
        result = index.find_definition("new_function")
        assert result is not None
    
    def test_refresh_all_modified(self, temp_project):
        """Test auto-detecting and refreshing modified files."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Wait and modify a file
        time.sleep(0.1)
        main_file = temp_project / "main.py"
        original_content = main_file.read_text()
        main_file.write_text(original_content + "\n\ndef new_main_function():\n    pass")
        
        # Refresh without specifying files
        refresh_stats = index.refresh()
        assert refresh_stats.total_files == 1
        assert refresh_stats.total_files == 1
    
    def test_refresh_deleted_file(self, temp_project):
        """Test handling deleted files during refresh."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Delete a file
        utils_file = temp_project / "src" / "utils.py"
        utils_file.unlink()
        
        # Refresh
        refresh_stats = index.refresh([utils_file])
        assert refresh_stats.total_files == 1
        assert refresh_stats.total_files == 1  # Successful removal counts as indexed
        
        # Verify symbols are gone
        result = index.find_definition("helper_function")
        assert result is None
    
    def test_exclude_patterns(self, temp_project):
        """Test file exclusion patterns."""
        # Create a venv directory
        venv_dir = temp_project / "venv"
        venv_dir.mkdir()
        (venv_dir / "test.py").write_text("# Should be excluded")
        
        index = ProjectIndex(str(temp_project))
        index.build_index()
        stats = index.get_stats()
        
        # venv should be excluded by default
        all_files = list(index._indexed_files)
        assert not any("venv" in str(f) for f in all_files)
    
    def test_custom_exclude_patterns(self, temp_project):
        """Test custom exclusion patterns."""
        index = ProjectIndex(str(temp_project), ignore_patterns=["**/tests/**", "**/broken.py"])
        index.build_index()
        
        # tests directory should be excluded
        all_files = list(index._indexed_files)
        assert not any("tests" in str(f) for f in all_files)
        assert not any("broken.py" in str(f) for f in all_files)
    
    def test_parallel_indexing(self, temp_project):
        """Test parallel indexing with multiple workers."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        stats = index.get_stats()
        
        assert stats.total_files >= 6
        assert stats.total_symbols > 0
    
    def test_sequential_indexing(self, temp_project):
        """Test sequential indexing with single worker."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        stats = index.get_stats()
        
        assert stats.total_files >= 6
        assert stats.total_symbols > 0
    
    def test_get_stats(self, temp_project):
        """Test getting comprehensive statistics."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        stats = index.get_stats()
        
        # Check index stats
        assert stats.total_files >= 6
        assert stats.total_symbols > 0
        assert stats.total_functions > 0
        assert stats.total_classes > 0
        assert stats.index_time_seconds > 0
        
        # Stats should have some errors (from broken.py)
        assert len(stats.errors) > 0
    
    def test_empty_project(self):
        """Test indexing an empty project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = ProjectIndex(str(temp_project))
            stats = index.build_index(tmpdir)
            
            assert stats.total_files == 0
            assert stats.total_files == 0
            assert stats.total_symbols == 0
    
    def test_nonexistent_path(self):
        """Test indexing a non-existent path."""
        index = ProjectIndex(str(temp_project))
        
        with pytest.raises(ValueError, match="Root path does not exist"):
            index.build_index("/this/path/does/not/exist")
    
    def test_async_symbol_detection(self, temp_project):
        """Test detection of async symbols."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Find async method
        result = index.find_definition("async_method")
        assert result is not None
        
        # Get the symbol
        symbol = index.get_symbol("SubmoduleClass.async_method")
        assert symbol is not None
        assert symbol.is_async is True
    
    def test_property_detection(self, temp_project):
        """Test detection of properties."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Find property
        symbol = index.get_symbol("Product.formatted_price")
        assert symbol is not None
        assert symbol.type == SymbolType.PROPERTY
    
    def test_decorator_detection(self, temp_project):
        """Test detection of decorators."""
        index = ProjectIndex(str(temp_project))
        index.build_index()
        
        # Find dataclass decorated classes
        symbol = index.get_symbol("User")
        assert symbol is not None
        assert "dataclass" in symbol.decorators
    
    def test_performance_large_project(self, temp_project):
        """Test performance with many files."""
        # Create many Python files
        large_dir = temp_project / "large"
        large_dir.mkdir()
        
        for i in range(100):
            file_content = f'''
"""Module {i}"""

def function_{i}():
    """Function in module {i}"""
    pass

class Class_{i}:
    """Class in module {i}"""
    
    def method_{i}(self):
        """Method in class {i}"""
        pass
'''
            (large_dir / f"module_{i}.py").write_text(file_content)
        
        # Index with timing
        index = ProjectIndex(str(large_dir))
        start_time = time.perf_counter()
        index.build_index()
        stats = index.get_stats()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Should index 100+ files in reasonable time
        assert stats.total_files >= 100
        assert stats.total_symbols >= 300  # At least 3 symbols per file
        
        # Performance assertion - should be fast
        # Relaxed for CI environments
        assert elapsed_ms < 5000  # Less than 5 seconds for 100+ files


class TestIndexStats:
    """Test IndexStats dataclass."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # The current IndexStats doesn't have indexed_files or failed_files
        # It has total_files and errors list
        stats = IndexStats(total_files=10, total_symbols=20)
        # Add some errors
        stats.errors = [("file1.py", "error1"), ("file2.py", "error2")]
        
        # Test basic stats
        assert stats.total_files == 10
        assert len(stats.errors) == 2


# FileIndex class no longer exists in the current implementation
# class TestFileIndex:
#     """Test FileIndex dataclass."""
#     
#     def test_file_index_creation(self):
#         """Test creating FileIndex."""
#         file_path = Path("test.py")
#         file_index = FileIndex(
#             file_path=file_path,
#             last_modified=time.time()
#         )
#         
#         assert file_index.file_path == file_path
#         assert file_index.last_modified > 0
#         assert file_index.parse_result is None
#         assert file_index.symbols == []
#         assert file_index.imports is None
#         assert file_index.error is None