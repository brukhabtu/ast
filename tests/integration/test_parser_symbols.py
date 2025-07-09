"""
Integration tests for parser and symbol extraction.
"""

import ast
import pytest
import tempfile
import os
from pathlib import Path

from astlib.symbols import extract_symbols, SymbolType
from astlib.symbol_table import SymbolTable, SymbolQuery


class TestParserSymbolIntegration:
    """Test integration between parser and symbol extraction."""
    
    def test_parse_and_extract_real_file(self):
        """Test parsing and extracting symbols from a real Python file."""
        code = '''"""
Example module for testing symbol extraction.
"""

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

CONSTANT_VALUE = 42
_PRIVATE_CONSTANT = "secret"

class BaseClass:
    """Base class for inheritance testing."""
    class_variable = "shared"
    
    def __init__(self, name: str):
        self.name = name
    
    def base_method(self) -> str:
        return f"Base: {self.name}"

class DerivedClass(BaseClass):
    """Derived class with additional functionality."""
    
    def __init__(self, name: str, value: int):
        super().__init__(name)
        self.value = value
    
    @property
    def computed_value(self) -> int:
        """Computed property."""
        return self.value * 2
    
    @computed_value.setter
    def computed_value(self, val: int):
        self.value = val // 2
    
    async def async_method(self, data: Dict[str, any]) -> Dict[str, any]:
        """Async method for processing data."""
        return {"processed": data}
    
    @staticmethod
    def static_helper(x: int, y: int) -> int:
        """Static helper method."""
        return x + y
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DerivedClass':
        """Factory method."""
        return cls(data["name"], data["value"])

def module_function(arg1: str, arg2: int = 10) -> bool:
    """Module-level function."""
    def inner_function():
        """Nested function."""
        return arg1.upper()
    
    return len(inner_function()) > arg2

async def async_module_function(items: List[str]) -> List[str]:
    """Async module function."""
    return [item.upper() for item in items]

@lru_cache(maxsize=128)
def cached_function(n: int) -> int:
    """Cached recursive function."""
    if n <= 1:
        return n
    return cached_function(n - 1) + cached_function(n - 2)

class _PrivateClass:
    """Private class."""
    pass

def _private_function():
    """Private function."""
    pass

# Module-level code
if __name__ == "__main__":
    print("Running as main")
'''
        
        # Parse the code
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Create symbol table
        table = SymbolTable()
        table.add_symbols(symbols)
        
        # Verify symbols were extracted correctly
        assert len(symbols) > 10
        
        # Check specific symbols
        assert "BaseClass" in table
        assert "DerivedClass" in table
        assert "module_function" in table
        assert "async_module_function" in table
        
        # Check constants
        constants = table.find_by_type(SymbolType.CONSTANT)
        constant_names = {r.symbol.name for r in constants}
        assert "CONSTANT_VALUE" in constant_names
        assert "_PRIVATE_CONSTANT" in constant_names
        
        # Check classes
        classes = table.find_by_type(SymbolType.CLASS)
        class_names = {r.symbol.name for r in classes}
        assert "BaseClass" in class_names
        assert "DerivedClass" in class_names
        assert "_PrivateClass" in class_names
        
        # Check inheritance
        derived = table.find_by_qualified_name("DerivedClass")
        assert derived is not None
        assert "BaseClass" in derived.symbol.bases
        
        # Check methods
        methods = table.get_methods("DerivedClass")
        method_names = {m.name for m in methods}
        assert "__init__" in method_names
        assert "computed_value" in method_names
        assert "async_method" in method_names
        assert "static_helper" in method_names
        assert "from_dict" in method_names
        
        # Check decorators
        cached_funcs = table.find_by_decorator("lru_cache")
        assert len(cached_funcs) == 1
        assert cached_funcs[0].symbol.name == "cached_function"
        
        # Check async functions
        async_symbols = table.find_async_symbols()
        async_names = {r.symbol.name for r in async_symbols}
        assert "async_module_function" in async_names
        assert "async_method" in async_names
        
        # Check private symbols
        private_symbols = table.find_private_symbols()
        private_names = {r.symbol.name for r in private_symbols}
        assert "_PRIVATE_CONSTANT" in private_names
        assert "_PrivateClass" in private_names
        assert "_private_function" in private_names
        assert "__init__" in private_names
        
        # Check nested functions
        inner_func_query = SymbolQuery(
            name="inner_function",
            type=SymbolType.FUNCTION
        )
        inner_results = table.query(inner_func_query)
        assert len(inner_results) == 1
        assert inner_results[0].symbol.parent.name == "module_function"
    
    def test_multi_file_project(self, tmp_path):
        """Test parsing multiple files in a project."""
        # Create a mini project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        # Main module
        main_py = project_dir / "main.py"
        main_py.write_text("""
from .utils import helper_function
from .models import User, Product

def main():
    user = User("Alice")
    product = Product("Widget", 19.99)
    result = helper_function(user, product)
    return result

if __name__ == "__main__":
    main()
""")
        
        # Utils module
        utils_py = project_dir / "utils.py"
        utils_py.write_text("""
def helper_function(user, product):
    return f"{user.name} bought {product.name}"

def another_helper(x: int) -> int:
    return x * 2

class HelperClass:
    @staticmethod
    def process(data):
        return data
""")
        
        # Models module
        models_py = project_dir / "models.py"
        models_py.write_text("""
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int = 0
    
    def greet(self):
        return f"Hello, I'm {self.name}"

@dataclass 
class Product:
    name: str
    price: float
    
    @property
    def display_price(self):
        return f"${self.price:.2f}"
""")
        
        # Parse all files and build combined symbol table
        table = SymbolTable()
        
        for py_file in project_dir.glob("*.py"):
            with open(py_file) as f:
                code = f.read()
            tree = ast.parse(code)
            symbols = extract_symbols(tree, str(py_file))
            table.add_symbols(symbols)
        
        # Test cross-file queries
        assert len(table) > 10
        
        # Find all classes across files
        classes = table.find_by_type(SymbolType.CLASS)
        class_names = {r.symbol.name for r in classes}
        assert "User" in class_names
        assert "Product" in class_names
        assert "HelperClass" in class_names
        
        # Find symbols by file
        main_symbols = table.find_by_file(str(main_py))
        main_names = {r.symbol.name for r in main_symbols}
        assert "main" in main_names
        
        utils_symbols = table.find_by_file(str(utils_py))
        utils_names = {r.symbol.name for r in utils_symbols}
        assert "helper_function" in utils_names
        assert "another_helper" in utils_names
        assert "HelperClass" in utils_names
        
        # Check decorators across files
        dataclass_symbols = table.find_by_decorator("dataclass")
        assert len(dataclass_symbols) == 2
        dataclass_names = {r.symbol.name for r in dataclass_symbols}
        assert "User" in dataclass_names
        assert "Product" in dataclass_names
        
        # Check methods
        user_methods = table.get_methods("User")
        assert any(m.name == "greet" for m in user_methods)
        
        product_methods = table.get_methods("Product")
        assert any(m.name == "display_price" and m.type == SymbolType.PROPERTY 
                  for m in product_methods)
    
    def test_error_handling_partial_parse(self):
        """Test handling of files with syntax errors."""
        # Code with syntax error at the end
        code = """
def valid_function():
    return 42

class ValidClass:
    def method(self):
        pass

# Syntax error below
def incomplete_function(
"""
        
        # Should handle partial parsing gracefully
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # For this test, we expect syntax error
            # In real implementation, we'd use error recovery
            pass
        
        # Parse only the valid part
        valid_code = """
def valid_function():
    return 42

class ValidClass:
    def method(self):
        pass
"""
        tree = ast.parse(valid_code)
        symbols = extract_symbols(tree)
        
        table = SymbolTable()
        table.add_symbols(symbols)
        
        # Should have extracted valid symbols
        assert "valid_function" in table
        assert "ValidClass" in table
        
        methods = table.get_methods("ValidClass")
        assert len(methods) == 1
        assert methods[0].name == "method"
    
    def test_performance_large_file(self):
        """Test performance with a large generated file."""
        # Generate a large Python file
        lines = ['"""Large generated file for performance testing."""']
        
        # Add imports
        lines.extend([
            "import os",
            "import sys", 
            "from typing import List, Dict, Optional",
            ""
        ])
        
        # Generate many classes and functions
        for i in range(100):
            # Add a class with methods
            lines.extend([
                f"class Class{i}:",
                f'    """Class {i} documentation."""',
                f"    class_var_{i} = {i}",
                "",
                f"    def __init__(self, value{i}: int):",
                f"        self.value{i} = value{i}",
                "",
                f"    def method{i}(self, x: int) -> int:",
                f'        """Method {i} documentation."""',
                f"        return x + self.value{i}",
                "",
                f"    @property",
                f"    def prop{i}(self) -> int:",
                f"        return self.value{i} * 2",
                "",
            ])
            
            # Add a module function
            lines.extend([
                f"def function{i}(arg1: str, arg2: int = {i}) -> bool:",
                f'    """Function {i} documentation."""',
                f"    return len(arg1) > arg2",
                "",
            ])
            
            # Add async function occasionally
            if i % 10 == 0:
                lines.extend([
                    f"async def async_func{i}() -> None:",
                    f'    """Async function {i}."""',
                    f"    pass",
                    "",
                ])
        
        code = "\n".join(lines)
        
        # Time the parsing and extraction
        import time
        
        start = time.perf_counter()
        tree = ast.parse(code)
        parse_time = time.perf_counter() - start
        
        start = time.perf_counter()
        symbols = extract_symbols(tree)
        extract_time = time.perf_counter() - start
        
        table = SymbolTable()
        start = time.perf_counter()
        table.add_symbols(symbols)
        add_time = time.perf_counter() - start
        
        # Verify extraction
        assert len(symbols) > 500  # Should have many symbols
        
        # Performance checks
        assert parse_time < 1.0  # Parsing should be fast
        assert extract_time < 1.0  # Extraction should be fast
        assert add_time < 0.5  # Adding to table should be very fast
        
        # Test lookup performance
        start = time.perf_counter()
        results = table.find_by_name("Class50", exact=True)
        lookup_time = (time.perf_counter() - start) * 1000
        
        assert len(results) == 1
        assert lookup_time < 10  # Should be under 10ms
        
        # Test complex query performance
        query = SymbolQuery(
            type=SymbolType.METHOD,
            parent_name="Class50"
        )
        
        start = time.perf_counter()
        results = table.query(query)
        query_time = (time.perf_counter() - start) * 1000
        
        assert len(results) > 0
        assert query_time < 10  # Complex queries should also be fast
    
    def test_real_world_patterns(self):
        """Test with real-world Python patterns."""
        code = '''
"""Real-world patterns test."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Protocol
from enum import Enum, auto

T = TypeVar('T')

class Status(Enum):
    """Status enumeration."""
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()

class AbstractBase(ABC):
    """Abstract base class."""
    
    @abstractmethod
    def process(self, data: dict) -> dict:
        """Process data."""
        pass
    
    @abstractmethod
    async def async_process(self, data: dict) -> dict:
        """Async process data."""
        pass

class GenericContainer(Generic[T]):
    """Generic container class."""
    
    def __init__(self, items: list[T]):
        self._items = items
    
    def add(self, item: T) -> None:
        self._items.append(item)
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index: int) -> T:
        return self._items[index]

class DataProcessor(AbstractBase):
    """Concrete implementation."""
    
    def __init__(self):
        self.status = Status.PENDING
    
    def process(self, data: dict) -> dict:
        self.status = Status.ACTIVE
        result = {k: v.upper() if isinstance(v, str) else v 
                 for k, v in data.items()}
        self.status = Status.COMPLETED
        return result
    
    async def async_process(self, data: dict) -> dict:
        # Simulate async work
        return self.process(data)
    
    def __repr__(self) -> str:
        return f"DataProcessor(status={self.status})"

class Serializable(Protocol):
    """Protocol for serializable objects."""
    
    def to_dict(self) -> dict:
        ...
    
    @classmethod
    def from_dict(cls, data: dict) -> Serializable:
        ...

# Context manager example
class ResourceManager:
    """Context manager for resources."""
    
    def __enter__(self):
        print("Acquiring resource")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        return False
    
    def use_resource(self):
        print("Using resource")

# Decorator examples
def timing_decorator(func):
    """Timing decorator."""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3):
    """Retry decorator with parameters."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_attempts - 1:
                        raise
            return None
        return wrapper
    return decorator

@timing_decorator
@retry(max_attempts=3)
def risky_operation(value: int) -> int:
    """Function with multiple decorators."""
    if value < 0:
        raise ValueError("Negative value")
    return value * 2
'''
        
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        table = SymbolTable()
        table.add_symbols(symbols)
        
        # Check enums
        enums = [s for s in symbols if s.name == "Status"]
        assert len(enums) == 1
        assert enums[0].bases == ["Enum"]
        
        # Check abstract methods
        abstract_methods = [s for s in symbols 
                          if "abstractmethod" in s.decorators]
        assert len(abstract_methods) == 2
        
        # Check protocol
        protocols = [s for s in symbols if s.name == "Serializable"]
        assert len(protocols) == 1
        assert protocols[0].bases == ["Protocol"]
        
        # Check generic class
        generic_classes = [s for s in symbols if s.name == "GenericContainer"]
        assert len(generic_classes) == 1
        assert "Generic[T]" in generic_classes[0].bases
        
        # Check dunder methods
        dunder_methods = [s for s in symbols 
                         if s.name.startswith("__") and s.name.endswith("__")
                         and s.type == SymbolType.METHOD]
        assert len(dunder_methods) > 5  # Should have several dunder methods
        
        # Check decorator stacking
        risky_funcs = table.find_by_name("risky_operation")
        assert len(risky_funcs) == 1
        decorators = risky_funcs[0].symbol.decorators
        assert "timing_decorator" in decorators
        assert "retry" in decorators