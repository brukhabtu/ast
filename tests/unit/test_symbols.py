"""
Unit tests for symbol extraction.
"""

import ast
import pytest
from typing import List

from astlib.symbols import (
    Symbol, SymbolType, Position, SymbolExtractor, extract_symbols
)


class TestSymbolExtraction:
    """Test basic symbol extraction functionality."""
    
    def test_extract_simple_function(self):
        """Test extracting a simple function definition."""
        code = """
def hello_world():
    print("Hello, World!")
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 1
        assert symbols[0].name == "hello_world"
        assert symbols[0].type == SymbolType.FUNCTION
        assert symbols[0].position.line == 2
        assert symbols[0].signature == "()"
        assert symbols[0].is_async is False
    
    def test_extract_function_with_args(self):
        """Test extracting function with various argument types."""
        code = """
def complex_function(a: int, b: str = "default", *args, c: float, **kwargs) -> bool:
    return True
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 1
        func = symbols[0]
        assert func.name == "complex_function"
        assert func.type == SymbolType.FUNCTION
        assert "a: int" in func.signature
        assert 'b: str = "default"' in func.signature or "b: str = 'default'" in func.signature
        assert "*args" in func.signature
        assert "c: float" in func.signature
        assert "**kwargs" in func.signature
        assert "-> bool" in func.signature
    
    def test_extract_async_function(self):
        """Test extracting async function."""
        code = """
async def fetch_data(url: str) -> dict:
    return {"data": "example"}
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 1
        assert symbols[0].name == "fetch_data"
        assert symbols[0].type == SymbolType.ASYNC_FUNCTION
        assert symbols[0].is_async is True
        assert symbols[0].signature == "(url: str) -> dict"
    
    def test_extract_class_definition(self):
        """Test extracting class definition with inheritance."""
        code = """
class MyClass(BaseClass, Mixin):
    '''A test class'''
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 1
        cls = symbols[0]
        assert cls.name == "MyClass"
        assert cls.type == SymbolType.CLASS
        assert cls.bases == ["BaseClass", "Mixin"]
        assert cls.docstring == "A test class"
    
    def test_extract_class_with_methods(self):
        """Test extracting class with methods and properties."""
        code = """
class Calculator:
    def __init__(self, value: int = 0):
        self.value = value
    
    def add(self, x: int) -> int:
        return self.value + x
    
    @property
    def double(self) -> int:
        return self.value * 2
    
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 6  # class + 5 methods
        
        # Check class
        cls = symbols[0]
        assert cls.name == "Calculator"
        assert cls.type == SymbolType.CLASS
        assert len(cls.children) == 5
        
        # Check methods
        methods = {s.name: s for s in symbols[1:]}
        
        assert methods["__init__"].type == SymbolType.METHOD
        assert methods["__init__"].is_private is True
        
        assert methods["add"].type == SymbolType.METHOD
        assert methods["add"].signature == "(self, x: int) -> int"
        
        assert methods["double"].type == SymbolType.PROPERTY
        assert "property" in methods["double"].decorators
        
        assert methods["static_method"].type == SymbolType.METHOD
        assert "staticmethod" in methods["static_method"].decorators
        
        assert methods["class_method"].type == SymbolType.METHOD
        assert "classmethod" in methods["class_method"].decorators
    
    def test_extract_nested_functions(self):
        """Test extracting nested function definitions."""
        code = """
def outer_function():
    def inner_function():
        def deeply_nested():
            pass
        return deeply_nested
    return inner_function
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 3
        
        # Check nesting structure
        outer = symbols[0]
        assert outer.name == "outer_function"
        assert len(outer.children) == 1
        
        inner = outer.children[0]
        assert inner.name == "inner_function"
        assert inner.parent == outer
        assert len(inner.children) == 1
        
        deeply = inner.children[0]
        assert deeply.name == "deeply_nested"
        assert deeply.parent == inner
        assert deeply.qualified_name == "outer_function.inner_function.deeply_nested"
    
    def test_extract_decorators(self):
        """Test extracting functions with decorators."""
        code = """
@decorator1
@decorator2.sub_decorator
@decorator3("arg")
def decorated_function():
    pass

@dataclass
class DecoratedClass:
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 2
        
        func = symbols[0]
        assert func.name == "decorated_function"
        assert "decorator1" in func.decorators
        assert "decorator2.sub_decorator" in func.decorators
        assert "decorator3" in func.decorators
        
        cls = symbols[1]
        assert cls.name == "DecoratedClass"
        assert "dataclass" in cls.decorators
    
    def test_extract_variables_and_constants(self):
        """Test extracting variable and constant assignments."""
        code = """
# Module level
x = 10
MESSAGE: str = "Hello"
CONSTANT_VALUE = 42

class MyClass:
    # Class level
    class_var = "shared"
    CONST_IN_CLASS = 100
    
    def __init__(self):
        # Instance level
        self.instance_var = "instance"
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Find symbols by name
        symbol_map = {s.name: s for s in symbols}
        
        assert symbol_map["x"].type == SymbolType.VARIABLE
        assert symbol_map["x"].is_private is False
        
        assert symbol_map["MESSAGE"].type == SymbolType.VARIABLE
        assert symbol_map["MESSAGE"].type_annotation == "str"
        
        assert symbol_map["CONSTANT_VALUE"].type == SymbolType.CONSTANT
        
        assert symbol_map["class_var"].type == SymbolType.VARIABLE
        assert symbol_map["class_var"].parent.name == "MyClass"
        
        assert symbol_map["CONST_IN_CLASS"].type == SymbolType.CONSTANT
        assert symbol_map["CONST_IN_CLASS"].parent.name == "MyClass"
    
    def test_extract_imports(self):
        """Test extracting import statements."""
        code = """
import os
import sys as system
from pathlib import Path
from typing import List, Dict as DictType
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Get import symbols
        imports = [s for s in symbols if s.type == SymbolType.IMPORT]
        import_names = {s.name for s in imports}
        
        assert len(imports) == 5
        assert "os" in import_names
        assert "system" in import_names  # aliased
        assert "Path" in import_names
        assert "List" in import_names
        assert "DictType" in import_names  # aliased
    
    def test_extract_async_methods(self):
        """Test extracting async methods in classes."""
        code = """
class AsyncAPI:
    async def fetch(self, url: str):
        pass
    
    async def process(self, data):
        pass
    
    def sync_method(self):
        pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        methods = [s for s in symbols if s.type == SymbolType.METHOD]
        
        async_methods = [m for m in methods if m.is_async]
        assert len(async_methods) == 2
        assert async_methods[0].name == "fetch"
        assert async_methods[1].name == "process"
        
        sync_methods = [m for m in methods if not m.is_async]
        assert len(sync_methods) == 1
        assert sync_methods[0].name == "sync_method"
    
    def test_extract_docstrings(self):
        """Test extracting docstrings from various constructs."""
        code = '''
"""Module docstring"""

def function_with_docstring():
    """This function has a docstring."""
    pass

class ClassWithDocstring:
    """
    Multi-line class docstring.
    
    With additional details.
    """
    
    def method_with_docstring(self):
        """Method docstring"""
        pass
'''
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        func = next(s for s in symbols if s.name == "function_with_docstring")
        assert func.docstring == "This function has a docstring."
        
        cls = next(s for s in symbols if s.name == "ClassWithDocstring")
        assert "Multi-line class docstring" in cls.docstring
        
        method = next(s for s in symbols if s.name == "method_with_docstring")
        assert method.docstring == "Method docstring"
    
    def test_private_symbols(self):
        """Test identification of private symbols."""
        code = """
def public_function():
    pass

def _private_function():
    pass

def __very_private():
    pass

class MyClass:
    def public_method(self):
        pass
    
    def _protected_method(self):
        pass
    
    def __private_method(self):
        pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        private_symbols = [s for s in symbols if s.is_private]
        public_symbols = [s for s in symbols if not s.is_private]
        
        assert len(private_symbols) == 4  # _private_function, __very_private, _protected_method, __private_method
        assert len(public_symbols) == 3  # public_function, MyClass, public_method
    
    def test_empty_ast(self):
        """Test handling of empty AST."""
        code = ""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 0
    
    def test_syntax_preserving_positions(self):
        """Test that positions are correctly preserved."""
        code = """def func1():
    pass

class MyClass:
    def method1(self):
        pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        func1 = next(s for s in symbols if s.name == "func1")
        assert func1.position.line == 1
        
        my_class = next(s for s in symbols if s.name == "MyClass")
        assert my_class.position.line == 4
        
        method1 = next(s for s in symbols if s.name == "method1")
        assert method1.position.line == 5


class TestEdgeCases:
    """Test edge cases and complex scenarios."""
    
    def test_lambda_functions(self):
        """Test that lambda functions are not extracted as symbols."""
        code = """
func = lambda x: x * 2
sorted_list = sorted(items, key=lambda item: item.name)
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Only variables should be extracted, not lambdas
        assert all(s.type in (SymbolType.VARIABLE, SymbolType.IMPORT) for s in symbols)
    
    def test_metaclass_definition(self):
        """Test class with metaclass."""
        code = """
class Meta(type):
    pass

class MyClass(metaclass=Meta):
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        classes = [s for s in symbols if s.type == SymbolType.CLASS]
        assert len(classes) == 2
        
        meta = next(c for c in classes if c.name == "Meta")
        assert meta.bases == ["type"]
        
        # Note: metaclass info is not currently extracted, but class is still found
        my_class = next(c for c in classes if c.name == "MyClass")
        assert my_class.name == "MyClass"
    
    def test_complex_decorators(self):
        """Test complex decorator patterns."""
        code = """
@decorator_factory(arg1, arg2="value")
@chain.of.decorators
def complex_decorated():
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        func = symbols[0]
        assert "decorator_factory" in func.decorators
        assert "chain.of.decorators" in func.decorators
    
    def test_type_annotated_class_variables(self):
        """Test class variables with type annotations."""
        code = """
class TypedClass:
    count: int = 0
    items: List[str] = []
    mapping: Dict[str, Any]
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Find class variables
        class_vars = [s for s in symbols if s.type == SymbolType.VARIABLE and s.parent]
        
        count_var = next(v for v in class_vars if v.name == "count")
        assert count_var.type_annotation == "int"
        
        items_var = next(v for v in class_vars if v.name == "items")
        assert count_var.type_annotation == "int"
        
        mapping_var = next(v for v in class_vars if v.name == "mapping")
        assert mapping_var.type_annotation == "Dict[str, Any]"
    
    def test_property_setter_deleter(self):
        """Test property with setter and deleter."""
        code = """
class PropertyClass:
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
    
    @value.deleter
    def value(self):
        del self._value
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # All three should be detected, but first one as property
        value_symbols = [s for s in symbols if s.name == "value"]
        property_symbols = [s for s in value_symbols if s.type == SymbolType.PROPERTY]
        
        assert len(property_symbols) >= 1
        assert property_symbols[0].decorators == ["property"]