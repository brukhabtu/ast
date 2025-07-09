"""
Edge case tests for symbol extraction.
"""

import ast
import pytest

from astlib.symbols import extract_symbols, SymbolType
from astlib.symbol_table import SymbolTable


class TestEdgeCases:
    """Test edge cases and unusual Python constructs."""
    
    def test_walrus_operator(self):
        """Test handling of walrus operator (Python 3.8+)."""
        code = """
def process_data(items):
    if (n := len(items)) > 10:
        print(f"Processing {n} items")
        return n
    return 0
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Should extract the function but not the walrus variable
        assert len(symbols) == 1
        assert symbols[0].name == "process_data"
    
    def test_match_statement(self):
        """Test handling of match statements (Python 3.10+)."""
        code = """
def handle_command(command):
    match command.split():
        case ["quit"]:
            return "Goodbye"
        case ["hello", name]:
            return f"Hello, {name}"
        case _:
            return "Unknown command"
"""
        try:
            tree = ast.parse(code)
            symbols = extract_symbols(tree)
            
            assert len(symbols) == 1
            assert symbols[0].name == "handle_command"
        except SyntaxError:
            # Skip if Python version doesn't support match
            pytest.skip("Match statements not supported in this Python version")
    
    def test_type_unions(self):
        """Test handling of type union syntax."""
        code = """
from typing import Union

def process(value: int | str) -> str | None:
    if isinstance(value, int):
        return str(value)
    return value

def legacy_process(value: Union[int, str]) -> Union[str, None]:
    return str(value)
"""
        try:
            tree = ast.parse(code)
            symbols = extract_symbols(tree)
            
            funcs = [s for s in symbols if s.type == SymbolType.FUNCTION]
            assert len(funcs) == 2
            
            # Both should have proper signatures
            process_func = next(f for f in funcs if f.name == "process")
            assert "value:" in process_func.signature
            assert "->" in process_func.signature
        except SyntaxError:
            # Fallback for older Python versions
            pytest.skip("Union syntax not supported in this Python version")
    
    def test_positional_only_args(self):
        """Test handling of positional-only arguments (Python 3.8+)."""
        code = """
def positional_only(a, b, /, c, *, d):
    return a + b + c + d

def mixed_args(pos_only, /, standard, *args, kw_only, **kwargs):
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        assert len(symbols) == 2
        
        # Check signatures contain all argument types
        pos_func = next(s for s in symbols if s.name == "positional_only")
        assert "a, b, /, c, *, d" in pos_func.signature
        
        mixed_func = next(s for s in symbols if s.name == "mixed_args")
        assert "/" in mixed_func.signature
        assert "*args" in mixed_func.signature
        assert "**kwargs" in mixed_func.signature
    
    def test_complex_decorators_with_args(self):
        """Test complex decorator patterns with arguments."""
        code = """
def decorator_factory(prefix):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{prefix}: calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_factory("DEBUG")
@decorator_factory(prefix="INFO")
def logged_function():
    pass

class DecoratedClass:
    @property
    @decorator_factory("PROP")
    def decorated_property(self):
        return 42
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Find the decorated function
        logged = next(s for s in symbols if s.name == "logged_function")
        assert "decorator_factory" in logged.decorators
        
        # Property decorators
        prop = next(s for s in symbols if s.name == "decorated_property")
        assert prop.type == SymbolType.PROPERTY
        assert "property" in prop.decorators
    
    def test_generator_expressions_and_comprehensions(self):
        """Test that comprehensions don't create false symbols."""
        code = """
def create_lists():
    # List comprehension
    squares = [x**2 for x in range(10)]
    
    # Dict comprehension
    mapping = {x: x**2 for x in range(5)}
    
    # Set comprehension
    unique = {x % 3 for x in range(10)}
    
    # Generator expression
    gen = (x * 2 for x in range(5))
    
    return squares, mapping, unique, gen

# Nested comprehensions
matrix = [[i * j for j in range(5)] for i in range(5)]
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Should only extract the function and matrix variable
        names = {s.name for s in symbols}
        assert "create_lists" in names
        assert "matrix" in names
        
        # Should not extract comprehension variables
        assert "x" not in names
        assert "i" not in names
        assert "j" not in names
    
    def test_ellipsis_in_signatures(self):
        """Test handling of ellipsis in type hints."""
        code = """
from typing import Callable, Tuple

def accepts_any_callable(func: Callable[..., int]) -> int:
    return func()

def returns_tuple() -> Tuple[int, ...]:
    return (1, 2, 3)

class MyProtocol:
    def method(self, *args: ...) -> None:
        ...
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Check that ellipsis doesn't break parsing
        accepts = next(s for s in symbols if s.name == "accepts_any_callable")
        assert "Callable[..., int]" in accepts.signature
        
        returns = next(s for s in symbols if s.name == "returns_tuple")
        assert "Tuple[int, ...]" in returns.signature
    
    def test_nested_classes(self):
        """Test deeply nested class definitions."""
        code = """
class Outer:
    class Middle:
        class Inner:
            class Deepest:
                @staticmethod
                def deep_method():
                    pass
            
            def inner_method(self):
                pass
        
        def middle_method(self):
            pass
    
    def outer_method(self):
        pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        table = SymbolTable()
        table.add_symbols(symbols)
        
        # Check nested structure
        deepest_method = table.find_by_qualified_name("Outer.Middle.Inner.Deepest.deep_method")
        assert deepest_method is not None
        assert deepest_method.symbol.name == "deep_method"
        
        # Check all classes are found
        classes = table.find_by_type(SymbolType.CLASS)
        class_names = {r.symbol.name for r in classes}
        assert all(name in class_names for name in ["Outer", "Middle", "Inner", "Deepest"])
    
    def test_dynamic_class_creation(self):
        """Test handling of dynamic class creation."""
        code = """
def create_class(name):
    class DynamicClass:
        dynamic_attr = name
        
        def dynamic_method(self):
            return self.dynamic_attr
    
    return DynamicClass

# Type creation
MyType = type('MyType', (object,), {'attr': 42})

# Using type annotations
NewType = type('NewType', (), {})
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Should find the inner class definition
        dynamic = next((s for s in symbols if s.name == "DynamicClass"), None)
        assert dynamic is not None
        assert dynamic.parent.name == "create_class"
        
        # Type assignments should be variables
        type_vars = [s for s in symbols if s.name in ["MyType", "NewType"]]
        assert all(s.type == SymbolType.VARIABLE for s in type_vars)
    
    def test_multiple_inheritance_complex(self):
        """Test complex multiple inheritance patterns."""
        code = """
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    '''Diamond inheritance'''
    pass

class E(D, dict):
    '''Mixing custom and builtin'''
    pass

class F(E, *[list]):
    '''With unpacking (if supported)'''
    pass
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        
        # Find class symbols
        class_map = {s.name: s for s in symbols if s.type == SymbolType.CLASS}
        
        assert class_map["D"].bases == ["B", "C"]
        assert class_map["E"].bases == ["D", "dict"]
        
        # F might have special handling depending on Python version
        assert "F" in class_map
    
    def test_special_method_names(self):
        """Test extraction of special method names."""
        code = """
class SpecialMethods:
    def __init__(self):
        pass
    
    def __str__(self):
        return "special"
    
    def __repr__(self):
        return "SpecialMethods()"
    
    def __call__(self, x):
        return x
    
    def __getattr__(self, name):
        return None
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
    
    def __delattr__(self, name):
        super().__delattr__(name)
    
    def __getitem__(self, key):
        return key
    
    def __setitem__(self, key, value):
        pass
    
    def __len__(self):
        return 0
    
    def __iter__(self):
        return iter([])
    
    def __next__(self):
        raise StopIteration
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
"""
        tree = ast.parse(code)
        symbols = extract_symbols(tree)
        table = SymbolTable()
        table.add_symbols(symbols)
        
        # Get all methods
        methods = table.get_methods("SpecialMethods")
        method_names = {m.name for m in methods}
        
        # Check common special methods
        special_methods = [
            "__init__", "__str__", "__repr__", "__call__",
            "__getattr__", "__setattr__", "__delattr__",
            "__getitem__", "__setitem__", "__len__",
            "__iter__", "__next__", "__enter__", "__exit__",
            "__aenter__", "__aexit__"
        ]
        
        for special in special_methods:
            assert special in method_names
        
        # Check async methods
        async_methods = [m for m in methods if m.is_async]
        async_names = {m.name for m in async_methods}
        assert "__aenter__" in async_names
        assert "__aexit__" in async_names