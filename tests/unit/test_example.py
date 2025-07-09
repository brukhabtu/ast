"""
Example unit test demonstrating testing patterns for AST library.

This serves as a template for other test files.
"""

import ast
import pytest
from hypothesis import given, strategies as st

# These imports are placeholders - the helpers don't exist yet
# from tests.factories import ASTFactory, CodeFactory
# from tests.helpers import (
#     assert_ast_equal,
#     assert_ast_structure,
#     find_nodes,
#     measure_performance,
# )
# from tests.helpers.hypothesis_strategies import valid_identifier, valid_python_code

# Simple helper for AST comparison
def assert_ast_equal(ast1, ast2):
    """Compare two AST nodes for equality."""
    return ast.dump(ast1) == ast.dump(ast2)

def assert_ast_structure(node, expected_type, **kwargs):
    """Assert that a node has expected type and attributes."""
    assert isinstance(node, expected_type)
    for key, value in kwargs.items():
        assert getattr(node, key) == value
    return node

def valid_identifier():
    """Hypothesis strategy for valid Python identifiers."""
    import string
    first_chars = string.ascii_letters + '_'
    other_chars = string.ascii_letters + string.digits + '_'
    return st.text(min_size=1, max_size=20, alphabet=first_chars).flatmap(
        lambda first: st.text(min_size=0, max_size=19, alphabet=other_chars).map(
            lambda rest: first + rest
        )
    )

def find_nodes(tree, node_type):
    """Find all nodes of a specific type in an AST."""
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            nodes.append(node)
    return nodes

class CodeFactory:
    """Factory for creating test code snippets."""
    
    @staticmethod
    def create_simple_function(name, params):
        """Create a simple function definition."""
        params_str = ", ".join(params)
        return f"def {name}({params_str}):\n    pass"
    
    @staticmethod
    def create_module_with_imports():
        """Create a module with imports and classes."""
        return '''
import os
import sys
from typing import List, Dict

class MyClass:
    def __init__(self):
        self.data = []
    
    def method1(self):
        return len(self.data)
    
    def method2(self, x: int) -> str:
        return str(x)

def standalone_function():
    return MyClass()
'''

class ASTFactory:
    """Factory for creating AST nodes programmatically."""
    
    @staticmethod
    def create_module(body):
        """Create a module AST node."""
        return ast.Module(body=body, type_ignores=[])
    
    @staticmethod
    def create_function(name, args):
        """Create a function definition AST node."""
        arguments = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=a, annotation=None) for a in args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )
        func = ast.FunctionDef(
            name=name,
            args=arguments,
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        # Add required attributes
        func.lineno = 1
        func.col_offset = 0
        return func
    
    @staticmethod
    def create_class(name):
        """Create a class definition AST node."""
        cls = ast.ClassDef(
            name=name,
            bases=[],
            keywords=[],
            body=[ast.Pass()],
            decorator_list=[]
        )
        # Add required attributes
        cls.lineno = 1
        cls.col_offset = 0
        return cls

def measure_performance(func, *args, **kwargs):
    """Measure performance of a function."""
    import time
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed


class TestExampleParser:
    """Example test class for a hypothetical Parser component."""
    
    @pytest.mark.unit
    def test_parse_simple_assignment(self):
        """Test parsing a simple assignment statement."""
        # Arrange
        code = "x = 42"
        
        # Act
        tree = ast.parse(code)
        
        # Assert
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        
        assign = assert_ast_structure(
            tree.body[0],
            ast.Assign,
        )
        assert len(assign.targets) == 1
        assert assign.targets[0].id == "x"
        assert assign.value.value == 42
    
    @pytest.mark.unit
    def test_parse_function_with_decorators(self):
        """Test parsing functions with decorators."""
        # Arrange
        code = """
@decorator1
@decorator2(arg="value")
def my_function(x, y):
    return x + y
"""
        
        # Act
        tree = ast.parse(code)
        
        # Assert
        func = assert_ast_structure(
            tree.body[0],
            ast.FunctionDef,
            name="my_function"
        )
        assert len(func.decorator_list) == 2
        assert len(func.args.args) == 2
    
    @pytest.mark.unit
    @pytest.mark.parametrize("code,expected_error", [
        ("def func(:", "unexpected EOF"),
        ("class Test\n    pass", "expected ':'"),
        ("x = 1 2 3", "invalid syntax"),
    ])
    def test_parse_syntax_errors(self, code, expected_error):
        """Test that syntax errors are properly reported."""
        with pytest.raises(SyntaxError) as exc_info:
            ast.parse(code)
        
        # Could check specific error message if needed
        assert exc_info.value.msg is not None
    
    @pytest.mark.unit
    @given(valid_identifier())
    def test_parse_any_valid_identifier(self, identifier):
        """Property test: parsing any valid identifier should work."""
        code = f"{identifier} = 'test'"
        
        tree = ast.parse(code)
        assign = tree.body[0]
        
        assert isinstance(assign, ast.Assign)
        assert assign.targets[0].id == identifier
    
    @pytest.mark.unit
    def test_parse_preserves_structure(self):
        """Test that parsing preserves AST structure."""
        # Create AST programmatically
        original = ASTFactory.create_module([
            ASTFactory.create_function("func1", ["x"]),
            ASTFactory.create_class("MyClass"),
        ])
        
        # Convert to code and parse back
        code = ast.unparse(original)
        parsed = ast.parse(code)
        
        # Should have same structure (ignoring location info)
        assert len(parsed.body) == len(original.body)
        assert isinstance(parsed.body[0], ast.FunctionDef)
        assert isinstance(parsed.body[1], ast.ClassDef)


class TestExampleSymbolExtraction:
    """Example tests for symbol extraction functionality."""
    
    @pytest.mark.unit
    def test_extract_function_names(self):
        """Test extracting function names from AST."""
        # Arrange
        code = CodeFactory.create_simple_function("test_func", ["a", "b"])
        tree = ast.parse(code)
        
        # Act
        functions = find_nodes(tree, ast.FunctionDef)
        
        # Assert
        assert len(functions) == 1
        assert functions[0].name == "test_func"
    
    @pytest.mark.unit
    def test_extract_nested_symbols(self):
        """Test extracting symbols from nested structures."""
        # Arrange
        code = """
class Outer:
    class Inner:
        def method(self):
            def nested_func():
                pass
            return nested_func
"""
        tree = ast.parse(code)
        
        # Act
        classes = find_nodes(tree, ast.ClassDef)
        functions = find_nodes(tree, ast.FunctionDef)
        
        # Assert
        assert len(classes) == 2
        assert {c.name for c in classes} == {"Outer", "Inner"}
        assert len(functions) == 2
        assert {f.name for f in functions} == {"method", "nested_func"}


class TestExamplePerformance:
    """Example performance tests."""
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_parse_performance(self, large_python_file):
        """Test parsing performance on large files."""
        with open(large_python_file) as f:
            code = f.read()
        
        def parse_file():
            return ast.parse(code)
        
        # Simple performance test
        elapsed = measure_performance(parse_file)
        
        # Assert performance requirements
        assert elapsed < 0.2  # Should parse in < 200ms
    
    @pytest.mark.unit
    @pytest.mark.benchmark
    def test_symbol_extraction_benchmark(self, benchmark):
        """Benchmark symbol extraction speed."""
        tree = ast.parse(CodeFactory.create_module_with_imports())
        
        def extract_all_names():
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.append(node.id)
            return names
        
        result = benchmark(extract_all_names)
        assert len(result) > 0  # Should find some names


# Example of testing with fixtures
def test_with_mock_cache(mock_cache):
    """Example test using the mock cache fixture."""
    # Store AST in cache
    tree = ast.parse("x = 1")
    mock_cache.set("test_key", tree)
    
    # Retrieve from cache
    cached = mock_cache.get("test_key")
    assert cached is not None
    assert_ast_equal(cached, tree)
    
    # Test cache invalidation
    mock_cache.invalidate("test_key")
    assert mock_cache.get("test_key") is None


# Example of integration test structure
@pytest.mark.integration
class TestExampleIntegration:
    """Example integration tests."""
    
    def test_parse_and_analyze_workflow(self, create_test_project):
        """Test complete parse and analyze workflow."""
        # Create a multi-file project
        project = create_test_project({
            "main.py": """
from utils import helper
from models import User

def main():
    user = User("test")
    helper(user)
""",
            "utils.py": """
def helper(obj):
    print(f"Helping {obj}")
""",
            "models.py": """
class User:
    def __init__(self, name):
        self.name = name
"""
        })
        
        # Parse all files
        trees = {}
        for py_file in project.glob("*.py"):
            with open(py_file) as f:
                trees[py_file.name] = ast.parse(f.read())
        
        # Verify cross-file references
        assert len(trees) == 3
        
        # Check imports in main.py
        main_imports = find_nodes(trees["main.py"], ast.ImportFrom)
        assert len(main_imports) == 2
        import_modules = {imp.module for imp in main_imports}
        assert import_modules == {"utils", "models"}