"""
Meta-tests for the test infrastructure itself.

These tests ensure our test utilities, fixtures, and helpers work correctly.
"""

import ast
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from tests.factories import ASTFactory, CodeFactory, EdgeCaseFactory, TestDataGenerator
from tests.helpers import (
    assert_ast_equal,
    assert_ast_structure,
    count_nodes,
    find_nodes,
    normalize_ast,
    create_temp_python_file,
    validate_python_syntax,
    measure_performance,
    assert_performance,
)
from tests.helpers.hypothesis_strategies import (
    valid_identifier,
    ast_function_def,
    valid_python_code,
)


class TestFactories:
    """Test the AST and code factories."""
    
    def test_ast_factory_creates_valid_nodes(self):
        """Test that ASTFactory creates valid AST nodes."""
        # Test each factory method
        name = ASTFactory.create_name("test_var")
        assert isinstance(name, ast.Name)
        assert name.id == "test_var"
        
        constant = ASTFactory.create_constant(42)
        assert isinstance(constant, ast.Constant)
        assert constant.value == 42
        
        func = ASTFactory.create_function("test_func", ["x", "y"])
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "test_func"
        assert len(func.args.args) == 2
        
        cls = ASTFactory.create_class("TestClass")
        assert isinstance(cls, ast.ClassDef)
        assert cls.name == "TestClass"
    
    def test_ast_factory_creates_parseable_module(self):
        """Test that we can create a complete parseable module."""
        # Create a module with various elements
        module = ASTFactory.create_module([
            ASTFactory.create_import([("os", None)]),
            ASTFactory.create_function("func1"),
            ASTFactory.create_class("Class1"),
            ASTFactory.create_assign("x", ASTFactory.create_constant(10)),
        ])
        
        # Should be a valid module
        assert isinstance(module, ast.Module)
        assert len(module.body) == 4
        
        # Should be compilable
        code = compile(module, "<test>", "exec")
        assert code is not None
    
    def test_code_factory_generates_valid_python(self):
        """Test that CodeFactory generates valid Python code."""
        test_cases = [
            CodeFactory.create_simple_function(),
            CodeFactory.create_simple_class(),
            CodeFactory.create_module_with_imports(),
            CodeFactory.create_async_patterns(),
        ]
        
        for code in test_cases:
            is_valid, error = validate_python_syntax(code)
            assert is_valid, f"Invalid Python generated: {error}\nCode:\n{code}"
    
    def test_edge_case_factory_creates_parseable_code(self):
        """Test that edge cases are still valid Python."""
        test_cases = [
            EdgeCaseFactory.create_deeply_nested_code(3),
            EdgeCaseFactory.create_unicode_identifiers(),
            EdgeCaseFactory.create_complex_decorators(),
            EdgeCaseFactory.create_type_annotation_edge_cases(),
        ]
        
        for code in test_cases:
            try:
                tree = ast.parse(code)
                assert isinstance(tree, ast.Module)
            except SyntaxError as e:
                pytest.fail(f"Edge case not parseable: {e}\nCode:\n{code}")
    
    def test_test_data_generator_completeness(self):
        """Test that TestDataGenerator provides comprehensive test data."""
        suite = TestDataGenerator.generate_test_suite()
        
        # Should have various pattern types
        assert len(suite) >= 5
        assert "simple_function" in suite
        assert "async_patterns" in suite
        
        # Each pattern should be valid
        for name, pattern in suite.items():
            is_valid, error = validate_python_syntax(pattern.code)
            assert is_valid, f"Pattern {name} invalid: {error}"
    
    def test_syntax_error_generation(self):
        """Test that syntax error cases are actually invalid."""
        errors = TestDataGenerator.generate_syntax_errors()
        
        for code, description in errors:
            is_valid, error = validate_python_syntax(code)
            assert not is_valid, f"Expected syntax error for: {description}\nCode: {code}"


class TestHelpers:
    """Test the helper utilities."""
    
    def test_assert_ast_equal(self):
        """Test AST equality assertion."""
        # Equal ASTs should pass
        tree1 = ast.parse("x = 1")
        tree2 = ast.parse("x = 1")
        assert_ast_equal(tree1, tree2)
        
        # Different ASTs should fail
        tree3 = ast.parse("x = 2")
        with pytest.raises(AssertionError):
            assert_ast_equal(tree1, tree3)
    
    def test_assert_ast_structure(self):
        """Test AST structure assertion."""
        tree = ast.parse("def func(x): return x * 2")
        func_def = tree.body[0]
        
        # Should pass with correct structure
        node = assert_ast_structure(func_def, ast.FunctionDef, name="func")
        assert isinstance(node, ast.FunctionDef)
        
        # Should fail with wrong type
        with pytest.raises(AssertionError):
            assert_ast_structure(func_def, ast.ClassDef)
        
        # Should fail with wrong field value
        with pytest.raises(AssertionError):
            assert_ast_structure(func_def, ast.FunctionDef, name="wrong_name")
    
    def test_find_nodes(self):
        """Test finding nodes by type."""
        code = """
def func1():
    pass

def func2():
    def inner():
        pass
    return inner

class MyClass:
    def method(self):
        pass
"""
        tree = ast.parse(code)
        
        # Find all functions (including methods)
        functions = find_nodes(tree, ast.FunctionDef)
        assert len(functions) == 4
        assert {f.name for f in functions} == {"func1", "func2", "inner", "method"}
        
        # Find all classes
        classes = find_nodes(tree, ast.ClassDef)
        assert len(classes) == 1
        assert classes[0].name == "MyClass"
    
    def test_count_nodes(self):
        """Test counting nodes."""
        tree = ast.parse("x = 1 + 2 * 3")
        
        # Count all nodes
        total = count_nodes(tree)
        assert total > 5  # Should have Module, Assign, BinOp, etc.
        
        # Count specific type
        binops = count_nodes(tree, ast.BinOp)
        assert binops == 2  # + and *
    
    def test_normalize_ast(self):
        """Test AST normalization."""
        # Parse code with location info
        tree = ast.parse("x = 1", filename="test.py", mode="exec")
        
        # Original should have location info
        assign = tree.body[0]
        assert hasattr(assign, 'lineno')
        
        # Normalized should not
        normalized = normalize_ast(tree)
        norm_assign = normalized.body[0]
        assert not hasattr(norm_assign, 'lineno')
    
    def test_create_temp_python_file(self):
        """Test temporary file creation."""
        content = "print('Hello, World!')"
        
        path = create_temp_python_file(content)
        assert path.exists()
        assert path.suffix == ".py"
        assert path.read_text() == content
        
        # Clean up
        path.unlink()
    
    def test_validate_python_syntax(self):
        """Test syntax validation."""
        # Valid code
        is_valid, error = validate_python_syntax("x = 1")
        assert is_valid
        assert error is None
        
        # Invalid code
        is_valid, error = validate_python_syntax("def func(")
        assert not is_valid
        assert "SyntaxError" in error


class TestPerformanceHelpers:
    """Test performance measurement utilities."""
    
    def test_measure_performance(self):
        """Test basic performance measurement."""
        def simple_func(n):
            return sum(range(n))
        
        result = measure_performance(simple_func, 1000, iterations=10)
        
        assert result.iterations == 10
        assert result.mean > 0
        assert result.min <= result.mean <= result.max
        assert result.duration >= result.mean * result.iterations
    
    def test_assert_performance_passes(self):
        """Test performance assertion when it should pass."""
        def fast_func():
            return 42
        
        # Should pass with generous limit
        assert_performance(fast_func, max_duration=1.0, iterations=10)
    
    def test_assert_performance_fails(self):
        """Test performance assertion when it should fail."""
        def slow_func():
            import time
            time.sleep(0.01)
        
        # Should fail with tight limit
        with pytest.raises(AssertionError, match="performance requirement not met"):
            assert_performance(slow_func, max_duration=0.001, iterations=5)


class TestFixtures:
    """Test pytest fixtures from conftest.py."""
    
    def test_sample_ast_tree_fixture(self, sample_ast_tree):
        """Test the sample AST tree fixture."""
        assert isinstance(sample_ast_tree, ast.Module)
        
        # Should contain expected elements
        functions = find_nodes(sample_ast_tree, ast.FunctionDef)
        classes = find_nodes(sample_ast_tree, ast.ClassDef)
        
        assert len(functions) >= 1
        assert len(classes) >= 1
    
    def test_create_test_project_fixture(self, create_test_project, temp_project_dir):
        """Test the project creation fixture."""
        files = {
            "main.py": "from utils import helper",
            "utils.py": "def helper(): pass",
        }
        
        project_path = create_test_project(files, temp_project_dir)
        
        assert project_path.exists()
        assert (project_path / "main.py").exists()
        assert (project_path / "utils.py").exists()
    
    def test_mock_cache_fixture(self, mock_cache):
        """Test the mock cache fixture."""
        # Should implement the cache protocol
        assert hasattr(mock_cache, 'get')
        assert hasattr(mock_cache, 'set')
        assert hasattr(mock_cache, 'invalidate')
        assert hasattr(mock_cache, 'clear')
        
        # Should actually work
        mock_cache.set("key", ast.parse("x = 1"))
        result = mock_cache.get("key")
        assert isinstance(result, ast.Module)
        
        mock_cache.invalidate("key")
        assert mock_cache.get("key") is None


class TestHypothesisStrategies:
    """Test custom Hypothesis strategies."""
    
    @given(valid_identifier())
    def test_valid_identifier_strategy(self, identifier):
        """Test that valid_identifier generates valid Python identifiers."""
        assert identifier.isidentifier()
        
        # Should not be a keyword
        import keyword
        assert not keyword.iskeyword(identifier)
    
    @given(ast_function_def())
    def test_ast_function_def_strategy(self, func_def):
        """Test that ast_function_def generates valid function definitions."""
        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name.isidentifier()
        assert len(func_def.body) >= 1
        
        # Should be part of a compilable module
        module = ast.Module(body=[func_def], type_ignores=[])
        code = compile(module, "<test>", "exec")
        assert code is not None
    
    @given(valid_python_code())
    def test_valid_python_code_strategy(self, code):
        """Test that valid_python_code generates parseable Python."""
        try:
            tree = ast.parse(code)
            assert isinstance(tree, ast.Module)
        except SyntaxError:
            pytest.fail(f"Generated invalid Python code: {code}")


class TestTestInfrastructureIntegration:
    """Integration tests for test infrastructure."""
    
    def test_factory_and_helper_integration(self):
        """Test that factories and helpers work together."""
        # Create AST nodes with factory
        func1 = ASTFactory.create_function("func1", ["x"])
        func2 = ASTFactory.create_function("func2", ["y"])
        module = ASTFactory.create_module([func1, func2])
        
        # Use helpers to analyze
        functions = find_nodes(module, ast.FunctionDef)
        assert len(functions) == 2
        
        # Use assert helpers
        assert_ast_structure(functions[0], ast.FunctionDef, name="func1")
        assert_ast_structure(functions[1], ast.FunctionDef, name="func2")
    
    def test_code_generation_and_validation(self):
        """Test code generation and validation pipeline."""
        # Generate various code patterns
        patterns = TestDataGenerator.generate_test_suite()
        
        for name, pattern in patterns.items():
            # Validate syntax
            is_valid, error = validate_python_syntax(pattern.code)
            assert is_valid, f"Pattern {name} is invalid"
            
            # Parse and analyze
            tree = ast.parse(pattern.code)
            node_count = count_nodes(tree)
            assert node_count > 0
    
    def test_performance_measurement_workflow(self):
        """Test complete performance measurement workflow."""
        def parse_code(code: str) -> ast.AST:
            return ast.parse(code)
        
        # Generate test code
        code = CodeFactory.create_simple_function("benchmark_func", ["a", "b", "c"])
        
        # Measure performance
        result = measure_performance(parse_code, code, iterations=100)
        
        # Verify results make sense
        assert result.mean < 0.001  # Parsing simple code should be fast
        assert result.iterations == 100


# Run meta-tests when module is executed
if __name__ == "__main__":
    pytest.main([__file__, "-v"])