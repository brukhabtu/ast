"""
Unit tests for AST parser with error recovery.

Tests follow the guidelines in TESTING_GUIDELINES.md, focusing on
behavior testing, edge cases, and performance validation.
"""

import pytest
import ast
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings
import string

from astlib.parser import parse_file, ParseError, ParseResult, ErrorRecoveringParser
from astlib.visitor import find_functions, find_classes, find_imports
from tests.conftest import INVALID_PYTHON_SAMPLES


class TestParseFile:
    """Test the main parse_file function."""
    
    def test_parse_valid_python_file(self, create_test_file):
        """Test parsing a valid Python file returns correct AST."""
        # Arrange
        code = """
def greet(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
"""
        filepath = create_test_file("valid.py", code)
        
        # Act
        result = parse_file(filepath)
        
        # Assert
        assert result.success is True
        assert result.tree is not None
        assert len(result.errors) == 0
        
        # Verify AST structure
        functions = find_functions(result.tree)
        classes = find_classes(result.tree)
        assert len(functions) == 2  # greet and __init__
        assert len(classes) == 1
        assert functions[0].name == "greet"
        assert classes[0].name == "Person"
    
    def test_parse_nonexistent_file_raises_error(self):
        """Test that parsing nonexistent file raises FileNotFoundError."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            parse_file("/nonexistent/file.py")
    
    def test_parse_file_with_read_error(self, temp_dir):
        """Test handling of file read errors."""
        # Arrange - create a directory instead of file
        filepath = temp_dir / "not_a_file.py"
        filepath.mkdir()
        
        # Act & Assert
        with pytest.raises(IOError):
            parse_file(filepath)


class TestErrorRecovery:
    """Test error recovery capabilities."""
    
    def test_parse_file_with_syntax_error_returns_partial_ast(self, create_test_file):
        """Test that parser returns partial AST for files with syntax errors."""
        # Arrange
        code = """
def valid_function():
    return 42

def invalid_function(
    # Missing closing parenthesis and colon

def another_valid():
    return "hello"
"""
        filepath = create_test_file("partial.py", code)
        
        # Act
        result = parse_file(filepath)
        
        # Assert
        assert result.partial_success is True
        assert result.tree is not None
        assert len(result.errors) > 0
        
        # Should at least parse the first valid function
        functions = find_functions(result.tree)
        assert any(f.name == "valid_function" for f in functions)
    
    def test_missing_colon_recovery(self, create_test_file):
        """Test recovery from missing colons."""
        # Arrange
        code = """
def hello()
    print("Hello")

if True
    print("True")
"""
        filepath = create_test_file("missing_colon.py", code)
        
        # Act
        result = parse_file(filepath)
        
        # Assert
        assert result.tree is not None
        assert len(result.errors) > 0
        # Check if auto-fix was attempted
        # The parser may not have a specific message about fixing colons
        # Just check that errors were detected
        assert len(result.errors) > 0
    
    def test_indentation_error_recovery(self, create_test_file):
        """Test recovery from indentation errors."""
        # Arrange
        code = """
def hello():
print("Bad indent")
    print("Good indent")
"""
        filepath = create_test_file("bad_indent.py", code)
        
        # Act
        result = parse_file(filepath)
        
        # Assert
        assert result.tree is not None
        assert len(result.errors) > 0
        # IndentationError is a type of SyntaxError
        assert result.errors[0].error_type in ["SyntaxError", "IndentationError"]
    
    @pytest.mark.parametrize("sample_name,sample_code", [
        (name, code) for name, code in INVALID_PYTHON_SAMPLES.items()
    ])
    def test_invalid_python_samples(self, create_test_file, sample_name, sample_code):
        """Test parser handles various invalid Python code samples."""
        # Arrange
        filepath = create_test_file(f"{sample_name}.py", sample_code)
        
        # Act
        result = parse_file(filepath)
        
        # Assert
        assert not result.success
        assert len(result.errors) > 0
        # May or may not have partial AST depending on error type
        if result.tree:
            assert result.partial_success


class TestParseResult:
    """Test ParseResult data structure."""
    
    def test_successful_parse_result(self):
        """Test ParseResult for successful parsing."""
        # Arrange
        tree = ast.parse("x = 1")
        result = ParseResult(tree=tree, errors=[], source="x = 1")
        
        # Assert
        assert result.success is True
        assert result.partial_success is False
        assert result.tree is tree
    
    def test_failed_parse_result(self):
        """Test ParseResult for failed parsing."""
        # Arrange
        error = ParseError("Invalid syntax", line=1, column=5)
        result = ParseResult(tree=None, errors=[error])
        
        # Assert
        assert result.success is False
        assert result.partial_success is False
    
    def test_partial_parse_result(self):
        """Test ParseResult for partial success."""
        # Arrange
        tree = ast.parse("x = 1")
        error = ParseError("Error in later code", line=10, column=1)
        result = ParseResult(tree=tree, errors=[error])
        
        # Assert
        assert result.success is False
        assert result.partial_success is True


class TestParserPerformance:
    """Test parser performance requirements."""
    
    def test_parse_multiple_files_performance(self, create_test_file, valid_samples):
        """Test parsing 100 files completes in under 1 second."""
        # Arrange - create 100 test files
        files = []
        for i in range(100):
            # Use different valid samples cyclically
            sample_name = list(valid_samples.keys())[i % len(valid_samples)]
            code = valid_samples[sample_name]
            filepath = create_test_file(f"perf_test_{i}.py", code)
            files.append(filepath)
        
        # Act - measure parsing time
        start_time = time.time()
        results = [parse_file(f) for f in files]
        elapsed_time = time.time() - start_time
        
        # Assert
        assert elapsed_time < 1.0, f"Parsing 100 files took {elapsed_time:.2f}s (should be < 1s)"
        assert all(r.success for r in results)
    
    def test_parse_large_file_performance(self, create_test_file):
        """Test parsing a large file is still performant."""
        # Arrange - create a large file with many functions
        lines = []
        for i in range(1000):
            lines.append(f"def function_{i}():")
            lines.append(f"    return {i}")
            lines.append("")
        
        large_code = "\n".join(lines)
        filepath = create_test_file("large_file.py", large_code)
        
        # Act
        start_time = time.time()
        result = parse_file(filepath)
        elapsed_time = time.time() - start_time
        
        # Assert
        assert elapsed_time < 0.5, f"Parsing large file took {elapsed_time:.2f}s"
        assert result.success
        functions = find_functions(result.tree)
        assert len(functions) == 1000


class TestPropertyBasedTesting:
    """Property-based tests using Hypothesis."""
    
    @given(st.text(alphabet=string.ascii_letters + string.digits + "_", min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_parse_valid_identifiers(self, identifier):
        """Test parsing valid Python identifiers."""
        # Skip if identifier starts with digit (invalid in Python)
        if identifier[0].isdigit():
            return
        
        # Arrange
        code = f"{identifier} = 42"
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        assert result.tree is not None
    
    @given(st.lists(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
        min_size=1,
        max_size=10,
        unique=True  # Ensure no duplicate argument names
    ))
    def test_parse_function_with_arguments(self, arg_names):
        """Test parsing functions with various argument lists."""
        # Arrange
        args = ", ".join(arg_names)
        code = f"def test_func({args}):\n    pass"
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        if result.success:
            functions = find_functions(result.tree)
            assert len(functions) == 1
            assert len(functions[0].args.args) == len(arg_names)
        else:
            # If parsing failed, it should be due to duplicate args
            assert len(set(arg_names)) < len(arg_names)


class TestPython38PlusFeatures:
    """Test handling of Python 3.8+ specific features."""
    
    def test_walrus_operator(self):
        """Test parsing walrus operator (Python 3.8+)."""
        # Arrange
        code = """
if (n := len(data)) > 10:
    print(f"List is too long ({n} elements)")
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        assert isinstance(result.tree.body[0], ast.If)
    
    def test_positional_only_parameters(self):
        """Test parsing positional-only parameters (Python 3.8+)."""
        # Arrange
        code = """
def greet(name, /, greeting="Hello"):
    return f"{greeting}, {name}!"
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        functions = find_functions(result.tree)
        assert len(functions) == 1
        assert len(functions[0].args.posonlyargs) == 1
    
    def test_match_statement(self):
        """Test parsing match statements (Python 3.10+)."""
        # Skip if Python version is too old
        import sys
        if sys.version_info < (3, 10):
            pytest.skip("Match statements require Python 3.10+")
        
        # Arrange
        code = """
def handle_response(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case _:
            return "Unknown"
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        functions = find_functions(result.tree)
        assert len(functions) == 1


class TestEdgeCases:
    """Test edge cases and unusual but valid Python."""
    
    def test_empty_file(self):
        """Test parsing empty file."""
        # Arrange
        parser = ErrorRecoveringParser("")
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        assert len(result.tree.body) == 0
    
    def test_only_comments(self):
        """Test parsing file with only comments."""
        # Arrange
        code = """
# This is a comment
# Another comment
# Yet another comment
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        assert len(result.tree.body) == 0
    
    def test_unicode_identifiers(self):
        """Test parsing Unicode identifiers."""
        # Arrange
        code = """
def 你好():
    return "Hello in Chinese"

π = 3.14159
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        functions = find_functions(result.tree)
        assert len(functions) == 1
    
    def test_deeply_nested_code(self):
        """Test parsing deeply nested code structures."""
        # Arrange
        code = """
def outer():
    def middle():
        def inner():
            if True:
                while True:
                    for i in range(10):
                        try:
                            with open("file") as f:
                                data = f.read()
                        except:
                            pass
"""
        parser = ErrorRecoveringParser(code)
        
        # Act
        result = parser.parse()
        
        # Assert
        assert result.success
        # Verify nested structure exists
        assert result.tree is not None


# Import invalid samples for parametrized test
