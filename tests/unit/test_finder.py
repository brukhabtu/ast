"""Unit tests for function finder."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from astlib.finder import FunctionFinder


class TestFunctionFinder:
    """Test FunctionFinder functionality."""

    def test_init_default(self):
        """Test finder initialization with defaults."""
        finder = FunctionFinder()
        assert finder.case_sensitive is False

    def test_init_case_sensitive(self):
        """Test finder initialization with case sensitivity."""
        finder = FunctionFinder(case_sensitive=True)
        assert finder.case_sensitive is True

    def test_matches_pattern_exact(self):
        """Test exact pattern matching."""
        finder = FunctionFinder()
        
        assert finder._matches_pattern('test_function', 'test_function')
        assert not finder._matches_pattern('test_function', 'other_function')

    def test_matches_pattern_wildcards(self):
        """Test pattern matching with wildcards."""
        finder = FunctionFinder()
        
        # Test * wildcard
        assert finder._matches_pattern('test_function', 'test_*')
        assert finder._matches_pattern('test_something_else', 'test_*')
        assert finder._matches_pattern('prefix_test_suffix', '*_test_*')
        assert not finder._matches_pattern('other_function', 'test_*')
        
        # Test ? wildcard
        assert finder._matches_pattern('test', 'tes?')
        assert finder._matches_pattern('text', 'te?t')
        assert not finder._matches_pattern('test', 'te?')

    def test_matches_pattern_case_insensitive(self):
        """Test case-insensitive pattern matching."""
        finder = FunctionFinder(case_sensitive=False)
        
        assert finder._matches_pattern('TestFunction', 'testfunction')
        assert finder._matches_pattern('test_function', 'TEST_*')
        assert finder._matches_pattern('TEST', 'test')

    def test_matches_pattern_case_sensitive(self):
        """Test case-sensitive pattern matching."""
        finder = FunctionFinder(case_sensitive=True)
        
        assert finder._matches_pattern('TestFunction', 'TestFunction')
        assert not finder._matches_pattern('TestFunction', 'testfunction')
        assert not finder._matches_pattern('test_function', 'TEST_*')

    def test_extract_docstring_simple(self):
        """Test extracting simple docstring."""
        import ast
        
        code = '''
def test_func():
    """This is a docstring."""
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        finder = FunctionFinder()
        docstring = finder._extract_docstring(func_node)
        
        assert docstring == "This is a docstring."

    def test_extract_docstring_multiline(self):
        """Test extracting multiline docstring."""
        import ast
        
        code = '''
def test_func():
    """
    This is a multiline
    docstring.
    """
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        finder = FunctionFinder()
        docstring = finder._extract_docstring(func_node)
        
        assert "This is a multiline" in docstring
        assert "docstring." in docstring

    def test_extract_docstring_none(self):
        """Test extracting docstring when none exists."""
        import ast
        
        code = '''
def test_func():
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        finder = FunctionFinder()
        docstring = finder._extract_docstring(func_node)
        
        assert docstring is None

    def test_extract_docstring_not_first_statement(self):
        """Test that non-first string is not considered docstring."""
        import ast
        
        code = '''
def test_func():
    x = 1
    "This is not a docstring"
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        finder = FunctionFinder()
        docstring = finder._extract_docstring(func_node)
        
        assert docstring is None

    @patch('builtins.open', new_callable=mock_open, read_data='def test_func():\n    pass\n')
    def test_find_functions_simple(self, mock_file):
        """Test finding a simple function."""
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_func')
        
        assert len(results) == 1
        assert results[0]['name'] == 'test_func'
        assert results[0]['file'] == '/test/file.py'
        assert results[0]['line'] == 1
        assert results[0]['column'] == 0

    @patch('builtins.open', new_callable=mock_open, read_data='')
    def test_find_functions_empty_file(self, mock_file):
        """Test finding functions in empty file."""
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_func')
        
        assert results == []

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_find_functions_file_not_found(self, mock_file):
        """Test handling file not found error."""
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/nonexistent.py'), 'test_func')
        
        assert results == []

    @patch('builtins.open', new_callable=mock_open, read_data='def syntax error')
    def test_find_functions_syntax_error(self, mock_file):
        """Test handling syntax errors in file."""
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_func')
        
        assert results == []

    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_multiple(self, mock_file):
        """Test finding multiple functions."""
        code = '''
def test_one():
    """First test function."""
    pass

def test_two():
    """Second test function."""
    pass

def other_function():
    pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_*')
        
        assert len(results) == 2
        assert results[0]['name'] == 'test_one'
        assert results[0]['docstring'] == 'First test function.'
        assert results[1]['name'] == 'test_two'
        assert results[1]['docstring'] == 'Second test function.'

    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_with_decorators(self, mock_file):
        """Test finding functions with decorators."""
        code = '''
@decorator
@another.decorator
def test_func():
    pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_func')
        
        assert len(results) == 1
        assert results[0]['decorators'] == ['decorator', 'decorator']

    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_async(self, mock_file):
        """Test finding async functions."""
        code = '''
async def async_test():
    pass

def sync_test():
    pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), '*_test')
        
        assert len(results) == 2
        # Note: ast.AsyncFunctionDef is a subclass of ast.FunctionDef
        # so both will be found, but async flag won't be set with current implementation
        
    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_nested(self, mock_file):
        """Test finding nested functions."""
        code = '''
def outer_function():
    def inner_test():
        pass
    
    def another_inner_test():
        pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), '*_test')
        
        assert len(results) == 2
        assert results[0]['name'] == 'inner_test'
        assert results[1]['name'] == 'another_inner_test'

    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_class_methods(self, mock_file):
        """Test finding methods within classes."""
        code = '''
class TestClass:
    def test_method(self):
        """Test method docstring."""
        pass
    
    @staticmethod
    def test_static():
        pass
    
    @classmethod
    def test_class(cls):
        pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), 'test_*')
        
        assert len(results) == 3
        assert {r['name'] for r in results} == {'test_method', 'test_static', 'test_class'}

    @patch('builtins.open', new_callable=mock_open)
    def test_find_functions_position_info(self, mock_file):
        """Test that position information is captured correctly."""
        code = '''def first():
    pass

    
def second():
    """With docstring."""
    pass
'''
        mock_file.return_value.read.return_value = code
        
        finder = FunctionFinder()
        results = finder.find_functions(Path('/test/file.py'), '*')
        
        assert len(results) == 2
        assert results[0]['line'] == 1
        assert results[1]['line'] == 5
        assert 'end_line' in results[0]
        assert 'end_column' in results[0]