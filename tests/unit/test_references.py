"""Unit tests for reference finding functionality."""

import pytest
import tempfile
import os
from pathlib import Path

from astlib.references import (
    find_references, 
    find_references_in_directory,
    Reference,
    ReferenceType,
    ReferenceVisitor
)


class TestReferenceVisitor:
    """Test the ReferenceVisitor class."""
    
    def test_find_function_calls(self, tmp_path):
        """Test finding function call references."""
        code = '''
def greet(name):
    print(f"Hello {name}")

greet("Alice")
result = greet("Bob")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"greet"})
        
        assert len(refs) == 2
        assert all(ref.symbol_name == "greet" for ref in refs)
        assert refs[0].line == 5
        assert refs[1].line == 6
        assert all(ref.reference_type == ReferenceType.FUNCTION_CALL for ref in refs)
    
    def test_find_class_instantiation(self, tmp_path):
        """Test finding class instantiation references."""
        code = '''
class Person:
    def __init__(self, name):
        self.name = name

alice = Person("Alice")
bob = Person("Bob")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"Person"})
        
        assert len(refs) == 2
        assert all(ref.symbol_name == "Person" for ref in refs)
        assert refs[0].line == 6
        assert refs[1].line == 7
        assert all(ref.reference_type == ReferenceType.CLASS_INSTANTIATION for ref in refs)
    
    def test_find_inheritance(self, tmp_path):
        """Test finding class inheritance references."""
        code = '''
class Animal:
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"Animal"})
        
        assert len(refs) == 2
        assert all(ref.symbol_name == "Animal" for ref in refs)
        assert refs[0].line == 5
        assert refs[1].line == 8
        assert all(ref.reference_type == ReferenceType.INHERITANCE for ref in refs)
    
    def test_find_type_annotations(self, tmp_path):
        """Test finding type annotation references."""
        code = '''
from typing import List

def process(items: List[str]) -> str:
    return ", ".join(items)

class Handler:
    def handle(self, data: str) -> None:
        pass

value: str = "hello"
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"str"})
        
        # Should find str in type annotations
        assert len(refs) >= 3  # Return type, parameter type, variable annotation
        assert all(ref.symbol_name == "str" for ref in refs)
        
    def test_find_imports(self, tmp_path):
        """Test finding import references."""
        code = '''
import os
from pathlib import Path
from typing import List, Dict

path = Path(".")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"Path"})
        
        assert len(refs) == 2  # Import and usage
        assert refs[0].line == 3  # Import line
        assert refs[0].reference_type == ReferenceType.IMPORT
        assert refs[1].line == 6  # Usage line
        assert refs[1].reference_type == ReferenceType.CLASS_INSTANTIATION
    
    def test_find_decorators(self, tmp_path):
        """Test finding decorator references."""
        code = '''
import functools

@functools.cache
def expensive_function(n):
    return n * n

@property
def value(self):
    return self._value
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"property"})
        
        assert len(refs) == 1
        assert refs[0].line == 8
        assert refs[0].reference_type == ReferenceType.DECORATOR
    
    def test_method_calls(self, tmp_path):
        """Test finding method call references."""
        code = '''
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
result = calc.add(1, 2)
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"add"})
        
        # Should find both attribute access and function call
        assert len(refs) == 2
        assert all(ref.line == 7 for ref in refs)
        
        # Check we have both types
        ref_types = {ref.reference_type for ref in refs}
        assert ReferenceType.FUNCTION_CALL in ref_types
        assert ReferenceType.ATTRIBUTE_ACCESS in ref_types
    
    def test_context_preservation(self, tmp_path):
        """Test that context lines are preserved."""
        code = '''
def main():
    # This is a comment
    result = process_data(input_value)
    return result
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"process_data"})
        
        assert len(refs) == 1
        assert "process_data(input_value)" in refs[0].context
    
    def test_no_references_found(self, tmp_path):
        """Test when no references are found."""
        code = '''
def hello():
    print("Hello world")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"NonExistentSymbol"})
        
        assert len(refs) == 0
    
    def test_multiple_symbols_filter(self, tmp_path):
        """Test filtering by multiple target symbols."""
        code = '''
def foo():
    pass

def bar():
    pass

foo()
bar()
baz()
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"foo", "bar"})
        
        assert len(refs) == 2
        assert {ref.symbol_name for ref in refs} == {"foo", "bar"}
        assert "baz" not in [ref.symbol_name for ref in refs]
    
    def test_find_all_references(self, tmp_path):
        """Test finding all references without filtering."""
        code = '''
import os

def process(data):
    return transform(data)

result = process(input_data)
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        # Find all references (no filter)
        refs = find_references(str(file_path))
        
        # Should find multiple references
        assert len(refs) > 0
        symbol_names = {ref.symbol_name for ref in refs}
        assert "os" in symbol_names  # Import
        assert "transform" in symbol_names  # Function call
        assert "process" in symbol_names  # Function call


class TestFindReferencesInDirectory:
    """Test directory-wide reference finding."""
    
    def test_find_across_multiple_files(self, tmp_path):
        """Test finding references across multiple files."""
        # Create file 1
        file1 = tmp_path / "module1.py"
        file1.write_text('''
def shared_function():
    pass

shared_function()
''')
        
        # Create file 2
        file2 = tmp_path / "module2.py"
        file2.write_text('''
from module1 import shared_function

result = shared_function()
''')
        
        refs = find_references_in_directory(str(tmp_path), "shared_function")
        
        assert len(refs) == 3  # 1 call in module1, 1 import + 1 call in module2
        
        # Check that references come from both files
        file_paths = {ref.file_path for ref in refs}
        assert len(file_paths) == 2
    
    def test_respects_file_pattern(self, tmp_path):
        """Test that file pattern filtering works."""
        # Create Python file
        py_file = tmp_path / "code.py"
        py_file.write_text('MyClass()')
        
        # Create non-Python file
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text('MyClass is used here')
        
        refs = find_references_in_directory(str(tmp_path), "MyClass", "*.py")
        
        assert len(refs) == 1
        assert refs[0].file_path == str(py_file)


class TestReferenceDataclass:
    """Test the Reference dataclass."""
    
    def test_location_property(self):
        """Test the location property formatting."""
        ref = Reference(
            symbol_name="test",
            file_path="/path/to/file.py",
            line=10,
            column=5
        )
        
        assert ref.location == "/path/to/file.py:10:5"
    
    def test_default_values(self):
        """Test default values for optional fields."""
        ref = Reference(
            symbol_name="test",
            file_path="/path/to/file.py",
            line=10,
            column=5
        )
        
        assert ref.reference_type == ReferenceType.FUNCTION_CALL
        assert ref.context == ""
        assert ref.end_line is None
        assert ref.end_column is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_python_file(self, tmp_path):
        """Test handling of files with syntax errors."""
        code = '''
def broken(
    # Missing closing parenthesis
'''
        file_path = tmp_path / "broken.py"
        file_path.write_text(code)
        
        refs = find_references(str(file_path), {"broken"})
        
        # Should return empty list for unparseable files
        assert len(refs) == 0
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        refs = find_references("/path/that/does/not/exist.py", {"test"})
        
        assert len(refs) == 0
    
    def test_unicode_identifiers(self, tmp_path):
        """Test handling of Unicode identifiers."""
        code = '''
def 你好():
    pass

你好()
'''
        file_path = tmp_path / "unicode.py"
        file_path.write_text(code, encoding='utf-8')
        
        refs = find_references(str(file_path), {"你好"})
        
        assert len(refs) == 1
        assert refs[0].symbol_name == "你好"