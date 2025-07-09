#!/usr/bin/env python3
"""
Example usage of the AST library with error recovery.
"""

from pathlib import Path
from ast import parse_file
from ast.visitor import find_functions, find_classes, walk_tree


def main():
    # Example 1: Parse a valid Python file
    print("Example 1: Parsing valid Python code")
    print("-" * 40)
    
    valid_code = '''
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

class Calculator:
    def add(self, x: int, y: int) -> int:
        return x + y
    
    def multiply(self, x: int, y: int) -> int:
        return x * y
'''
    
    # Create a temporary file
    test_file = Path("test_valid.py")
    test_file.write_text(valid_code)
    
    try:
        result = parse_file(test_file)
        print(f"Parse success: {result.success}")
        print(f"Functions found: {[f.name for f in find_functions(result.tree)]}")
        print(f"Classes found: {[c.name for c in find_classes(result.tree)]}")
    finally:
        test_file.unlink()
    
    print()
    
    # Example 2: Parse Python code with syntax errors
    print("Example 2: Parsing code with syntax errors (error recovery)")
    print("-" * 40)
    
    invalid_code = '''
def valid_function():
    return "This parses fine"

def broken_function(  # Missing closing parenthesis
    print("This has syntax error")

def another_valid():
    return "This should still parse"
'''
    
    test_file = Path("test_invalid.py")
    test_file.write_text(invalid_code)
    
    try:
        result = parse_file(test_file)
        print(f"Parse success: {result.success}")
        print(f"Partial success: {result.partial_success}")
        print(f"Errors found: {len(result.errors)}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  Line {error.line}: {error.message}")
        
        if result.tree:
            functions = find_functions(result.tree)
            print(f"\nFunctions recovered: {[f.name for f in functions]}")
    finally:
        test_file.unlink()
    
    print()
    
    # Example 3: Using the visitor pattern
    print("Example 3: Custom visitor to analyze code")
    print("-" * 40)
    
    analysis_code = '''
import os
from pathlib import Path

class FileHandler:
    def __init__(self):
        self.files = []
    
    def process_file(self, path: Path):
        if path.exists():
            self.files.append(path)
    
    def count_lines(self):
        total = 0
        for file in self.files:
            total += len(file.read_text().splitlines())
        return total
'''
    
    test_file = Path("test_analysis.py")
    test_file.write_text(analysis_code)
    
    try:
        result = parse_file(test_file)
        
        # Count different node types
        node_counts = {}
        
        def count_nodes(node, context):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        walk_tree(result.tree, count_nodes)
        
        print("Node type counts:")
        for node_type, count in sorted(node_counts.items()):
            if count > 1:  # Only show types that appear multiple times
                print(f"  {node_type}: {count}")
    finally:
        test_file.unlink()


if __name__ == "__main__":
    main()