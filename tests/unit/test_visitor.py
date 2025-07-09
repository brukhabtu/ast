"""
Unit tests for AST visitor pattern implementation.

Tests the visitor functionality including context tracking,
traversal control, and node collection.
"""

import pytest
import ast
from astlib.visitor import (
    NodeVisitor, ASTVisitor, VisitorContext,
    find_nodes, find_functions, find_classes, find_imports, walk_tree
)


class TestNodeVisitor:
    """Test the enhanced NodeVisitor class."""
    
    def test_basic_traversal(self):
        """Test basic AST traversal with visitor."""
        # Arrange
        code = """
def hello():
    x = 1
    return x + 2
"""
        tree = ast.parse(code)
        
        class CountingVisitor(NodeVisitor):
            def __init__(self):
                super().__init__()
                self.node_count = 0
            
            def generic_visit(self, node):
                self.node_count += 1
                super().generic_visit(node)
        
        # Act
        visitor = CountingVisitor()
        visitor.visit(tree)
        
        # Assert
        assert visitor.node_count > 0
    
    def test_context_tracking(self):
        """Test that visitor tracks context correctly."""
        # Arrange
        code = """
class Outer:
    def method(self):
        return 42
"""
        tree = ast.parse(code)
        
        class ContextCheckVisitor(NodeVisitor):
            def __init__(self):
                super().__init__()
                self.contexts = []
            
            def visit_FunctionDef(self, node):
                self.contexts.append({
                    'name': node.name,
                    'depth': self.current_context.depth,
                    'parent_type': type(self.get_parent()).__name__
                })
                self.generic_visit(node)
        
        # Act
        visitor = ContextCheckVisitor()
        visitor.visit(tree)
        
        # Assert
        assert len(visitor.contexts) == 1
        context = visitor.contexts[0]
        assert context['name'] == 'method'
        assert context['depth'] == 2  # Module -> ClassDef -> FunctionDef
        assert context['parent_type'] == 'ClassDef'
    
    def test_skip_children(self):
        """Test skipping children during traversal."""
        # Arrange
        code = """
def outer():
    def inner():
        pass
"""
        tree = ast.parse(code)
        
        class SkippingVisitor(NodeVisitor):
            def __init__(self):
                super().__init__()
                self.visited_functions = []
            
            def visit_FunctionDef(self, node):
                self.visited_functions.append(node.name)
                if node.name == 'outer':
                    self.skip_children()
        
        # Act
        visitor = SkippingVisitor()
        visitor.visit(tree)
        
        # Assert
        assert visitor.visited_functions == ['outer']  # inner was skipped
    
    def test_get_parent_levels(self):
        """Test getting parent nodes at different levels."""
        # Arrange
        code = """
class A:
    class B:
        def c(self):
            pass
"""
        tree = ast.parse(code)
        
        class ParentTrackingVisitor(NodeVisitor):
            def __init__(self):
                super().__init__()
                self.function_parents = []
            
            def visit_FunctionDef(self, node):
                parents = []
                for i in range(1, 4):
                    parent = self.get_parent(i)
                    if parent:
                        parents.append(type(parent).__name__)
                self.function_parents = parents
                self.generic_visit(node)
        
        # Act
        visitor = ParentTrackingVisitor()
        visitor.visit(tree)
        
        # Assert - The visitor tracks ClassDef nodes as parents
        assert visitor.function_parents == ['ClassDef', 'ClassDef']


class TestASTVisitor:
    """Test the flexible ASTVisitor class."""
    
    def test_collect_nodes_by_type(self):
        """Test collecting nodes by type filter."""
        # Arrange
        code = """
x = 1
y = 2
def func():
    z = 3
"""
        tree = ast.parse(code)
        
        # Act
        visitor = ASTVisitor(
            collect_filter=lambda node: isinstance(node, ast.Assign)
        )
        assigns = visitor.visit(tree)
        
        # Assert
        assert len(assigns) == 3
    
    def test_transform_collected_nodes(self):
        """Test transforming nodes during collection."""
        # Arrange
        code = """
def func1():
    pass

class Class1:
    def method1(self):
        pass
"""
        tree = ast.parse(code)
        
        # Act
        visitor = ASTVisitor(
            collect_filter=lambda node: isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)),
            transform=lambda node: node.name
        )
        function_names = visitor.visit(tree)
        
        # Assert
        assert set(function_names) == {'func1', 'method1'}
    
    def test_breadth_first_traversal(self):
        """Test breadth-first traversal order."""
        # Arrange
        code = """
class A:
    def a1(self):
        pass
    def a2(self):
        pass

class B:
    def b1(self):
        pass
"""
        tree = ast.parse(code)
        
        # Act
        visitor = ASTVisitor(
            collect_filter=lambda node: isinstance(node, (ast.ClassDef, ast.FunctionDef)),
            transform=lambda node: node.name
        )
        bfs_order = visitor.visit_breadth_first(tree)
        
        # Assert
        # In BFS, we should see classes before their methods
        assert bfs_order[:2] == ['A', 'B']
        assert set(bfs_order[2:]) == {'a1', 'a2', 'b1'}
    
    def test_avoid_infinite_loops(self):
        """Test that visitor avoids infinite loops with circular references."""
        # This is more of a safety test - ASTs shouldn't have cycles normally
        # Arrange
        tree = ast.parse("x = 1")
        
        # Act
        visitor = ASTVisitor()
        nodes = visitor.visit(tree)
        
        # Assert - should complete without hanging
        assert len(nodes) > 0


class TestConvenienceFunctions:
    """Test convenience functions for common patterns."""
    
    def test_find_functions(self):
        """Test finding all function definitions."""
        # Arrange
        code = """
def regular():
    pass

async def async_func():
    pass

class C:
    def method(self):
        pass
    
    async def async_method(self):
        pass
"""
        tree = ast.parse(code)
        
        # Act
        functions = find_functions(tree)
        
        # Assert
        assert len(functions) == 4
        function_names = [f.name for f in functions]
        assert set(function_names) == {'regular', 'async_func', 'method', 'async_method'}
    
    def test_find_classes(self):
        """Test finding all class definitions."""
        # Arrange
        code = """
class A:
    pass

class B(A):
    class Nested:
        pass

def not_a_class():
    pass
"""
        tree = ast.parse(code)
        
        # Act
        classes = find_classes(tree)
        
        # Assert
        assert len(classes) == 3
        class_names = [c.name for c in classes]
        assert set(class_names) == {'A', 'B', 'Nested'}
    
    def test_find_imports(self):
        """Test finding all import statements."""
        # Arrange
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict

def func():
    import json  # nested import
    from collections import defaultdict
"""
        tree = ast.parse(code)
        
        # Act
        imports = find_imports(tree)
        
        # Assert
        assert len(imports) == 6
        
        # Verify different import types
        regular_imports = [i for i in imports if isinstance(i, ast.Import)]
        from_imports = [i for i in imports if isinstance(i, ast.ImportFrom)]
        assert len(regular_imports) == 3  # os, sys, json
        assert len(from_imports) == 3  # pathlib, typing, collections
    
    def test_walk_tree_callback(self):
        """Test walking tree with callback function."""
        # Arrange
        code = """
def outer():
    x = 1
    def inner():
        y = 2
"""
        tree = ast.parse(code)
        
        visited_nodes = []
        
        def callback(node, context):
            if isinstance(node, (ast.FunctionDef, ast.Assign)):
                visited_nodes.append({
                    'type': type(node).__name__,
                    'depth': context.depth
                })
        
        # Act
        walk_tree(tree, callback)
        
        # Assert
        assert len(visited_nodes) == 4  # outer, x=1, inner, y=2
        # Check depths
        depths = [n['depth'] for n in visited_nodes]
        assert depths[0] == 1  # outer function at module level
        assert depths[1] == 3  # x = 1 inside outer
        assert depths[2] == 3  # inner function inside outer
        assert depths[3] == 5  # y = 2 inside inner


class TestVisitorContext:
    """Test VisitorContext functionality."""
    
    def test_context_initialization(self):
        """Test context object initialization."""
        # Arrange & Act
        context = VisitorContext()
        
        # Assert
        assert context.parent is None
        assert context.depth == 0
        assert context.path == []
    
    def test_child_context_creation(self):
        """Test creating child contexts."""
        # Arrange
        parent_node = ast.parse("x = 1").body[0]
        parent_context = VisitorContext(parent=None, depth=0, path=[])
        
        # Act
        child_context = parent_context.child_context(parent_node)
        
        # Assert
        assert child_context.parent == parent_node
        assert child_context.depth == 1
        assert child_context.path == [parent_node]
    
    def test_context_path_tracking(self):
        """Test that context maintains correct path."""
        # Arrange
        tree = ast.parse("""
class A:
    def b(self):
        c = 1
""")
        
        paths_at_assign = []
        
        class PathTracker(NodeVisitor):
            def visit_Assign(self, node):
                path_types = [type(n).__name__ for n in self.current_context.path]
                paths_at_assign.append(path_types)
        
        # Act
        visitor = PathTracker()
        visitor.visit(tree)
        
        # Assert
        assert len(paths_at_assign) == 1
        assert paths_at_assign[0] == ['Module', 'ClassDef', 'FunctionDef', 'Assign']