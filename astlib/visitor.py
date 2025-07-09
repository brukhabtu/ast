"""
AST visitor pattern implementation for tree traversal.

Provides both a basic NodeVisitor and an enhanced ASTVisitor with
additional traversal capabilities.
"""

import ast
from typing import Any, Optional, Callable, List, TypeVar, Generic
from collections import deque
from dataclasses import dataclass, field


T = TypeVar('T')


@dataclass
class VisitorContext:
    """Context information maintained during traversal."""
    parent: Optional[ast.AST] = None
    depth: int = 0
    path: List[ast.AST] = field(default_factory=list)
    
    def child_context(self, node: ast.AST) -> 'VisitorContext':
        """Create context for a child node."""
        return VisitorContext(
            parent=node,
            depth=self.depth + 1,
            path=self.path + [node]
        )


class NodeVisitor(ast.NodeVisitor):
    """
    Enhanced node visitor with context tracking.
    
    Extends ast.NodeVisitor with additional features like parent tracking,
    depth information, and traversal control.
    """
    
    def __init__(self):
        self._context_stack: List[VisitorContext] = []
        self._skip_children = False
    
    @property
    def current_context(self) -> Optional[VisitorContext]:
        """Get current traversal context."""
        return self._context_stack[-1] if self._context_stack else None
    
    def skip_children(self) -> None:
        """Skip visiting children of current node."""
        self._skip_children = True
    
    def visit(self, node: ast.AST) -> Any:
        """Visit a node with context tracking."""
        # Create context
        if self._context_stack:
            context = self._context_stack[-1].child_context(node)
        else:
            context = VisitorContext()
        
        self._context_stack.append(context)
        self._skip_children = False
        
        try:
            # Call node-specific visitor
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            result = visitor(node)
            
            # Visit children unless skipped
            if not self._skip_children:
                self.generic_visit(node)
            
            return result
        finally:
            self._context_stack.pop()
    
    def get_parent(self, levels: int = 1) -> Optional[ast.AST]:
        """Get parent node at specified level."""
        if len(self._context_stack) > levels:
            return self._context_stack[-(levels + 1)].parent
        return None


class ASTVisitor(Generic[T]):
    """
    Flexible AST visitor with collection and filtering capabilities.
    
    This visitor can collect nodes matching certain criteria and supports
    different traversal strategies.
    """
    
    def __init__(
        self,
        collect_filter: Optional[Callable[[ast.AST], bool]] = None,
        transform: Optional[Callable[[ast.AST], T]] = None
    ):
        self.collect_filter = collect_filter
        self.transform = transform or (lambda x: x)
        self.collected: List[T] = []
        self._visited_nodes = set()
    
    def visit(self, node: ast.AST) -> List[T]:
        """Visit tree and return collected nodes."""
        self.collected = []
        self._visited_nodes = set()
        self._visit_recursive(node)
        return self.collected
    
    def _visit_recursive(self, node: ast.AST) -> None:
        """Recursively visit nodes."""
        # Avoid infinite loops
        if id(node) in self._visited_nodes:
            return
        self._visited_nodes.add(id(node))
        
        # Check if we should collect this node
        if self.collect_filter is None or self.collect_filter(node):
            self.collected.append(self.transform(node))
        
        # Visit children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._visit_recursive(item)
            elif isinstance(value, ast.AST):
                self._visit_recursive(value)
    
    def visit_breadth_first(self, node: ast.AST) -> List[T]:
        """Visit tree in breadth-first order."""
        self.collected = []
        self._visited_nodes = set()
        
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            
            if id(current) in self._visited_nodes:
                continue
            self._visited_nodes.add(id(current))
            
            # Check if we should collect this node
            if self.collect_filter is None or self.collect_filter(current):
                self.collected.append(self.transform(current))
            
            # Add children to queue
            for field, value in ast.iter_fields(current):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            queue.append(item)
                elif isinstance(value, ast.AST):
                    queue.append(value)
        
        return self.collected


# Convenience functions for common visiting patterns

def find_nodes(tree: ast.AST, node_type: type) -> List[ast.AST]:
    """Find all nodes of a specific type."""
    visitor = ASTVisitor(
        collect_filter=lambda node: isinstance(node, node_type)
    )
    return visitor.visit(tree)


def find_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    """Find all function definitions."""
    return find_nodes(tree, (ast.FunctionDef, ast.AsyncFunctionDef))


def find_classes(tree: ast.AST) -> List[ast.ClassDef]:
    """Find all class definitions."""
    return find_nodes(tree, ast.ClassDef)


def find_imports(tree: ast.AST) -> List[ast.AST]:
    """Find all import statements."""
    return find_nodes(tree, (ast.Import, ast.ImportFrom))


def walk_tree(tree: ast.AST, callback: Callable[[ast.AST, VisitorContext], None]) -> None:
    """Walk tree with a callback for each node."""
    class CallbackVisitor(NodeVisitor):
        def generic_visit(self, node):
            callback(node, self.current_context)
            super().generic_visit(node)
    
    visitor = CallbackVisitor()
    visitor.visit(tree)