"""
AST-specific test helper utilities.
"""

import ast
from typing import Any, Type, TypeVar, cast

T = TypeVar("T", bound=ast.AST)


def assert_ast_equal(node1: ast.AST, node2: ast.AST, ignore_ctx: bool = True) -> None:
    """
    Assert that two AST nodes are structurally equal.
    
    Args:
        node1: First AST node
        node2: Second AST node
        ignore_ctx: Whether to ignore Load/Store/Del context differences
    
    Raises:
        AssertionError: If nodes are not equal
    """
    def normalize(node: ast.AST) -> ast.AST:
        """Normalize an AST node for comparison."""
        if ignore_ctx and isinstance(node, ast.Name):
            # Create a copy without context for comparison
            return ast.Name(id=node.id, ctx=ast.Load())
        return node
    
    # For simple comparison, we can use ast.dump
    dump1 = ast.dump(normalize(node1), annotate_fields=True)
    dump2 = ast.dump(normalize(node2), annotate_fields=True)
    
    if dump1 != dump2:
        raise AssertionError(
            f"AST nodes are not equal:\n"
            f"Node 1: {dump1}\n"
            f"Node 2: {dump2}"
        )


def assert_ast_structure(
    node: ast.AST,
    expected_type: Type[T],
    **expected_fields: Any
) -> T:
    """
    Assert that an AST node has the expected type and field values.
    
    Args:
        node: AST node to check
        expected_type: Expected node type
        **expected_fields: Expected field values
    
    Returns:
        The node cast to the expected type
    
    Raises:
        AssertionError: If node doesn't match expectations
    """
    if not isinstance(node, expected_type):
        raise AssertionError(
            f"Expected node type {expected_type.__name__}, "
            f"got {type(node).__name__}"
        )
    
    for field, expected_value in expected_fields.items():
        if not hasattr(node, field):
            raise AssertionError(f"Node missing expected field: {field}")
        
        actual_value = getattr(node, field)
        if actual_value != expected_value:
            raise AssertionError(
                f"Field {field}: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
    
    return cast(T, node)


def find_nodes(tree: ast.AST, node_type: Type[T]) -> list[T]:
    """
    Find all nodes of a specific type in an AST tree.
    
    Args:
        tree: AST tree to search
        node_type: Type of nodes to find
    
    Returns:
        List of matching nodes
    """
    nodes: list[T] = []
    
    class Visitor(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            if isinstance(node, node_type):
                nodes.append(node)
            self.generic_visit(node)
    
    Visitor().visit(tree)
    return nodes


def count_nodes(tree: ast.AST, node_type: Type[ast.AST] | None = None) -> int:
    """
    Count nodes in an AST tree.
    
    Args:
        tree: AST tree to count nodes in
        node_type: Optional specific type to count (None counts all)
    
    Returns:
        Number of matching nodes
    """
    count = 0
    
    class Counter(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            nonlocal count
            if node_type is None or isinstance(node, node_type):
                count += 1
            self.generic_visit(node)
    
    Counter().visit(tree)
    return count


def get_node_types(tree: ast.AST) -> set[Type[ast.AST]]:
    """
    Get all unique node types present in an AST tree.
    
    Args:
        tree: AST tree to analyze
    
    Returns:
        Set of node types found
    """
    types: set[Type[ast.AST]] = set()
    
    class TypeCollector(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            types.add(type(node))
            self.generic_visit(node)
    
    TypeCollector().visit(tree)
    return types


def normalize_ast(tree: ast.AST) -> ast.AST:
    """
    Normalize an AST tree for comparison by removing location info.
    
    Args:
        tree: AST tree to normalize
    
    Returns:
        Normalized AST tree
    """
    class Normalizer(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            # Remove location information
            for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                if hasattr(node, attr):
                    delattr(node, attr)
            
            # Normalize type comment/type ignore
            if hasattr(node, 'type_comment'):
                node.type_comment = None
            
            return self.generic_visit(node)
    
    return Normalizer().visit(tree)


def ast_to_dict(node: ast.AST) -> dict[str, Any]:
    """
    Convert an AST node to a dictionary representation.
    
    Args:
        node: AST node to convert
    
    Returns:
        Dictionary representation
    """
    result: dict[str, Any] = {"_type": type(node).__name__}
    
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            result[field] = [
                ast_to_dict(item) if isinstance(item, ast.AST) else item
                for item in value
            ]
        elif isinstance(value, ast.AST):
            result[field] = ast_to_dict(value)
        else:
            result[field] = value
    
    return result


def compare_ast_structure(tree1: ast.AST, tree2: ast.AST) -> list[str]:
    """
    Compare two AST trees and return differences.
    
    Args:
        tree1: First AST tree
        tree2: Second AST tree
    
    Returns:
        List of difference descriptions
    """
    differences: list[str] = []
    
    def compare_nodes(
        node1: ast.AST | None,
        node2: ast.AST | None,
        path: str = "root"
    ) -> None:
        if node1 is None and node2 is None:
            return
        
        if node1 is None or node2 is None:
            differences.append(
                f"{path}: One node is None (node1={node1}, node2={node2})"
            )
            return
        
        if type(node1) != type(node2):
            differences.append(
                f"{path}: Different types ({type(node1).__name__} vs "
                f"{type(node2).__name__})"
            )
            return
        
        # Compare fields
        fields1 = set(field for field, _ in ast.iter_fields(node1))
        fields2 = set(field for field, _ in ast.iter_fields(node2))
        
        if fields1 != fields2:
            differences.append(
                f"{path}: Different fields ({fields1} vs {fields2})"
            )
        
        # Compare common fields
        for field in fields1 & fields2:
            value1 = getattr(node1, field)
            value2 = getattr(node2, field)
            
            if isinstance(value1, list) and isinstance(value2, list):
                if len(value1) != len(value2):
                    differences.append(
                        f"{path}.{field}: Different list lengths "
                        f"({len(value1)} vs {len(value2)})"
                    )
                else:
                    for i, (item1, item2) in enumerate(zip(value1, value2)):
                        if isinstance(item1, ast.AST):
                            compare_nodes(item1, item2, f"{path}.{field}[{i}]")
            elif isinstance(value1, ast.AST):
                compare_nodes(value1, value2, f"{path}.{field}")
            elif value1 != value2:
                differences.append(
                    f"{path}.{field}: Different values ({value1!r} vs {value2!r})"
                )
    
    compare_nodes(tree1, tree2)
    return differences