"""
Symbol extraction from Python AST.

This module provides functionality to extract symbols (functions, classes, variables)
from Python Abstract Syntax Trees for fast navigation and lookup.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Union
from enum import Enum


class SymbolType(Enum):
    """Types of symbols that can be extracted."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    ASYNC_FUNCTION = "async_function"
    PROPERTY = "property"


@dataclass
class Position:
    """Position in source code."""
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class Symbol:
    """Represents a symbol in the code."""
    name: str
    type: SymbolType
    position: Position
    file_path: Optional[str] = None
    parent: Optional['Symbol'] = None
    children: List['Symbol'] = field(default_factory=list)
    
    # Additional metadata
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  # For classes
    is_private: bool = False
    is_async: bool = False
    docstring: Optional[str] = None
    type_annotation: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed properties."""
        self.is_private = self.name.startswith('_')
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including parent symbols."""
        if self.parent:
            return f"{self.parent.qualified_name}.{self.name}"
        return self.name
    
    def __hash__(self) -> int:
        """Make Symbol hashable based on qualified name and file path."""
        return hash((self.qualified_name, self.file_path, self.type.value))
    
    def __eq__(self, other) -> bool:
        """Compare symbols based on qualified name, file path, and type."""
        if not isinstance(other, Symbol):
            return False
        return (self.qualified_name == other.qualified_name and 
                self.file_path == other.file_path and
                self.type == other.type)


class SymbolExtractor(ast.NodeVisitor):
    """Extracts symbols from an AST."""
    
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self.current_scope: Optional[Symbol] = None
        self._symbol_stack: List[Symbol] = []
        
    def extract_symbols(self, tree: ast.AST) -> List[Symbol]:
        """Extract all symbols from an AST tree."""
        self.visit(tree)
        return self.symbols
    
    def _enter_scope(self, symbol: Symbol):
        """Enter a new scope (class or function)."""
        if self.current_scope:
            symbol.parent = self.current_scope
            self.current_scope.children.append(symbol)
        self._symbol_stack.append(symbol)
        self.current_scope = symbol
        self.symbols.append(symbol)
        
    def _exit_scope(self):
        """Exit the current scope."""
        if self._symbol_stack:
            self._symbol_stack.pop()
            self.current_scope = self._symbol_stack[-1] if self._symbol_stack else None
    
    def _get_position(self, node: ast.AST) -> Position:
        """Extract position information from AST node."""
        return Position(
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, 'end_lineno', None),
            end_column=getattr(node, 'end_col_offset', None)
        )
    
    def _get_decorators(self, node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract decorator names from a node."""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{self._get_attribute_name(dec)}")
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attribute_name(dec.func))
        return decorators
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full attribute name (e.g., 'obj.attr')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _get_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Extract function signature as a string."""
        args = []
        
        # Positional arguments
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            # Add default value if exists
            default_offset = len(node.args.args) - len(node.args.defaults)
            if i >= default_offset:
                default_idx = i - default_offset
                if default_idx < len(node.args.defaults):
                    arg_str += f" = {ast.unparse(node.args.defaults[default_idx])}"
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)
        
        # Keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            if i < len(node.args.kw_defaults) and node.args.kw_defaults[i]:
                arg_str += f" = {ast.unparse(node.args.kw_defaults[i])}"
            args.append(arg_str)
        
        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)
        
        signature = f"({', '.join(args)})"
        
        # Return type annotation
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
            
        return signature
    
    def _get_docstring(self, node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module]) -> Optional[str]:
        """Extract docstring from a node."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                return node.body[0].value.value
        return None
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))
        
        symbol = Symbol(
            name=node.name,
            type=SymbolType.CLASS,
            position=self._get_position(node),
            file_path=self.file_path,
            decorators=self._get_decorators(node),
            bases=bases,
            docstring=self._get_docstring(node)
        )
        
        self._enter_scope(symbol)
        self.generic_visit(node)
        self._exit_scope()
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        decorators = self._get_decorators(node)
        
        # Check if it's a property
        symbol_type = SymbolType.PROPERTY if 'property' in decorators else SymbolType.FUNCTION
        
        # If inside a class, it's a method
        if self.current_scope and self.current_scope.type == SymbolType.CLASS:
            if symbol_type != SymbolType.PROPERTY:
                symbol_type = SymbolType.METHOD
        
        symbol = Symbol(
            name=node.name,
            type=symbol_type,
            position=self._get_position(node),
            file_path=self.file_path,
            signature=self._get_function_signature(node),
            decorators=decorators,
            docstring=self._get_docstring(node),
            is_async=False
        )
        
        self._enter_scope(symbol)
        self.generic_visit(node)
        self._exit_scope()
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        decorators = self._get_decorators(node)
        
        # Async functions are either async functions or async methods
        symbol_type = SymbolType.ASYNC_FUNCTION
        if self.current_scope and self.current_scope.type == SymbolType.CLASS:
            symbol_type = SymbolType.METHOD
        
        symbol = Symbol(
            name=node.name,
            type=symbol_type,
            position=self._get_position(node),
            file_path=self.file_path,
            signature=self._get_function_signature(node),
            decorators=decorators,
            docstring=self._get_docstring(node),
            is_async=True
        )
        
        self._enter_scope(symbol)
        self.generic_visit(node)
        self._exit_scope()
    
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment to extract variables/constants."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple variable assignment
                symbol_type = SymbolType.CONSTANT if target.id.isupper() else SymbolType.VARIABLE
                
                # Get type annotation if available
                type_annotation = None
                if isinstance(node, ast.AnnAssign) and node.annotation:
                    type_annotation = ast.unparse(node.annotation)
                
                symbol = Symbol(
                    name=target.id,
                    type=symbol_type,
                    position=self._get_position(target),
                    file_path=self.file_path,
                    parent=self.current_scope,
                    type_annotation=type_annotation
                )
                
                if self.current_scope:
                    self.current_scope.children.append(symbol)
                self.symbols.append(symbol)
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment."""
        if isinstance(node.target, ast.Name):
            symbol_type = SymbolType.CONSTANT if node.target.id.isupper() else SymbolType.VARIABLE
            
            type_annotation = ast.unparse(node.annotation) if node.annotation else None
            
            symbol = Symbol(
                name=node.target.id,
                type=symbol_type,
                position=self._get_position(node.target),
                file_path=self.file_path,
                parent=self.current_scope,
                type_annotation=type_annotation
            )
            
            if self.current_scope:
                self.current_scope.children.append(symbol)
            self.symbols.append(symbol)
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            symbol = Symbol(
                name=name,
                type=SymbolType.IMPORT,
                position=self._get_position(node),
                file_path=self.file_path,
                parent=self.current_scope
            )
            
            if self.current_scope:
                self.current_scope.children.append(symbol)
            self.symbols.append(symbol)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from-import statement."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            symbol = Symbol(
                name=name,
                type=SymbolType.IMPORT,
                position=self._get_position(node),
                file_path=self.file_path,
                parent=self.current_scope
            )
            
            if self.current_scope:
                self.current_scope.children.append(symbol)
            self.symbols.append(symbol)
        
        self.generic_visit(node)


def extract_symbols(ast_tree: ast.AST, file_path: Optional[str] = None) -> List[Symbol]:
    """
    Extract symbols from an AST tree.
    
    Args:
        ast_tree: The AST tree to extract symbols from
        file_path: Optional file path for the source
        
    Returns:
        List of Symbol objects found in the AST
    """
    # Try to get from cache first
    if file_path:
        from pathlib import Path
        from .cache import get_cache_manager
        manager = get_cache_manager()
        file_path_obj = Path(file_path)
        
        cached_symbols = manager.get_cached_symbols(file_path_obj, ast_tree)
        if cached_symbols is not None:
            return cached_symbols
    
    # Extract symbols
    extractor = SymbolExtractor(file_path)
    symbols = extractor.extract_symbols(ast_tree)
    
    # Cache the result
    if file_path:
        manager.cache_symbols(file_path_obj, ast_tree, symbols)
    
    return symbols