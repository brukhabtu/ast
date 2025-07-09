"""
Reference finding functionality for the AST library.

This module provides tools to find all references (usages) of symbols
throughout a codebase, including function calls, class instantiations,
imports, and more.
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Set, Union
from pathlib import Path

from .symbols import Symbol, SymbolType
from .parser import parse_file


class ReferenceType(Enum):
    """Types of references to symbols."""
    FUNCTION_CALL = "function_call"
    CLASS_INSTANTIATION = "class_instantiation"
    INHERITANCE = "inheritance"
    TYPE_ANNOTATION = "type_annotation"
    IMPORT = "import"
    ATTRIBUTE_ACCESS = "attribute_access"
    ASSIGNMENT = "assignment"
    DECORATOR = "decorator"


@dataclass
class Reference:
    """Represents a reference to a symbol in the code."""
    symbol_name: str
    file_path: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    reference_type: ReferenceType = ReferenceType.FUNCTION_CALL
    context: str = ""  # Line of code containing the reference
    
    @property
    def location(self) -> str:
        """Get human-readable location string."""
        return f"{self.file_path}:{self.line}:{self.column}"


class ReferenceVisitor(ast.NodeVisitor):
    """AST visitor that finds references to symbols."""
    
    def __init__(self, file_path: str, source_lines: Optional[List[str]] = None):
        self.file_path = file_path
        self.source_lines = source_lines or []
        self.references: List[Reference] = []
        self.current_class: Optional[str] = None
        self._in_annotation = False
        self._in_decorator = False
        
    def _get_context(self, line_no: int) -> str:
        """Get the source line for context."""
        if 0 < line_no <= len(self.source_lines):
            return self.source_lines[line_no - 1].strip()
        return ""
        
    def _add_reference(self, name: str, node: ast.AST, ref_type: ReferenceType):
        """Add a reference to the list."""
        ref = Reference(
            symbol_name=name,
            file_path=self.file_path,
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, 'end_lineno', None),
            end_column=getattr(node, 'end_col_offset', None),
            reference_type=ref_type,
            context=self._get_context(node.lineno)
        )
        self.references.append(ref)
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls and class instantiations."""
        # Handle direct calls like func()
        if isinstance(node.func, ast.Name):
            # Determine if it's a class instantiation by checking naming convention
            # This is a heuristic - classes typically start with uppercase
            if node.func.id[0].isupper():
                ref_type = ReferenceType.CLASS_INSTANTIATION
            else:
                ref_type = ReferenceType.FUNCTION_CALL
            self._add_reference(node.func.id, node, ref_type)
            
        # Handle method calls like obj.method()
        elif isinstance(node.func, ast.Attribute):
            self._add_reference(node.func.attr, node, ReferenceType.FUNCTION_CALL)
            
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find inheritance."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Check base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                self._add_reference(base.id, base, ReferenceType.INHERITANCE)
            elif isinstance(base, ast.Attribute):
                # Handle module.Class inheritance
                self._add_reference(base.attr, base, ReferenceType.INHERITANCE)
                
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to find type annotations."""
        # Check return type annotation
        if node.returns:
            self._in_annotation = True
            self.visit(node.returns)
            self._in_annotation = False
            
        # Check parameter annotations
        for arg in node.args.args:
            if arg.annotation:
                self._in_annotation = True
                self.visit(arg.annotation)
                self._in_annotation = False
                
        # Check decorators
        for decorator in node.decorator_list:
            self._in_decorator = True
            self.visit(decorator)
            self._in_decorator = False
                
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments like x: Type = value."""
        if node.annotation:
            self._in_annotation = True
            self.visit(node.annotation)
            self._in_annotation = False
            
        if node.value:
            self.visit(node.value)
            
    def visit_Name(self, node: ast.Name) -> None:
        """Visit name nodes."""
        if self._in_annotation:
            self._add_reference(node.id, node, ReferenceType.TYPE_ANNOTATION)
        elif self._in_decorator:
            self._add_reference(node.id, node, ReferenceType.DECORATOR)
        # Don't add as reference if it's just a variable assignment target
        elif isinstance(node.ctx, ast.Load):
            # This catches variable usage but not in function calls
            # (those are handled by visit_Call)
            pass
            
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access like obj.attr."""
        if isinstance(node.ctx, ast.Load):
            # Always add attribute access references
            # The visit_Call will also add FUNCTION_CALL if it's part of a call
            self._add_reference(node.attr, node, ReferenceType.ATTRIBUTE_ACCESS)
            
        self.generic_visit(node)
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            name = alias.name.split('.')[-1]  # Get the last part for module.submodule
            self._add_reference(name, node, ReferenceType.IMPORT)
            
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from ... import ... statements."""
        for alias in node.names:
            self._add_reference(alias.name, node, ReferenceType.IMPORT)
            
    def _get_parent_node(self) -> Optional[ast.AST]:
        """Get parent node (simplified - would need proper implementation)."""
        return None  # TODO: Implement parent tracking
        

def find_references(file_path: str, target_symbols: Optional[Set[str]] = None) -> List[Reference]:
    """
    Find all references in a file, optionally filtered by target symbols.
    
    Args:
        file_path: Path to the Python file to analyze
        target_symbols: Optional set of symbol names to find references for.
                       If None, finds all references.
                       
    Returns:
        List of Reference objects found in the file
    """
    try:
        # Parse the file
        result = parse_file(file_path)
        if not result.tree:
            return []
    except FileNotFoundError:
        return []
        
    # Read source lines for context
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
    except:
        source_lines = []
        
    # Find references
    visitor = ReferenceVisitor(file_path, source_lines)
    visitor.visit(result.tree)
    
    # Filter by target symbols if specified
    if target_symbols:
        return [ref for ref in visitor.references if ref.symbol_name in target_symbols]
    
    return visitor.references


def find_references_in_directory(
    directory: str, 
    target_symbol: str,
    file_pattern: str = "*.py"
) -> List[Reference]:
    """
    Find all references to a symbol in a directory.
    
    Args:
        directory: Root directory to search
        target_symbol: Symbol name to find references for
        file_pattern: File pattern to match (default: "*.py")
        
    Returns:
        List of all references found
    """
    from .walker import DirectoryWalker
    
    references = []
    walker = DirectoryWalker()
    
    # DirectoryWalker.walk doesn't accept pattern, so filter manually
    for file_path in walker.walk(directory):
        if file_pattern == "*.py" or str(file_path).endswith(".py"):
            refs = find_references(str(file_path), {target_symbol})
            references.extend(refs)
        
    return references