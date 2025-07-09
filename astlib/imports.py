#!/usr/bin/env python3
"""Import analysis functionality for AST library.

This module provides tools to:
- Extract all import statements from Python code
- Resolve relative imports to absolute paths
- Build import dependency graphs
- Detect circular imports
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class ImportType(Enum):
    """Type of import statement."""
    ABSOLUTE = "absolute"  # import module or from module import name
    RELATIVE = "relative"  # from . import name or from ..module import name
    STAR = "star"         # from module import *


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    module: Optional[str]  # Module being imported from (None for 'import x')
    names: List[Tuple[str, Optional[str]]]  # List of (name, alias) tuples
    import_type: ImportType
    level: int  # Number of dots for relative imports (0 for absolute)
    line: int
    column: int
    is_star: bool = False
    
    @property
    def imported_names(self) -> List[str]:
        """Get list of names that are actually imported into namespace."""
        return [alias if alias else name for name, alias in self.names]
    
    def resolve_relative(self, current_package: str) -> Optional[str]:
        """Resolve relative import to absolute module path.
        
        Args:
            current_package: The package containing this import (e.g., 'astlib.utils')
            
        Returns:
            Absolute module path or None if cannot resolve
        """
        if self.import_type != ImportType.RELATIVE:
            return self.module
            
        # Handle relative imports
        if not current_package:
            return None  # Can't resolve relative import without package context
            
        parts = current_package.split('.')
        
        # Go up 'level' directories
        if self.level > len(parts):
            return None  # Too many dots
            
        # Remove 'level' parts from the end
        base_parts = parts[:-self.level] if self.level > 0 else parts
        
        # Add the module if specified
        if self.module:
            base_parts.append(self.module)
            
        return '.'.join(base_parts) if base_parts else None
    
    def resolve_import_name(self, name: str, current_package: str) -> str:
        """Resolve an imported name to its full module path.
        
        This is used for resolving names in relative imports like 'from . import name'.
        
        Args:
            name: The name being imported
            current_package: The current package context
            
        Returns:
            Full module path
        """
        if self.import_type != ImportType.RELATIVE or self.module is not None:
            # Not a relative import or has explicit module
            return name
            
        # For 'from . import name', resolve relative to current package
        parts = current_package.split('.')
        
        if self.level > len(parts):
            return name  # Can't resolve, return as-is
            
        # Remove 'level' parts from the end
        base_parts = parts[:-self.level] if self.level > 0 else parts
        base_parts.append(name)
        
        return '.'.join(base_parts)


@dataclass 
class ModuleImports:
    """All imports for a single module."""
    module_name: str
    file_path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    
    @property
    def imported_modules(self) -> Set[str]:
        """Get set of all modules imported (resolved to absolute)."""
        modules = set()
        for imp in self.imports:
            if imp.is_star:
                # For star imports, we need the module
                if imp.module:
                    abs_module = imp.resolve_relative(self.module_name)
                    if abs_module:
                        modules.add(abs_module)
            else:
                # Regular imports
                if imp.module:
                    abs_module = imp.resolve_relative(self.module_name)
                    if abs_module:
                        # For from imports, we might be importing specific items
                        # from a module, but we still track the module itself
                        modules.add(abs_module)
                else:
                    # Direct import like 'import os' or 'from . import name'
                    for name, _ in imp.names:
                        if imp.import_type == ImportType.RELATIVE:
                            # Resolve relative import like 'from . import tools'
                            resolved = imp.resolve_import_name(name, self.module_name)
                            modules.add(resolved)
                        else:
                            # Regular import
                            modules.add(name)
        return modules
    
    @property
    def has_circular_potential(self) -> bool:
        """Check if module imports from its own package."""
        package = self.module_name.rsplit('.', 1)[0] if '.' in self.module_name else None
        if not package:
            return False
            
        for module in self.imported_modules:
            if module.startswith(package):
                return True
        return False


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""
    
    def __init__(self):
        self.imports: List[ImportInfo] = []
        
    def visit_Import(self, node: ast.Import) -> None:
        """Handle 'import module' statements."""
        names = [(alias.name, alias.asname) for alias in node.names]
        
        # For regular imports, each name is a separate module
        for name, asname in names:
            import_info = ImportInfo(
                module=None,  # No 'from' part
                names=[(name, asname)],
                import_type=ImportType.ABSOLUTE,
                level=0,
                line=node.lineno,
                column=node.col_offset
            )
            self.imports.append(import_info)
            
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle 'from module import name' statements."""
        # Check for star import
        is_star = len(node.names) == 1 and node.names[0].name == '*'
        
        names = [(alias.name, alias.asname) for alias in node.names]
        
        import_info = ImportInfo(
            module=node.module,
            names=names,
            import_type=ImportType.RELATIVE if node.level > 0 else ImportType.ABSOLUTE,
            level=node.level,
            line=node.lineno,
            column=node.col_offset,
            is_star=is_star
        )
        
        if is_star:
            import_info.import_type = ImportType.STAR
            
        self.imports.append(import_info)


def extract_imports(tree: ast.AST) -> List[ImportInfo]:
    """Extract all import statements from an AST.
    
    Args:
        tree: Parsed AST tree
        
    Returns:
        List of ImportInfo objects
    """
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports


def analyze_imports(
    file_path: Union[str, Path],
    module_name: Optional[str] = None
) -> ModuleImports:
    """Analyze imports in a Python file.
    
    Args:
        file_path: Path to Python file
        module_name: Module name (e.g., 'astlib.imports'). If not provided,
                    will try to infer from file path.
                    
    Returns:
        ModuleImports object containing all import information
        
    Raises:
        FileNotFoundError: If file doesn't exist
        SyntaxError: If file has syntax errors
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Try to infer module name if not provided
    if module_name is None:
        module_name = _infer_module_name(file_path)
        
    # Parse the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in {file_path}: {e}")
        
    # Extract imports
    imports = extract_imports(tree)
    
    return ModuleImports(
        module_name=module_name or str(file_path),
        file_path=file_path,
        imports=imports
    )


def _infer_module_name(file_path: Path) -> str:
    """Try to infer module name from file path.
    
    This looks for __init__.py files to determine package boundaries.
    """
    parts = []
    current = file_path
    
    # Remove .py extension from filename
    if current.suffix == '.py':
        if current.stem != '__init__':
            parts.append(current.stem)
        current = current.parent
    
    # Walk up looking for __init__.py files
    while current != current.parent:
        init_file = current / '__init__.py'
        if init_file.exists():
            parts.append(current.name)
            current = current.parent
        else:
            break
            
    # Reverse to get correct order
    parts.reverse()
    return '.'.join(parts) if parts else file_path.stem


def find_all_imports(
    directory: Union[str, Path],
    ignore_patterns: Optional[List[str]] = None
) -> Dict[str, ModuleImports]:
    """Find all imports in a directory tree.
    
    Args:
        directory: Root directory to search
        ignore_patterns: Glob patterns to ignore
        
    Returns:
        Dictionary mapping module names to their imports
    """
    from astlib.walker import DirectoryWalker
    
    directory = Path(directory)
    walker = DirectoryWalker(ignore_patterns=ignore_patterns or [])
    
    results = {}
    
    for py_file in walker.walk(directory):
        try:
            module_imports = analyze_imports(py_file)
            results[module_imports.module_name] = module_imports
        except (SyntaxError, UnicodeDecodeError):
            # Skip files with errors
            pass
            
    return results