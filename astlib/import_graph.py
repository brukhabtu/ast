#!/usr/bin/env python3
"""Import dependency graph construction and analysis.

This module provides tools to:
- Build directed graphs of import dependencies
- Detect circular imports
- Analyze import relationships
- Visualize import structure
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

from astlib.imports import ModuleImports, find_all_imports


@dataclass
class ImportNode:
    """A node in the import dependency graph."""
    module_name: str
    file_path: Optional[Path] = None
    imports: Set[str] = field(default_factory=set)  # Modules this node imports
    imported_by: Set[str] = field(default_factory=set)  # Modules that import this node
    
    @property
    def is_external(self) -> bool:
        """Check if this is an external (not in project) module."""
        return self.file_path is None
        
    @property
    def import_count(self) -> int:
        """Number of modules this node imports."""
        return len(self.imports)
        
    @property
    def imported_by_count(self) -> int:
        """Number of modules that import this node."""
        return len(self.imported_by)


@dataclass
class CircularImport:
    """Information about a circular import."""
    cycle: List[str]  # List of modules forming the cycle
    
    @property
    def cycle_length(self) -> int:
        """Length of the circular import chain."""
        return len(self.cycle)
        
    def __str__(self) -> str:
        """String representation of the cycle."""
        return " -> ".join(self.cycle) + " -> " + self.cycle[0]


class ImportGraph:
    """Directed graph of import dependencies."""
    
    def __init__(self):
        self.nodes: Dict[str, ImportNode] = {}
        self._cycles: Optional[List[CircularImport]] = None
        
    def add_module(self, module_imports: ModuleImports) -> None:
        """Add a module and its imports to the graph.
        
        Args:
            module_imports: Module import information
        """
        module_name = module_imports.module_name
        
        # Create or get the node for this module
        if module_name not in self.nodes:
            self.nodes[module_name] = ImportNode(
                module_name=module_name,
                file_path=module_imports.file_path
            )
            
        node = self.nodes[module_name]
        
        # Add all imports
        for imported_module in module_imports.imported_modules:
            # Add to this node's imports
            node.imports.add(imported_module)
            
            # Create node for imported module if it doesn't exist
            if imported_module not in self.nodes:
                self.nodes[imported_module] = ImportNode(
                    module_name=imported_module,
                    file_path=None  # External module
                )
                
            # Add reverse reference
            self.nodes[imported_module].imported_by.add(module_name)
            
        # Clear cached cycles
        self._cycles = None
        
    def get_dependencies(self, module_name: str, recursive: bool = False) -> Set[str]:
        """Get all modules that a given module depends on.
        
        Args:
            module_name: Name of the module
            recursive: If True, get transitive dependencies
            
        Returns:
            Set of module names
        """
        if module_name not in self.nodes:
            return set()
            
        if not recursive:
            return self.nodes[module_name].imports.copy()
            
        # BFS for all transitive dependencies
        visited = set()
        queue = deque([module_name])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            
            if current in self.nodes:
                for imported in self.nodes[current].imports:
                    if imported not in visited:
                        queue.append(imported)
                        
        visited.remove(module_name)  # Don't include the module itself
        return visited
        
    def get_dependents(self, module_name: str, recursive: bool = False) -> Set[str]:
        """Get all modules that depend on a given module.
        
        Args:
            module_name: Name of the module
            recursive: If True, get transitive dependents
            
        Returns:
            Set of module names
        """
        if module_name not in self.nodes:
            return set()
            
        if not recursive:
            return self.nodes[module_name].imported_by.copy()
            
        # BFS for all transitive dependents
        visited = set()
        queue = deque([module_name])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            
            if current in self.nodes:
                for dependent in self.nodes[current].imported_by:
                    if dependent not in visited:
                        queue.append(dependent)
                        
        visited.remove(module_name)  # Don't include the module itself
        return visited
        
    def find_circular_imports(self) -> List[CircularImport]:
        """Find all circular imports in the graph.
        
        Returns:
            List of CircularImport objects
        """
        if self._cycles is not None:
            return self._cycles
            
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def _dfs(node_name: str) -> None:
            """DFS to find cycles."""
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)
            
            if node_name in self.nodes:
                for neighbor in self.nodes[node_name].imports:
                    # Only consider internal nodes for cycles
                    if neighbor in self.nodes and not self.nodes[neighbor].is_external:
                        if neighbor not in visited:
                            _dfs(neighbor)
                        elif neighbor in rec_stack:
                            # Found a cycle
                            cycle_start = path.index(neighbor)
                            cycle = path[cycle_start:]
                            cycles.append(CircularImport(cycle=cycle))
                        
            path.pop()
            rec_stack.remove(node_name)
            
        # Check all internal nodes
        for node_name in self.nodes:
            if node_name not in visited and not self.nodes[node_name].is_external:
                _dfs(node_name)
                
        # Remove duplicate cycles
        unique_cycles = []
        seen = set()
        
        for cycle in cycles:
            # Normalize cycle (start with smallest module name)
            min_idx = cycle.cycle.index(min(cycle.cycle))
            normalized = cycle.cycle[min_idx:] + cycle.cycle[:min_idx]
            cycle_key = tuple(normalized)
            
            if cycle_key not in seen:
                seen.add(cycle_key)
                unique_cycles.append(CircularImport(cycle=normalized))
                
        self._cycles = unique_cycles
        return unique_cycles
        
    def get_import_levels(self) -> Dict[int, List[str]]:
        """Get modules organized by import levels.
        
        Level 0: modules with no dependencies
        Level 1: modules that only depend on level 0
        etc.
        
        Returns:
            Dict mapping level to list of module names
        """
        levels = {}
        assigned = set()
        
        # Find modules with no internal dependencies
        level_0 = []
        for name, node in self.nodes.items():
            internal_deps = {
                dep for dep in node.imports 
                if dep in self.nodes and not self.nodes[dep].is_external
            }
            if not internal_deps:
                level_0.append(name)
                assigned.add(name)
                
        levels[0] = level_0
        
        # Build subsequent levels
        level = 1
        while len(assigned) < len(self.nodes):
            current_level = []
            
            for name, node in self.nodes.items():
                if name in assigned or node.is_external:
                    continue
                    
                # Check if all dependencies are assigned
                internal_deps = {
                    dep for dep in node.imports 
                    if dep in self.nodes and not self.nodes[dep].is_external
                }
                
                if internal_deps.issubset(assigned):
                    current_level.append(name)
                    
            if not current_level:
                # Remaining modules must have circular dependencies
                unassigned = [
                    name for name in self.nodes 
                    if name not in assigned and not self.nodes[name].is_external
                ]
                levels[-1] = unassigned  # -1 indicates circular deps
                break
                
            levels[level] = current_level
            assigned.update(current_level)
            level += 1
            
        return levels
        
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the import graph.
        
        Returns:
            Dictionary with various statistics
        """
        internal_nodes = [n for n in self.nodes.values() if not n.is_external]
        
        if not internal_nodes:
            return {
                'total_modules': 0,
                'internal_modules': 0,
                'external_modules': 0,
                'total_imports': 0,
                'avg_imports_per_module': 0.0,
                'max_imports': 0,
                'max_imported_by': 0,
                'circular_import_count': 0
            }
            
        total_imports = sum(n.import_count for n in internal_nodes)
        
        return {
            'total_modules': len(self.nodes),
            'internal_modules': len(internal_nodes),
            'external_modules': len(self.nodes) - len(internal_nodes),
            'total_imports': total_imports,
            'avg_imports_per_module': total_imports / len(internal_nodes),
            'max_imports': max(n.import_count for n in internal_nodes),
            'max_imported_by': max(n.imported_by_count for n in internal_nodes),
            'circular_import_count': len(self.find_circular_imports())
        }


def build_import_graph(
    directory: Union[str, Path],
    ignore_patterns: Optional[List[str]] = None
) -> ImportGraph:
    """Build import graph for all Python files in a directory.
    
    Args:
        directory: Root directory to analyze
        ignore_patterns: Glob patterns to ignore
        
    Returns:
        ImportGraph object
    """
    # Find all imports
    all_imports = find_all_imports(directory, ignore_patterns)
    
    # Build graph
    graph = ImportGraph()
    
    # First pass: add all modules
    for module_imports in all_imports.values():
        graph.add_module(module_imports)
    
    # Second pass: update file paths for internal modules
    for module_name, module_imports in all_imports.items():
        if module_name in graph.nodes:
            graph.nodes[module_name].file_path = module_imports.file_path
        
    return graph