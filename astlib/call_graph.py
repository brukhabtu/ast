"""
Call graph analysis for understanding function relationships.

This module builds on reference finding to create call graphs showing
caller/callee relationships, call chains, and dependency analysis.
"""

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

from .parser import parse_file
from .symbols import Symbol, SymbolType
from .references import find_references, ReferenceType


@dataclass
class CallNode:
    """Represents a function/method in the call graph."""
    name: str
    qualified_name: str
    file_path: str
    line: int
    is_method: bool = False
    class_name: Optional[str] = None
    
    def __hash__(self):
        return hash((self.qualified_name, self.file_path))
    
    def __eq__(self, other):
        if not isinstance(other, CallNode):
            return False
        return (self.qualified_name == other.qualified_name and 
                self.file_path == other.file_path)


@dataclass
class CallEdge:
    """Represents a call relationship between two functions."""
    caller: CallNode
    callee: CallNode
    line: int  # Line number where the call occurs
    call_type: str = "direct"  # direct, method, recursive
    
    @property
    def is_recursive(self) -> bool:
        """Check if this is a recursive call."""
        return self.caller == self.callee


@dataclass 
class CallChain:
    """Represents a chain of function calls."""
    nodes: List[CallNode]
    
    def __str__(self):
        return " → ".join(node.name for node in self.nodes)
    
    @property
    def depth(self) -> int:
        return len(self.nodes)
    
    @property 
    def contains_recursion(self) -> bool:
        """Check if the chain contains recursive calls."""
        seen = set()
        for node in self.nodes:
            if node in seen:
                return True
            seen.add(node)
        return False


@dataclass
class CallGraphStats:
    """Statistics about the call graph."""
    total_functions: int = 0
    total_calls: int = 0
    recursive_functions: Set[str] = field(default_factory=set)
    max_call_depth: int = 0
    most_called_functions: List[Tuple[str, int]] = field(default_factory=list)
    isolated_functions: Set[str] = field(default_factory=set)


class CallGraph:
    """
    Builds and analyzes function call relationships.
    
    This class creates a directed graph of function calls, enabling
    analysis of caller/callee relationships, call chains, and dependencies.
    """
    
    def __init__(self):
        self.nodes: Dict[str, CallNode] = {}
        self.edges: List[CallEdge] = []
        self.callers: Dict[str, Set[CallNode]] = defaultdict(set)
        self.callees: Dict[str, Set[CallNode]] = defaultdict(set)
        self._call_counts: Dict[str, int] = defaultdict(int)
        
    def add_node(self, node: CallNode) -> None:
        """Add a function node to the graph."""
        key = f"{node.file_path}:{node.qualified_name}"
        self.nodes[key] = node
        
    def add_edge(self, edge: CallEdge) -> None:
        """Add a call relationship to the graph."""
        self.edges.append(edge)
        
        # Update caller/callee mappings
        caller_key = f"{edge.caller.file_path}:{edge.caller.qualified_name}"
        callee_key = f"{edge.callee.file_path}:{edge.callee.qualified_name}"
        
        self.callees[caller_key].add(edge.callee)
        self.callers[callee_key].add(edge.caller)
        self._call_counts[callee_key] += 1
        
    def get_callers(self, function_name: str, file_path: Optional[str] = None) -> List[CallNode]:
        """Get all functions that call the given function."""
        results = []
        
        for key, node in self.nodes.items():
            if node.name == function_name or node.qualified_name == function_name:
                if file_path and node.file_path != file_path:
                    continue
                results.extend(self.callers.get(key, []))
                
        return list(set(results))
    
    def get_callees(self, function_name: str, file_path: Optional[str] = None) -> List[CallNode]:
        """Get all functions called by the given function."""
        results = []
        
        for key, node in self.nodes.items():
            if node.name == function_name or node.qualified_name == function_name:
                if file_path and node.file_path != file_path:
                    continue
                results.extend(self.callees.get(key, []))
                
        return list(set(results))
    
    def find_call_chains(self, start: str, end: str, max_depth: int = 10) -> List[CallChain]:
        """Find all call chains from start function to end function."""
        chains = []
        
        # Find start nodes
        start_nodes = [node for node in self.nodes.values() 
                      if node.name == start or node.qualified_name == start]
        
        # Find end nodes  
        end_nodes = [node for node in self.nodes.values()
                    if node.name == end or node.qualified_name == end]
        
        if not start_nodes or not end_nodes:
            return chains
            
        # BFS to find paths
        for start_node in start_nodes:
            for end_node in end_nodes:
                paths = self._find_paths_bfs(start_node, end_node, max_depth)
                chains.extend(CallChain(nodes=path) for path in paths)
                
        return chains
    
    def _find_paths_bfs(self, start: CallNode, end: CallNode, max_depth: int) -> List[List[CallNode]]:
        """Use BFS to find all paths between two nodes."""
        if start == end:
            return [[start]]
            
        paths = []
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
                
            current_key = f"{current.file_path}:{current.qualified_name}"
            
            for next_node in self.callees.get(current_key, []):
                if next_node in path:  # Avoid cycles
                    continue
                    
                new_path = path + [next_node]
                
                if next_node == end:
                    paths.append(new_path)
                else:
                    queue.append((next_node, new_path))
                    
        return paths
    
    def find_recursive_functions(self) -> List[CallNode]:
        """Find all functions that call themselves (directly or indirectly)."""
        recursive_funcs = []
        
        for node in self.nodes.values():
            # Check direct recursion
            node_key = f"{node.file_path}:{node.qualified_name}"
            if node in self.callees.get(node_key, []):
                recursive_funcs.append(node)
                continue
                
            # Check indirect recursion
            if self._has_cycle_from(node):
                recursive_funcs.append(node)
                
        return list(set(recursive_funcs))
    
    def _has_cycle_from(self, start: CallNode, visited: Optional[Set[CallNode]] = None) -> bool:
        """Check if there's a cycle starting from the given node."""
        if visited is None:
            visited = set()
            
        if start in visited:
            return True
            
        visited.add(start)
        start_key = f"{start.file_path}:{start.qualified_name}"
        
        for next_node in self.callees.get(start_key, []):
            if self._has_cycle_from(next_node, visited.copy()):
                return True
                
        return False
    
    def get_call_hierarchy(self, function_name: str, direction: str = "down", max_depth: int = 5) -> Dict:
        """
        Get the call hierarchy for a function.
        
        Args:
            function_name: Name of the function
            direction: "down" for callees, "up" for callers, "both" for both
            max_depth: Maximum depth to traverse
            
        Returns:
            Hierarchical dictionary of calls
        """
        # Find the node
        target_nodes = [node for node in self.nodes.values()
                       if node.name == function_name or node.qualified_name == function_name]
        
        if not target_nodes:
            return {}
            
        hierarchy = {}
        
        for node in target_nodes:
            node_hierarchy = {
                "name": node.qualified_name,
                "file": node.file_path,
                "line": node.line
            }
            
            if direction in ("down", "both"):
                node_hierarchy["calls"] = self._get_callees_hierarchy(node, max_depth, set())
                
            if direction in ("up", "both"):
                node_hierarchy["called_by"] = self._get_callers_hierarchy(node, max_depth, set())
                
            hierarchy[node.qualified_name] = node_hierarchy
            
        return hierarchy
    
    def _get_callees_hierarchy(self, node: CallNode, depth: int, visited: Set[CallNode]) -> List[Dict]:
        """Recursively get callee hierarchy."""
        if depth <= 0 or node in visited:
            return []
            
        visited.add(node)
        node_key = f"{node.file_path}:{node.qualified_name}"
        callees = []
        
        for callee in self.callees.get(node_key, []):
            callee_info = {
                "name": callee.qualified_name,
                "file": callee.file_path,
                "line": callee.line,
                "calls": self._get_callees_hierarchy(callee, depth - 1, visited.copy())
            }
            callees.append(callee_info)
            
        return callees
    
    def _get_callers_hierarchy(self, node: CallNode, depth: int, visited: Set[CallNode]) -> List[Dict]:
        """Recursively get caller hierarchy."""
        if depth <= 0 or node in visited:
            return []
            
        visited.add(node)
        node_key = f"{node.file_path}:{node.qualified_name}"
        callers = []
        
        for caller in self.callers.get(node_key, []):
            caller_info = {
                "name": caller.qualified_name,
                "file": caller.file_path,
                "line": caller.line,
                "called_by": self._get_callers_hierarchy(caller, depth - 1, visited.copy())
            }
            callers.append(caller_info)
            
        return callers
    
    def get_stats(self) -> CallGraphStats:
        """Get statistics about the call graph."""
        stats = CallGraphStats()
        
        stats.total_functions = len(self.nodes)
        stats.total_calls = len(self.edges)
        
        # Find recursive functions
        recursive = self.find_recursive_functions()
        stats.recursive_functions = {f.qualified_name for f in recursive}
        
        # Find most called functions
        call_counts = [(name.split(":")[-1], count) 
                      for name, count in self._call_counts.items()]
        call_counts.sort(key=lambda x: x[1], reverse=True)
        stats.most_called_functions = call_counts[:10]
        
        # Find isolated functions (no callers or callees)
        for key, node in self.nodes.items():
            if not self.callers.get(key) and not self.callees.get(key):
                stats.isolated_functions.add(node.qualified_name)
                
        # Calculate max call depth
        for node in self.nodes.values():
            depth = self._calculate_max_depth(node, set())
            stats.max_call_depth = max(stats.max_call_depth, depth)
            
        return stats
    
    def _calculate_max_depth(self, node: CallNode, visited: Set[CallNode]) -> int:
        """Calculate the maximum call depth from a node."""
        if node in visited:
            return 0
            
        visited.add(node)
        node_key = f"{node.file_path}:{node.qualified_name}"
        
        max_depth = 0
        for callee in self.callees.get(node_key, []):
            depth = 1 + self._calculate_max_depth(callee, visited.copy())
            max_depth = max(max_depth, depth)
            
        return max_depth
    
    def export_dot(self) -> str:
        """Export the call graph in Graphviz DOT format."""
        lines = ["digraph CallGraph {"]
        lines.append('  rankdir="LR";')
        lines.append('  node [shape=box];')
        
        # Add nodes
        for node in self.nodes.values():
            label = f"{node.name}\\n{Path(node.file_path).name}:{node.line}"
            lines.append(f'  "{node.qualified_name}" [label="{label}"];')
            
        # Add edges
        for edge in self.edges:
            style = "dashed" if edge.is_recursive else "solid"
            lines.append(f'  "{edge.caller.qualified_name}" -> "{edge.callee.qualified_name}" [style={style}];')
            
        lines.append("}")
        return "\n".join(lines)


def build_call_graph(project_path: str, show_progress: bool = True) -> CallGraph:
    """
    Build a call graph for an entire project.
    
    Args:
        project_path: Root directory of the project
        show_progress: Whether to show progress bars
        
    Returns:
        CallGraph object containing all call relationships
    """
    from .indexer import ProjectIndex
    from .walker import DirectoryWalker
    
    graph = CallGraph()
    index = ProjectIndex(project_path)
    
    # Build index first
    index.build_index(show_progress=show_progress)
    
    # Get all function/method symbols
    all_symbols = index.get_all_symbols()
    function_symbols = [s for s in all_symbols 
                       if s.type in (SymbolType.FUNCTION, SymbolType.METHOD)]
    
    # Add nodes for all functions
    for symbol in function_symbols:
        node = CallNode(
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            file_path=symbol.file_path,
            line=symbol.position.line,
            is_method=symbol.type == SymbolType.METHOD,
            class_name=symbol.parent.name if symbol.parent else None
        )
        graph.add_node(node)
    
    # Analyze each function to find calls
    walker = DirectoryWalker()
    python_files = list(walker.walk(project_path))
    
    if show_progress:
        from tqdm import tqdm
        file_iterator = tqdm(python_files, desc="Building call graph")
    else:
        file_iterator = python_files
        
    for file_path in file_iterator:
        _analyze_file_calls(str(file_path), graph, index)
        
    return graph


def _analyze_file_calls(file_path: str, graph: CallGraph, index) -> None:
    """Analyze a single file for function calls."""
    result = parse_file(file_path)
    if not result.tree:
        return
        
    # Find all function calls in the file
    for node in ast.walk(result.tree):
        if isinstance(node, ast.Call):
            caller_func = _find_containing_function(node, result.tree)
            if not caller_func:
                continue
                
            # Determine what's being called
            called_name = None
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
                
            if not called_name:
                continue
                
            # Find the caller in our graph
            caller_node = None
            for graph_node in graph.nodes.values():
                if (graph_node.file_path == file_path and 
                    graph_node.name == caller_func.name):
                    caller_node = graph_node
                    break
                    
            if not caller_node:
                continue
                
            # Try to resolve the callee
            callee_def = index.find_definition(called_name, file_path)
            if callee_def:
                # Find the callee in our graph
                for graph_node in graph.nodes.values():
                    if (graph_node.file_path == callee_def.file_path and
                        graph_node.line == callee_def.line):
                        edge = CallEdge(
                            caller=caller_node,
                            callee=graph_node,
                            line=node.lineno,
                            call_type="recursive" if caller_node == graph_node else "direct"
                        )
                        graph.add_edge(edge)
                        break


def _find_containing_function(node: ast.AST, tree: ast.AST) -> Optional[ast.FunctionDef]:
    """Find the innermost function that contains the given node."""
    class FunctionFinder(ast.NodeVisitor):
        def __init__(self, target_node):
            self.target_node = target_node
            self.current_function = None
            self.containing_function = None
            
        def visit_FunctionDef(self, func_node):
            # Save previous function context
            prev_function = self.current_function
            self.current_function = func_node
            
            # Visit children
            self.generic_visit(func_node)
            
            # Restore previous context
            self.current_function = prev_function
            
        visit_AsyncFunctionDef = visit_FunctionDef
            
        def generic_visit(self, node):
            if node is self.target_node and self.current_function:
                self.containing_function = self.current_function
            super().generic_visit(node)
    
    finder = FunctionFinder(node)
    finder.visit(tree)
    return finder.containing_function