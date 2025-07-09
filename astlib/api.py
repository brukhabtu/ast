"""
Unified API for the AST library.

This module provides the ASTLib class, which is the main entry point
for all AST operations. It provides a clean, intuitive interface that
wraps all the library's functionality.
"""

from pathlib import Path
from typing import List, Optional, Callable, Any
from dataclasses import dataclass

from .indexer import ProjectIndex, DefinitionResult, IndexStats
from .symbols import Symbol
from .imports import ModuleImports, analyze_imports
from .import_graph import CircularImport
from .parser import parse_file, ParseResult
from .references import Reference as ReferenceImport
from .call_graph import CallGraph, CallNode, CallChain, CallGraphStats, build_call_graph


@dataclass
class Reference:
    """Represents a reference to a symbol (placeholder for iteration 2)."""
    symbol_name: str
    file_path: str
    line: int
    column: int
    context: str = ""


@dataclass
class FileAnalysis:
    """Complete analysis of a single file."""
    file_path: str
    parse_result: ParseResult
    symbols: List[Symbol]
    imports: ModuleImports
    errors: List[str]
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return len(self.errors) == 0 and self.parse_result.success


@dataclass
class ProjectAnalysis:
    """Complete analysis of a project."""
    root_path: str
    total_files: int
    total_symbols: int
    total_functions: int
    total_classes: int
    circular_imports: List[CircularImport]
    import_depth: int
    errors: List[tuple[str, str]]
    analysis_time: float


class ASTLib:
    """
    Main API for the AST library.
    
    This class provides a unified interface to all AST operations,
    including parsing, symbol extraction, import analysis, and more.
    
    Example:
        >>> from astlib import ASTLib
        >>> ast_lib = ASTLib("my_project/")
        >>> 
        >>> # Find a symbol definition
        >>> definition = ast_lib.find_definition("MyClass")
        >>> print(f"Found at {definition.file_path}:{definition.line}")
        >>> 
        >>> # Analyze imports
        >>> circular = ast_lib.find_circular_imports()
        >>> for circle in circular:
        ...     print(f"Circular import: {' -> '.join(circle.modules)}")
    """
    
    def __init__(self, project_path: str, lazy: bool = True, 
                 progress_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize AST library for a project.
        
        Args:
            project_path: Root directory of the project
            lazy: If True, don't index until needed (default: True)
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path).resolve()
        self._index = ProjectIndex(str(self.project_path))
        self._indexed = False
        self._lazy = lazy
        self._progress_callback = progress_callback
        self._call_graph: Optional[CallGraph] = None
        
        if not lazy:
            self._ensure_indexed()
            
    def _ensure_indexed(self) -> None:
        """Ensure the project is indexed."""
        if not self._indexed:
            if self._progress_callback:
                self._progress_callback("Building project index...")
                
            # Use progress bars if no callback provided
            show_progress = self._progress_callback is None
            self._index.build_index(show_progress=show_progress)
            self._indexed = True
            
            if self._progress_callback:
                stats = self._index.get_stats()
                self._progress_callback(
                    f"Indexed {stats.total_files} files with {stats.total_symbols} symbols"
                )
                
    # Symbol operations
    
    def find_definition(self, symbol: str, from_file: Optional[str] = None) -> Optional[DefinitionResult]:
        """
        Find the definition of a symbol.
        
        Args:
            symbol: Name of the symbol to find
            from_file: Optional context file for resolving ambiguity
            
        Returns:
            DefinitionResult if found, None otherwise
            
        Example:
            >>> result = ast_lib.find_definition("parse_file")
            >>> if result:
            ...     print(f"Found at {result.file_path}:{result.line}")
        """
        self._ensure_indexed()
        return self._index.find_definition(symbol, from_file)
        
    def find_references(self, symbol: str) -> List[Reference]:
        """
        Find all references to a symbol.
        
        Args:
            symbol: Name of the symbol to find references for
            
        Returns:
            List of Reference objects
            
        Example:
            >>> refs = ast_lib.find_references("parse_file")
            >>> for ref in refs:
            ...     print(f"{ref.file_path}:{ref.line} - {ref.reference_type.value}")
        """
        self._ensure_indexed()
        
        # Convert from internal Reference type to API Reference type
        internal_refs = self._index.find_references(symbol)
        
        # Convert to API Reference objects
        api_refs = []
        for ref in internal_refs:
            api_ref = Reference(
                symbol_name=ref.symbol_name,
                file_path=ref.file_path,
                line=ref.line,
                column=ref.column,
                context=ref.context
            )
            api_refs.append(api_ref)
            
        return api_refs
        
    def get_symbols(self, file_path: Optional[str] = None) -> List[Symbol]:
        """
        Get symbols from a file or the entire project.
        
        Args:
            file_path: Optional file to get symbols from. If None,
                      returns all symbols in the project.
                      
        Returns:
            List of Symbol objects
        """
        self._ensure_indexed()
        
        if file_path:
            return self._index.get_symbols_in_file(file_path)
        else:
            return self._index.get_all_symbols()
            
    # Import operations
    
    def get_imports(self, file_path: str) -> ModuleImports:
        """
        Get all imports from a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ModuleImports object with all imports
        """
        return analyze_imports(file_path)
        
    def get_dependencies(self, module: str) -> List[str]:
        """
        Get all modules that a given module depends on.
        
        Args:
            module: Module name or path
            
        Returns:
            List of module names
        """
        self._ensure_indexed()
        graph = self._index.get_import_graph()
        
        if graph and module in graph.nodes:
            return list(graph.nodes[module].imports)
        return []
        
    def find_circular_imports(self) -> List[CircularImport]:
        """
        Find all circular imports in the project.
        
        Returns:
            List of CircularImport objects
        """
        self._ensure_indexed()
        graph = self._index.get_import_graph()
        
        if graph:
            return graph.find_circular_imports()
        return []
        
    # Analysis operations
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Perform complete analysis of a single file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileAnalysis object with all information
        """
        errors = []
        
        # Parse the file
        parse_result = parse_file(file_path)
        
        # Extract symbols if parsing succeeded
        symbols = []
        if parse_result.tree:
            from .symbols import extract_symbols
            symbols = extract_symbols(parse_result.tree, file_path)
        else:
            errors.append("Failed to parse file")
            
        # Analyze imports
        imports = analyze_imports(file_path)
        
        return FileAnalysis(
            file_path=file_path,
            parse_result=parse_result,
            symbols=symbols,
            imports=imports,
            errors=errors
        )
        
    def analyze_project(self) -> ProjectAnalysis:
        """
        Perform complete analysis of the entire project.
        
        Returns:
            ProjectAnalysis object with project-wide statistics
        """
        self._ensure_indexed()
        stats = self._index.get_stats()
        graph = self._index.get_import_graph()
        
        # Get circular imports
        circular_imports = []
        import_depth = 0
        
        if graph:
            circular_imports = graph.find_circular_imports()
            levels = graph.get_import_levels()
            import_depth = len(levels) if levels else 0
            
        return ProjectAnalysis(
            root_path=str(self.project_path),
            total_files=stats.total_files,
            total_symbols=stats.total_symbols,
            total_functions=stats.total_functions,
            total_classes=stats.total_classes,
            circular_imports=circular_imports,
            import_depth=import_depth,
            errors=stats.errors,
            analysis_time=stats.index_time_seconds
        )
        
    # Call graph operations
    
    def _ensure_call_graph(self) -> None:
        """Ensure the call graph is built."""
        self._ensure_indexed()
        if self._call_graph is None:
            if self._progress_callback:
                self._progress_callback("Building call graph...")
            show_progress = self._progress_callback is None
            self._call_graph = build_call_graph(str(self.project_path), show_progress=show_progress)
            
    def get_call_graph(self) -> CallGraph:
        """
        Get the call graph for the project.
        
        Returns:
            CallGraph object containing all call relationships
        """
        self._ensure_call_graph()
        return self._call_graph
        
    def find_callers(self, function_name: str, file_path: Optional[str] = None) -> List[CallNode]:
        """
        Find all functions that call the given function.
        
        Args:
            function_name: Name of the function
            file_path: Optional file path to narrow the search
            
        Returns:
            List of CallNode objects representing callers
            
        Example:
            >>> callers = ast_lib.find_callers("parse_file")
            >>> for caller in callers:
            ...     print(f"{caller.name} in {caller.file_path}:{caller.line}")
        """
        self._ensure_call_graph()
        return self._call_graph.get_callers(function_name, file_path)
        
    def find_callees(self, function_name: str, file_path: Optional[str] = None) -> List[CallNode]:
        """
        Find all functions called by the given function.
        
        Args:
            function_name: Name of the function
            file_path: Optional file path to narrow the search
            
        Returns:
            List of CallNode objects representing callees
            
        Example:
            >>> callees = ast_lib.find_callees("main")
            >>> for callee in callees:
            ...     print(f"Calls {callee.name}")
        """
        self._ensure_call_graph()
        return self._call_graph.get_callees(function_name, file_path)
        
    def find_call_chains(self, start: str, end: str, max_depth: int = 10) -> List[CallChain]:
        """
        Find all call chains from one function to another.
        
        Args:
            start: Starting function name
            end: Ending function name
            max_depth: Maximum depth to search (default: 10)
            
        Returns:
            List of CallChain objects representing paths
            
        Example:
            >>> chains = ast_lib.find_call_chains("main", "save_file")
            >>> for chain in chains:
            ...     print(f"Path: {chain}")
        """
        self._ensure_call_graph()
        return self._call_graph.find_call_chains(start, end, max_depth)
        
    def get_call_hierarchy(self, function_name: str, direction: str = "down", max_depth: int = 5) -> dict:
        """
        Get the call hierarchy for a function.
        
        Args:
            function_name: Name of the function
            direction: "down" for callees, "up" for callers, "both" for both
            max_depth: Maximum depth to traverse
            
        Returns:
            Hierarchical dictionary of calls
        """
        self._ensure_call_graph()
        return self._call_graph.get_call_hierarchy(function_name, direction, max_depth)
        
    def find_recursive_functions(self) -> List[CallNode]:
        """
        Find all recursive functions in the project.
        
        Returns:
            List of CallNode objects that are recursive
        """
        self._ensure_call_graph()
        return self._call_graph.find_recursive_functions()
        
    def get_call_graph_stats(self) -> CallGraphStats:
        """
        Get statistics about the call graph.
        
        Returns:
            CallGraphStats object with metrics
        """
        self._ensure_call_graph()
        return self._call_graph.get_stats()
        
    def export_call_graph_dot(self) -> str:
        """
        Export the call graph in Graphviz DOT format.
        
        Returns:
            DOT format string
            
        Example:
            >>> dot = ast_lib.export_call_graph_dot()
            >>> with open("callgraph.dot", "w") as f:
            ...     f.write(dot)
        """
        self._ensure_call_graph()
        return self._call_graph.export_dot()

    # Utility operations
    
    def refresh(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Refresh the index for modified files.
        
        Args:
            file_paths: Optional list of specific files to refresh.
                       If None, refreshes all modified files.
        """
        if not self._indexed:
            return  # Nothing to refresh
            
        if self._progress_callback:
            self._progress_callback("Refreshing index...")
            
        self._index.refresh(file_paths)
        
        # Clear call graph cache as it may be outdated
        self._call_graph = None
        
    def clear_cache(self) -> None:
        """Clear all caches."""
        from . import clear_caches
        clear_caches()
        
    def reindex(self) -> None:
        """Force a complete reindex of the project."""
        self._index.clear()
        self._indexed = False
        self._call_graph = None
        self._ensure_indexed()
        
    def get_stats(self) -> IndexStats:
        """Get indexing statistics."""
        self._ensure_indexed()
        return self._index.get_stats()