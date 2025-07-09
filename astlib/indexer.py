"""
Cross-file project indexer for the AST library.

This module provides the ProjectIndex class that coordinates parsing,
symbol extraction, import analysis, and caching across an entire project.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import time

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1):
            pass
        def set_description(self, desc):
            self.desc = desc

from .parser import parse_file
from .symbols import extract_symbols, Symbol, SymbolType
from .symbol_table import SymbolTable
from .imports import analyze_imports
from .import_graph import build_import_graph, ImportGraph
from .walker import DirectoryWalker
from .types import Position
from .references import find_references, Reference, ReferenceType


@dataclass
class DefinitionResult:
    """Result of finding a symbol definition."""
    symbol: Symbol
    file_path: str
    line: int
    column: int
    qualified_name: str
    
    @classmethod
    def from_symbol(cls, symbol: Symbol) -> 'DefinitionResult':
        """Create a DefinitionResult from a Symbol."""
        return cls(
            symbol=symbol,
            file_path=symbol.file_path or "",
            line=symbol.position.line,
            column=symbol.position.column,
            qualified_name=symbol.qualified_name
        )


@dataclass 
class IndexStats:
    """Statistics about the project index."""
    total_files: int = 0
    total_symbols: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0
    index_time_seconds: float = 0.0
    errors: List[Tuple[str, str]] = field(default_factory=list)


class ProjectIndex:
    """
    Cross-file project indexer that coordinates all AST operations.
    
    This is the main entry point for indexing an entire Python project,
    providing fast symbol lookup and cross-file navigation.
    """
    
    def __init__(self, root_path: str, ignore_patterns: Optional[List[str]] = None):
        """
        Initialize a project index.
        
        Args:
            root_path: Root directory of the project to index
            ignore_patterns: Additional patterns to ignore (beyond .gitignore)
        """
        self.root_path = Path(root_path).resolve()
        self.symbol_table = SymbolTable()
        self.import_graph: Optional[ImportGraph] = None
        self._file_symbols: Dict[str, List[Symbol]] = {}
        self._indexed_files: Set[str] = set()
        self._last_modified: Dict[str, float] = {}
        self.stats = IndexStats()
        
        # Configure directory walker
        default_ignore = ['.git', '__pycache__', '.pytest_cache', '.mypy_cache']
        self.ignore_patterns = default_ignore + (ignore_patterns or [])
        self.walker = DirectoryWalker(self.ignore_patterns)
        
    def build_index(self, show_progress: bool = True) -> IndexStats:
        """
        Build the complete project index.
        
        Args:
            show_progress: Whether to show progress bars
            
        Returns:
            IndexStats with information about the indexing process
        """
        start_time = time.time()
        self.stats = IndexStats()
        
        # Discover Python files
        python_files = list(self.walker.walk(str(self.root_path)))
        self.stats.total_files = len(python_files)
        
        # Index each file
        if show_progress:
            file_iterator = tqdm(python_files, desc="Indexing files")
        else:
            file_iterator = python_files
            
        for file_path in file_iterator:
            if show_progress and hasattr(file_iterator, 'set_description'):
                file_iterator.set_description(f"Indexing {Path(file_path).name}")
            
            self._index_file(str(file_path))
            
        # Build import graph
        if show_progress:
            print("Building import graph...")
        self.import_graph = build_import_graph(str(self.root_path))
        
        # Calculate final statistics
        self.stats.index_time_seconds = time.time() - start_time
        self.stats.total_symbols = len(self.symbol_table._symbols)
        self.stats.total_functions = len(self.symbol_table.find_by_type(SymbolType.FUNCTION))
        self.stats.total_classes = len(self.symbol_table.find_by_type(SymbolType.CLASS))
        
        return self.stats
        
    def _index_file(self, file_path: str) -> None:
        """Index a single file."""
        try:
            # Check if already indexed and up to date
            stat = os.stat(file_path)
            mtime = stat.st_mtime
            
            if (file_path in self._indexed_files and 
                file_path in self._last_modified and
                self._last_modified[file_path] >= mtime):
                return  # Already up to date
                
            # Parse file
            result = parse_file(file_path)
            if not result.tree:
                if result.errors:
                    self.stats.errors.append((file_path, str(result.errors[0])))
                return
                
            # Extract symbols
            symbols = extract_symbols(result.tree, file_path)
            
            # Update symbol table
            if file_path in self._file_symbols:
                # Remove old symbols
                old_symbols = self._file_symbols[file_path]
                # TODO: Add remove_symbols method to SymbolTable
                
            self._file_symbols[file_path] = symbols
            self.symbol_table.add_symbols(symbols)
            
            # Track state
            self._indexed_files.add(file_path)
            self._last_modified[file_path] = mtime
            
        except Exception as e:
            self.stats.errors.append((file_path, str(e)))
            
    def find_references(self, symbol_name: str, show_progress: bool = False) -> List[Reference]:
        """
        Find all references to a symbol across the project.
        
        Args:
            symbol_name: Name of the symbol to find references for
            show_progress: Whether to show progress during search
            
        Returns:
            List of Reference objects
        """
        references = []
        
        files_to_search = list(self._indexed_files)
        if show_progress:
            from tqdm import tqdm
            file_iterator = tqdm(files_to_search, desc=f"Finding references to '{symbol_name}'")
        else:
            file_iterator = files_to_search
            
        for file_path in file_iterator:
            refs = find_references(file_path, {symbol_name})
            references.extend(refs)
            
        return references
        
    def find_definition(self, symbol_name: str, from_file: Optional[str] = None) -> Optional[DefinitionResult]:
        """
        Find the definition of a symbol.
        
        Args:
            symbol_name: Name of the symbol to find
            from_file: Optional file path to use for context (for relative imports)
            
        Returns:
            DefinitionResult if found, None otherwise
        """
        # Try direct lookup first
        results = self.symbol_table.find_by_name(symbol_name)
        
        if results:
            # If we have context, try to find the most relevant result
            if from_file and len(results) > 1:
                # TODO: Use import information to resolve the correct symbol
                pass
                
            # Return the first result
            symbol = results[0].symbol
            return DefinitionResult.from_symbol(symbol)
            
        # Try qualified name lookup
        if '.' in symbol_name:
            qualified_results = self.symbol_table.find_by_qualified_name(symbol_name)
            if qualified_results:
                symbol = qualified_results[0].symbol
                return DefinitionResult.from_symbol(symbol)
                
        return None
        
    def get_symbol(self, qualified_name: str) -> Optional[Symbol]:
        """Get a symbol by its fully qualified name."""
        results = self.symbol_table.find_by_qualified_name(qualified_name)
        return results[0].symbol if results else None
        
    def get_all_symbols(self) -> List[Symbol]:
        """Get all indexed symbols."""
        return self.symbol_table._symbols.copy()
        
    def get_symbols_in_file(self, file_path: str) -> List[Symbol]:
        """Get all symbols defined in a specific file."""
        abs_path = str(Path(file_path).resolve())
        return self._file_symbols.get(abs_path, [])
        
    def get_import_graph(self) -> Optional[ImportGraph]:
        """Get the project's import graph."""
        return self.import_graph
        
    def refresh(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Refresh the index for specific files or the entire project.
        
        Args:
            file_paths: Optional list of files to refresh. If None, refreshes
                       only modified files.
        """
        if file_paths:
            # Refresh specific files
            for file_path in file_paths:
                self._index_file(file_path)
        else:
            # Refresh modified files
            for file_path in self._indexed_files:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    if stat.st_mtime > self._last_modified.get(file_path, 0):
                        self._index_file(file_path)
                        
    def get_stats(self) -> IndexStats:
        """Get current index statistics."""
        return self.stats
        
    def clear(self) -> None:
        """Clear the entire index."""
        self.symbol_table = SymbolTable()
        self.import_graph = None
        self._file_symbols.clear()
        self._indexed_files.clear()
        self._last_modified.clear()
        self.stats = IndexStats()