"""
AST - LLM Codebase Navigation Library

Fast Python AST analysis and symbol extraction for LLM-powered code navigation.
"""

from .parser import (
    parse_file,
    ParseResult,
    ParseError
)

from .symbols import (
    Symbol,
    SymbolType,
    Position,
    SymbolExtractor,
    extract_symbols
)

from .symbol_table import (
    SymbolTable,
    SymbolQuery,
    LookupResult
)

from .cache import (
    get_cache_manager,
    clear_caches,
    get_cache_stats
)

from .imports import (
    ImportType,
    ImportInfo,
    ModuleImports,
    analyze_imports,
    extract_imports,
    find_all_imports
)

from .import_graph import (
    ImportNode,
    CircularImport,
    ImportGraph,
    build_import_graph
)

from .indexer import (
    ProjectIndex,
    IndexStats,
    DefinitionResult
)

from .api import (
    ASTLib,
    Reference,
    FileAnalysis,
    ProjectAnalysis
)

__version__ = "0.1.0"

__all__ = [
    # Parser
    "parse_file",
    "ParseResult",
    "ParseError",
    
    # Symbols
    "Symbol",
    "SymbolType", 
    "Position",
    "SymbolExtractor",
    "extract_symbols",
    
    # Symbol Table
    "SymbolTable",
    "SymbolQuery",
    "LookupResult",
    
    # Cache
    "get_cache_manager",
    "clear_caches", 
    "get_cache_stats",
    
    # Imports
    "ImportType",
    "ImportInfo",
    "ModuleImports",
    "analyze_imports",
    "extract_imports",
    "find_all_imports",
    
    # Import Graph
    "ImportNode",
    "CircularImport",
    "ImportGraph",
    "build_import_graph",
    
    # Indexer
    "ProjectIndex",
    "IndexStats",
    "DefinitionResult",
    
    # Main API
    "ASTLib",
    "Reference",
    "FileAnalysis",
    "ProjectAnalysis",
]