"""
Result types and dataclasses for the AST library API.

This module defines the data structures returned by the unified API,
providing a consistent interface for all API operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum, auto

from .symbols import Symbol, SymbolType, Position


class ErrorLevel(Enum):
    """Severity levels for errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class APIError:
    """Represents an error that occurred during API operation."""
    message: str
    level: ErrorLevel = ErrorLevel.ERROR
    file_path: Optional[Path] = None
    line: Optional[int] = None
    column: Optional[int] = None
    error_type: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.level.value.upper()}] {self.message}"]
        if self.file_path:
            parts.append(f"in {self.file_path}")
        if self.line:
            parts.append(f"at line {self.line}")
            if self.column:
                parts.append(f"column {self.column}")
        return " ".join(parts)


@dataclass
class Reference:
    """Represents a reference to a symbol in code."""
    symbol_name: str
    file_path: Path
    position: Position
    context: Optional[str] = None  # Line of code containing the reference
    reference_type: str = "usage"  # "usage", "import", "definition", etc.
    
    @property
    def location_str(self) -> str:
        """Get location as string for display."""
        return f"{self.file_path}:{self.position.line}:{self.position.column}"


@dataclass
class DefinitionResult:
    """Result of finding a symbol definition."""
    symbol: Optional[Symbol] = None
    found: bool = False
    errors: List[APIError] = field(default_factory=list)
    search_paths: List[Path] = field(default_factory=list)
    lookup_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if the operation was successful."""
        return self.found and self.symbol is not None
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


@dataclass
class ReferencesResult:
    """Result of finding references to a symbol."""
    references: List[Reference] = field(default_factory=list)
    errors: List[APIError] = field(default_factory=list)
    files_searched: int = 0
    search_time_ms: float = 0.0
    
    @property
    def count(self) -> int:
        """Get number of references found."""
        return len(self.references)
    
    @property
    def success(self) -> bool:
        """Check if the operation completed without critical errors."""
        return not any(e.level == ErrorLevel.CRITICAL for e in self.errors)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module_name: str
    imported_names: List[str] = field(default_factory=list)  # Empty for 'import module'
    alias: Optional[str] = None
    is_relative: bool = False
    level: int = 0  # For relative imports (number of dots)
    position: Position = field(default_factory=Position)
    
    @property
    def import_str(self) -> str:
        """Get import as string."""
        if self.imported_names:
            names = ", ".join(self.imported_names)
            base = f"from {self.module_name} import {names}"
        else:
            base = f"import {self.module_name}"
        
        if self.alias:
            base += f" as {self.alias}"
        
        return base


@dataclass
class ImportGraph:
    """Graph of imports in a file or project."""
    file_path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    imported_by: List[Path] = field(default_factory=list)  # Files that import this file
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # module -> [symbols]
    
    @property
    def external_imports(self) -> List[ImportInfo]:
        """Get imports from external packages."""
        return [imp for imp in self.imports if not imp.is_relative]
    
    @property
    def internal_imports(self) -> List[ImportInfo]:
        """Get internal/relative imports."""
        return [imp for imp in self.imports if imp.is_relative]


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: Path
    symbols: List[Symbol] = field(default_factory=list)
    imports: ImportGraph = field(default_factory=lambda: ImportGraph(Path()))
    errors: List[APIError] = field(default_factory=list)
    parse_time_ms: float = 0.0
    line_count: int = 0
    
    @property
    def functions(self) -> List[Symbol]:
        """Get all function symbols."""
        return [s for s in self.symbols if s.type in (SymbolType.FUNCTION, SymbolType.ASYNC_FUNCTION)]
    
    @property
    def classes(self) -> List[Symbol]:
        """Get all class symbols."""
        return [s for s in self.symbols if s.type == SymbolType.CLASS]
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return not any(e.level == ErrorLevel.CRITICAL for e in self.errors)


@dataclass
class ProjectStats:
    """Statistics about a project."""
    total_files: int = 0
    total_lines: int = 0
    total_symbols: int = 0
    symbols_by_type: Dict[str, int] = field(default_factory=dict)
    files_with_errors: int = 0
    parse_errors: int = 0


@dataclass
class ProjectAnalysis:
    """Complete analysis results for a project."""
    root_path: Path
    files: Dict[Path, FileAnalysis] = field(default_factory=dict)
    symbol_table: Optional[Any] = None  # Will be SymbolTable instance
    errors: List[APIError] = field(default_factory=list)
    stats: ProjectStats = field(default_factory=ProjectStats)
    analysis_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return not any(e.level == ErrorLevel.CRITICAL for e in self.errors)
    
    def get_file(self, path: Union[str, Path]) -> Optional[FileAnalysis]:
        """Get analysis for a specific file."""
        path = Path(path) if isinstance(path, str) else path
        # Try both absolute and relative paths
        if path in self.files:
            return self.files[path]
        
        # Try resolving relative to root
        full_path = self.root_path / path
        if full_path in self.files:
            return self.files[full_path]
        
        return None


@dataclass
class SymbolSearchResult:
    """Result of searching for symbols."""
    symbols: List[Symbol] = field(default_factory=list)
    errors: List[APIError] = field(default_factory=list)
    total_matches: int = 0
    files_searched: int = 0
    search_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return not any(e.level == ErrorLevel.CRITICAL for e in self.errors)
    
    def __iter__(self):
        """Allow iterating over symbols directly."""
        return iter(self.symbols)
    
    def __len__(self):
        """Get number of symbols found."""
        return len(self.symbols)


@dataclass
class APIResponse:
    """Generic API response wrapper."""
    success: bool
    data: Optional[Any] = None
    errors: List[APIError] = field(default_factory=list)
    warnings: List[APIError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success_response(cls, data: Any, **metadata) -> 'APIResponse':
        """Create a successful response."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(cls, error_msg: str, **metadata) -> 'APIResponse':
        """Create an error response."""
        error = APIError(message=error_msg, level=ErrorLevel.ERROR)
        return cls(success=False, errors=[error], metadata=metadata)


@dataclass
class CircularImport:
    """Information about a circular import dependency."""
    cycle: List[str]  # List of modules forming the cycle
    
    @property
    def cycle_length(self) -> int:
        """Length of the circular import chain."""
        return len(self.cycle)
        
    def __str__(self) -> str:
        """String representation of the cycle."""
        return " -> ".join(self.cycle) + " -> " + self.cycle[0]