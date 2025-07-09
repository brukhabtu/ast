# Layered Architecture - AST Library

## Minimal Layer Design

Our architecture naturally splits into 5 clean layers, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 5: Interface Layer                  │
│                  CLI Commands, Public API                    │
├─────────────────────────────────────────────────────────────┤
│                    Layer 4: Analysis Layer                   │
│            Import Graph, References, Patterns                │
├─────────────────────────────────────────────────────────────┤
│                    Layer 3: Index Layer                      │
│            Symbol Table, Cross-file Index, Cache             │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Extraction Layer                 │
│              Symbol Extraction, Import Parsing               │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Parse Layer                      │
│                AST Parsing, Visitor, Walker                  │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Parse Layer (Foundation) 🏗️

**Purpose**: Convert Python source code into traversable AST

**Components**:
- `parser.py` - Parse files with error recovery
- `visitor.py` - Enhanced AST traversal
- `walker.py` - Directory/file discovery

**Key Interfaces**:
```python
# Input: File path or source code
# Output: AST tree
def parse_file(filepath: str) -> ParseResult:
    """Returns AST or partial AST with errors"""

class NodeVisitor:
    """Traverse AST with context tracking"""
```

**Dependencies**: Python stdlib only (`ast`, `pathlib`)

## Layer 2: Extraction Layer 🔍

**Purpose**: Extract semantic information from AST

**Components**:
- `symbols.py` - Extract functions, classes, variables
- `imports.py` - Extract and classify imports
- `types.py` - Data structures (Symbol, ImportInfo)

**Key Interfaces**:
```python
# Input: AST tree
# Output: Semantic information
def extract_symbols(tree: ast.AST) -> List[Symbol]:
    """Extract all symbols from AST"""

def extract_imports(tree: ast.AST) -> ModuleImports:
    """Extract all imports from AST"""
```

**Dependencies**: Layer 1 (Parse)

## Layer 3: Index Layer 📚

**Purpose**: Store and retrieve information efficiently

**Components**:
- `symbol_table.py` - Multi-index symbol storage
- `cache.py` - Transparent caching layer
- `indexer.py` - Cross-file indexing (TODO)

**Key Interfaces**:
```python
# Input: Symbols/data
# Output: Fast lookups
class SymbolTable:
    def add_symbols(symbols: List[Symbol])
    def find_by_name(name: str) -> List[LookupResult]

class CacheManager:
    """Transparent caching for expensive operations"""
```

**Dependencies**: Layer 2 (Extraction)

## Layer 4: Analysis Layer 🧠

**Purpose**: Higher-level code analysis and relationships

**Components**:
- `import_graph.py` - Dependency analysis
- `references.py` - Find symbol usages (TODO)
- `patterns.py` - Pattern matching (TODO)
- `call_graph.py` - Function relationships (TODO)

**Key Interfaces**:
```python
# Input: Indexed data
# Output: Relationships and insights
def build_import_graph(directory: str) -> ImportGraph:
    """Build dependency graph"""

class ImportGraph:
    def find_circular_imports() -> List[CircularImport]
    def get_transitive_dependencies(module: str) -> Set[str]
```

**Dependencies**: Layers 1-3

## Layer 5: Interface Layer 🎯

**Purpose**: User-facing interfaces (CLI, API)

**Components**:
- `cli.py` - Command-line interface
- `api.py` - Unified Python API (TODO)
- `formatter.py` - Output formatting

**Key Interfaces**:
```python
# Input: User commands
# Output: Formatted results
def main():
    """CLI entry point"""

class ASTLib:
    """High-level API for all operations"""
```

**Dependencies**: All layers

## Cross-Cutting Concerns 🔄

**Performance** (`benchmark.py`)
- Available to all layers
- Not a layer itself

**Testing** (`tests/`)
- Tests each layer independently
- Integration tests cross layers

## Data Flow Example

```
User Request: "Find function parse_file"
    ↓
Layer 5: CLI parses command
    ↓
Layer 3: Check cache
    ↓ (cache miss)
Layer 1: Walk directories, find Python files
    ↓
Layer 1: Parse each file → AST
    ↓
Layer 2: Extract symbols from AST
    ↓
Layer 3: Store in symbol table, cache results
    ↓
Layer 5: Format and display results
```

## Layer Independence

Each layer can be used independently:

```python
# Just parsing
from astlib.parser import parse_file
result = parse_file("example.py")

# Just extraction  
from astlib.symbols import extract_symbols
symbols = extract_symbols(ast_tree)

# Just indexing
from astlib.symbol_table import SymbolTable
table = SymbolTable()
table.add_symbols(symbols)

# Just analysis
from astlib.import_graph import ImportGraph
graph = ImportGraph()
graph.add_import("mod1", "mod2")
```

## Benefits of This Architecture

1. **Clear Dependencies** - Each layer only depends on layers below
2. **Independent Testing** - Test each layer in isolation
3. **Easy to Extend** - Add new analysis without touching parse layer
4. **Performance** - Cache at the right layer (Layer 3)
5. **Reusability** - Use any layer independently

## Future Layer Additions

**Potential Layer 6: Intelligence Layer**
- Code suggestions
- Refactoring recommendations
- Pattern learning

**Potential Layer 2.5: Semantic Layer**
- Type inference
- Scope resolution
- Name binding

## Summary

The 5-layer architecture provides:
- **Separation of concerns** - Each layer has one job
- **Clear interfaces** - Well-defined contracts between layers
- **Testability** - Each layer can be tested independently
- **Flexibility** - Easy to add new functionality at the right layer
- **Performance** - Caching and indexing at the appropriate level

This minimal layering emerged naturally from the implementation and provides a clean, maintainable architecture.