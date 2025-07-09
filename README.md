# AST Library

A high-performance Python library for AST (Abstract Syntax Tree) analysis, designed to help LLMs and developers navigate codebases quickly and efficiently.

## Features

- **🚀 Fast Navigation**: Parse files in <50ms, symbol lookups in <0.1ms
- **🔍 Comprehensive Analysis**: Symbols, imports, references, and call graphs
- **💾 Smart Caching**: 10-14x performance improvement with transparent caching
- **🔄 Cross-file Navigation**: Resolve symbols and track dependencies across projects
- **🛡️ Error Recovery**: Parse and analyze even syntactically invalid code
- **📊 Rich CLI**: Multiple output formats (JSON, Markdown, Graphviz DOT)

## Installation

```bash
pip install -e .
```

## Quick Start

### Command Line

```bash
# Find all functions containing "test"
ast find-function "test*"

# Find where a symbol is defined
ast find-symbol MyClass

# Show what functions call 'process_data'
ast show-callers process_data

# Find circular imports
ast show-imports --circular

# Generate call graph statistics
ast call-stats --export-dot callgraph.dot
```

### Python API

```python
from astlib import ASTLib

# Initialize on a project
ast_lib = ASTLib("./my_project")

# Find symbol definition
definition = ast_lib.find_definition("MyClass")
print(f"Found at {definition.file_path}:{definition.line}")

# Find all references
refs = ast_lib.find_references("process_data")
for ref in refs:
    print(f"{ref.file_path}:{ref.line} - {ref.reference_type}")

# Analyze imports
circular = ast_lib.find_circular_imports()
for cycle in circular:
    print(f"Circular import: {' -> '.join(cycle.cycle)}")

# Get call graph
stats = ast_lib.get_call_graph_stats()
print(f"Total functions: {stats.total_functions}")
print(f"Recursive functions: {len(stats.recursive_functions)}")
```

## Core Capabilities

### 1. Symbol Analysis
- Extract functions, classes, methods, and variables
- Track symbol hierarchies and relationships
- Find definitions and references across files

### 2. Import Analysis
- Map import dependencies
- Detect circular imports
- Generate import graphs

### 3. Call Graph Analysis
- Track function call relationships
- Find call chains between functions
- Detect recursive functions
- Generate call hierarchies

### 4. Code Navigation
- Cross-file symbol resolution
- Pattern-based searching with wildcards
- Context-aware symbol lookup

## Architecture

The library is organized in 5 clean layers:

1. **Parse Layer**: Fast AST parsing with error recovery
2. **Extraction Layer**: Symbol and import extraction
3. **Index Layer**: Efficient storage and retrieval
4. **Analysis Layer**: Call graphs and pattern matching
5. **Interface Layer**: CLI and Python API

## Performance

- **Parsing**: <50ms for typical files
- **Symbol Lookup**: <0.1ms with caching
- **Project Indexing**: ~1000 files/second
- **Memory Efficient**: Incremental updates, LRU caching

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
# All tests
pytest

# Run with coverage
pytest --cov=astlib

# Run benchmarks
python run_benchmarks.py
```

## Documentation

- [Architecture Overview](ARCHITECTURAL_PATTERNS.md)
- [CLI Usage Guide](CLI_USAGE.md) 
- [Testing Guide](TESTING_GUIDE.md)
- [Project Structure](PROJECT_STRUCTURE.md)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.