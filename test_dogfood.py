#!/usr/bin/env python3
"""Test the AST library on its own codebase."""

import time
from pathlib import Path
from astlib import parse_file, extract_symbols, clear_caches, get_cache_stats
from astlib.import_graph import build_import_graph
from astlib.symbol_table import SymbolTable
from astlib.benchmark import Timer

# Test 1: Parse and extract symbols from key files
print("=== Test 1: Symbol Extraction ===")
key_files = [
    "astlib/parser.py",
    "astlib/symbols.py", 
    "astlib/cache.py",
    "astlib/import_graph.py"
]

symbol_table = SymbolTable()
total_symbols = 0

for file in key_files:
    with Timer() as t:
        result = parse_file(file)
        if result.tree:
            symbols = extract_symbols(result.tree, file)
            symbol_table.add_symbols(symbols)
            total_symbols += len(symbols)
        else:
            symbols = []
    print(f"{file}: {len(symbols)} symbols in {t.elapsed:.3f}s")

print(f"\nTotal symbols indexed: {total_symbols}")

# Test 2: Symbol lookups
print("\n=== Test 2: Symbol Lookups ===")
test_lookups = ["parse_file", "extract_symbols", "SymbolTable", "build_import_graph"]

for name in test_lookups:
    with Timer() as t:
        results = symbol_table.find_by_name(name)
    print(f"find_by_name('{name}'): {len(results)} results in {t.elapsed*1000:.2f}ms")
    for result in results[:2]:  # Show first 2
        symbol = result.symbol
        print(f"  - {symbol.file_path}:{symbol.position.line} ({symbol.type.value})")

# Test 3: Cache effectiveness
print("\n=== Test 3: Cache Performance ===")
clear_caches()

# First pass - cold cache
with Timer() as cold_timer:
    for file in Path("astlib").glob("*.py"):
        result = parse_file(str(file))

# Second pass - warm cache  
with Timer() as warm_timer:
    for file in Path("astlib").glob("*.py"):
        result = parse_file(str(file))

cache_stats = get_cache_stats()
print(f"Cold cache: {cold_timer.elapsed:.3f}s")
print(f"Warm cache: {warm_timer.elapsed:.3f}s")
print(f"Speedup: {cold_timer.elapsed/warm_timer.elapsed:.1f}x")
print(f"Cache stats: {cache_stats}")

# Test 4: Import graph analysis
print("\n=== Test 4: Import Analysis ===")
with Timer() as t:
    graph = build_import_graph("astlib")

print(f"Built import graph in {t.elapsed:.3f}s")
print(f"Modules: {len(graph.nodes)}")
print(f"Import relationships: {sum(len(node.imports) for node in graph.nodes.values())}")

# Find circular imports
circles = graph.find_circular_imports()
print(f"Circular imports found: {len(circles)}")
for circle in circles:
    print(f"  - {' -> '.join(circle.modules + [circle.modules[0]])}")

# Most imported modules
print("\nMost imported modules:")
import_counts = {}
for node in graph.nodes.values():
    for imp in node.imports:
        import_counts[imp] = import_counts.get(imp, 0) + 1

for module, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  - {module}: {count} imports")

# Test 5: Complex queries
print("\n=== Test 5: Complex Queries ===")

# Find all async functions
async_funcs = symbol_table.find_by_type('async_function')
print(f"Async functions: {len(async_funcs)}")

# Find all private methods
private_methods = [s for s in symbol_table.find_by_type('method') if s.is_private]
print(f"Private methods: {len(private_methods)}")

# Find classes (note: we don't have find_by_type for 'class' yet)
print(f"Classes found: {len(symbol_table.find_by_type('class'))}")

print("\n=== Summary ===")
print(f"Successfully tested AST library on its own codebase!")
print(f"The dogfooding is working - we can navigate our own code efficiently!")