#!/usr/bin/env python3
"""Test the unified API with our AST library."""

from pathlib import Path
from astlib import ASTLib

# Test the unified API
print("=== Testing Unified API ===\n")

# Initialize with the AST library itself
ast_lib = ASTLib(".", lazy=False)  # Index immediately

# Test 1: Find definition
print("1. Finding definitions:")
definitions = ["parse_file", "SymbolTable", "build_import_graph"]
for name in definitions:
    result = ast_lib.find_definition(name)
    if result:
        print(f"  {name}: {result.file_path}:{result.line}")
    else:
        print(f"  {name}: Not found")

# Test 2: Get project statistics
print("\n2. Project Analysis:")
analysis = ast_lib.analyze_project()
print(f"  Total files: {analysis.total_files}")
print(f"  Total symbols: {analysis.total_symbols}")
print(f"  Total functions: {analysis.total_functions}")
print(f"  Total classes: {analysis.total_classes}")
print(f"  Circular imports: {len(analysis.circular_imports)}")
print(f"  Import depth: {analysis.import_depth}")
print(f"  Analysis time: {analysis.analysis_time:.2f}s")

# Test 3: Find circular imports
print("\n3. Circular Imports:")
circles = ast_lib.find_circular_imports()
if circles:
    for circle in circles:
        print(f"  {' -> '.join(circle.cycle)}")
else:
    print("  No circular imports found!")

# Test 4: Get dependencies
print("\n4. Module Dependencies:")
test_modules = ["astlib.parser", "astlib.symbols", "astlib.api"]
for module in test_modules:
    deps = ast_lib.get_dependencies(module)
    if deps:
        print(f"  {module} depends on: {', '.join(deps[:3])}{'...' if len(deps) > 3 else ''}")

# Test 5: Analyze a specific file
print("\n5. File Analysis:")
file_analysis = ast_lib.analyze_file("astlib/parser.py")
print(f"  File: {file_analysis.file_path}")
print(f"  Success: {file_analysis.success}")
print(f"  Symbols: {len(file_analysis.symbols)}")
print(f"  Imports: {len(file_analysis.imports.imports)}")

# Test 6: Get symbols from a file
print("\n6. Symbols in parser.py:")
symbols = ast_lib.get_symbols("astlib/parser.py")
# Show first 5 functions
functions = [s for s in symbols if s.type.value == 'function'][:5]
for func in functions:
    print(f"  - {func.name} (line {func.position.line})")

# Test 7: Performance
print("\n7. Cache Performance:")
ast_lib.clear_cache()
import time

# First access (cold)
start = time.time()
ast_lib.find_definition("parse_file")
cold_time = time.time() - start

# Second access (warm)
start = time.time()
ast_lib.find_definition("parse_file")
warm_time = time.time() - start

print(f"  Cold lookup: {cold_time*1000:.2f}ms")
print(f"  Warm lookup: {warm_time*1000:.2f}ms")
print(f"  Speedup: {cold_time/warm_time:.1f}x")

print("\n=== Unified API Test Complete ===")
print("The API provides a clean interface to all AST operations!")