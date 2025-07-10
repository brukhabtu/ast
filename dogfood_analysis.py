#!/usr/bin/env python3
"""
Dogfood Analysis - Using AST lib to analyze itself
"""

from astlib import ASTLib
from collections import defaultdict
import json

def analyze_ast_library():
    """Use AST library to analyze its own codebase"""
    
    print("=== AST Library Self-Analysis ===\n")
    
    # Initialize on our own codebase
    ast_lib = ASTLib("./astlib")
    
    # 1. Project Overview
    print("📊 Project Overview:")
    analysis = ast_lib.analyze_project()
    print(f"  Files analyzed: {analysis.total_files}")
    print(f"  Total symbols: {analysis.total_symbols}")
    print(f"  Unique imports: {analysis.unique_imports}")
    print(f"  Circular imports: {len(analysis.circular_imports)}")
    print()
    
    # 2. File Breakdown
    print("📁 File Analysis:")
    for file_info in sorted(analysis.files, key=lambda x: x.file_path):
        print(f"  {file_info.file_path}:")
        print(f"    - Symbols: {file_info.symbol_count}")
        print(f"    - Imports: {file_info.import_count}")
    print()
    
    # 3. Main Classes
    print("🏗️ Core Classes:")
    main_classes = ast_lib.find_symbols("*", symbol_type="class")
    for cls in sorted(main_classes, key=lambda x: x.name):
        print(f"  {cls.name} ({cls.file_path}:{cls.line})")
    print()
    
    # 4. Entry Points
    print("🚀 Main Entry Points:")
    main_funcs = ast_lib.find_symbols("main", symbol_type="function")
    for func in main_funcs:
        print(f"  {func.name} in {func.file_path}:{func.line}")
    print()
    
    # 5. Test Coverage
    print("🧪 Test Functions:")
    test_funcs = ast_lib.find_symbols("test_*", symbol_type="function")
    print(f"  Total test functions: {len(test_funcs)}")
    
    # Group by file
    tests_by_file = defaultdict(list)
    for test in test_funcs:
        tests_by_file[test.file_path].append(test.name)
    
    for file_path, tests in sorted(tests_by_file.items()):
        print(f"  {file_path}: {len(tests)} tests")
    print()
    
    # 6. Import Dependencies
    print("📦 Import Graph:")
    imports = ast_lib.get_import_graph()
    
    # Find most imported modules
    import_counts = defaultdict(int)
    for module, deps in imports.items():
        for dep in deps:
            import_counts[dep] += 1
    
    print("  Most imported modules:")
    for module, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {module}: imported by {count} modules")
    print()
    
    # 7. Call Graph Stats
    print("📞 Call Graph Statistics:")
    call_stats = ast_lib.get_call_graph_stats()
    print(f"  Total functions: {call_stats.total_functions}")
    print(f"  Total calls: {call_stats.total_calls}")
    print(f"  Avg calls per function: {call_stats.avg_calls_per_function:.2f}")
    print(f"  Max call depth: {call_stats.max_depth}")
    print(f"  Recursive functions: {len(call_stats.recursive_functions)}")
    
    if call_stats.most_called:
        print("\n  Most called functions:")
        for func, count in call_stats.most_called[:5]:
            print(f"    - {func}: {count} calls")
    
    if call_stats.most_calling:
        print("\n  Functions with most calls:")
        for func, count in call_stats.most_calling[:5]:
            print(f"    - {func}: {count} outgoing calls")
    print()
    
    # 8. Circular Dependencies
    if analysis.circular_imports:
        print("⚠️ Circular Imports Detected:")
        for cycle in analysis.circular_imports:
            print(f"  {' -> '.join(cycle.cycle)}")
    else:
        print("✅ No circular imports detected!")
    
    # 9. Find key integration points
    print("\n🔗 Key Integration Points:")
    
    # Find ASTLib methods
    ast_lib_def = ast_lib.find_definition("ASTLib")
    if ast_lib_def:
        print(f"\n  ASTLib class at {ast_lib_def.file_path}:{ast_lib_def.line}")
        
        # Show its public methods
        ast_lib_methods = ast_lib.find_symbols("*", symbol_type="method")
        public_methods = [m for m in ast_lib_methods 
                         if m.file_path == ast_lib_def.file_path 
                         and not m.name.startswith("_")]
        
        print("  Public API methods:")
        for method in sorted(public_methods, key=lambda x: x.line):
            print(f"    - {method.name}")

if __name__ == "__main__":
    analyze_ast_library()